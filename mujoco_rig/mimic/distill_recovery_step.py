"""Diagnostic distillation of a native step teacher into the Stand actor shape."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .checkpoints import standing_passed
from .evaluate_stand import evaluate as evaluate_stand
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import EntryReference
from .recovery_step_motion import evaluate as evaluate_step
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_step_teacher_distillation_pilot_v2"


class StepStudent(torch.nn.Module):
    """Stand actor with an optional calibrated, information-preserving input path."""

    def __init__(self, checkpoint):
        super().__init__()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        obs = weights["_obs_norm._mean"]
        action = weights["_a_norm._mean"]
        self.register_buffer("obs_mean", obs.clone())
        self.register_buffer("obs_std", weights["_obs_norm._std"].clone())
        self.register_buffer("action_mean", action.clone())
        self.register_buffer("action_std", weights["_a_norm._std"].clone())
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(len(obs), 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), torch.nn.ReLU())
        self.mean = torch.nn.Linear(512, len(action))
        for destination, prefix in ((self.actor, "_model._actor_layers."),
                                    (self.mean, "_model._action_dist._mean_net.")):
            state = {key.removeprefix(prefix): value for key, value in weights.items()
                     if key.startswith(prefix)}
            destination.load_state_dict(state)

    def calibrate_features(self, observations):
        """Keep the pretrained path exact and expose channels saturated by Stand."""
        if hasattr(self, "feature_std"):
            raise ValueError("Feature calibration is immutable once installed")
        if (observations.ndim != 2 or observations.shape[1] != len(self.obs_mean)
                or not len(observations) or not torch.isfinite(observations).all()):
            raise ValueError("Require finite calibration observations matching the actor")
        scale = torch.maximum(self.obs_std,
                              (observations - self.obs_mean).abs().amax(dim=0) / 6.)
        self._install_features(scale)

    def _install_features(self, scale):
        first = self.actor[0]
        expanded = torch.nn.Linear(len(self.obs_mean) * 2, first.out_features,
                                   device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :len(self.obs_mean)].copy_(first.weight)
            expanded.bias.copy_(first.bias)
        self.actor[0] = expanded
        self.register_buffer("feature_std", scale.clone())

    def load_student_state(self, state):
        if "feature_std" in state and not hasattr(self, "feature_std"):
            self._install_features(state["feature_std"])
        self.load_state_dict(state)

    def forward(self, observation):
        normalized = ((observation - self.obs_mean) / self.obs_std).clamp(-10., 10.)
        if hasattr(self, "feature_std"):
            features = ((observation - self.obs_mean) / self.feature_std).clamp(-10., 10.)
            normalized = torch.cat((normalized, features), dim=-1)
        return (self.mean(self.actor(normalized)) * self.action_std
                + self.action_mean).clamp(-1., 1.)


class TrajectoryStudent(torch.nn.Module):
    """A feedforward motion with learned state feedback and a frozen Stand expert.

    Reference identity selects the expert; physical state never selects it.
    Feedback is zero on the demonstrated trajectory by construction.
    """

    def __init__(self, source, observations, actions, calibration):
        super().__init__()
        self.stand = source.requires_grad_(False)
        self.register_buffer("trajectory_observations", observations.clone())
        self.register_buffer("trajectory_actions", actions.clone())
        self.register_buffer("phase_step", observations[1, -1] - observations[0, -1])
        indices = self.indices(calibration)
        error = calibration[:, :217] - observations[indices, :217]
        self.register_buffer("error_scale", error.abs().amax(dim=0).clamp(min=.03) / 3.)
        self.register_buffer("frequencies", torch.arange(1, 9) * torch.pi)
        self.feedback = torch.nn.Sequential(
            torch.nn.Linear(217 + 17, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 128), torch.nn.Tanh(), torch.nn.Linear(128, 30))
        torch.nn.init.zeros_(self.feedback[-1].weight)
        torch.nn.init.zeros_(self.feedback[-1].bias)

    def indices(self, observation):
        return (observation[:, -1] / self.phase_step).round().long().clamp(
            0, len(self.trajectory_observations) - 1)

    def forward(self, observation):
        indices = self.indices(observation)
        nominal = self.trajectory_observations[indices]
        # The task's future reference is exogenous and identifies the selected skill.
        selected = (observation[:, 217:361] - nominal[:, 217:361]).abs().amax(dim=1) < .002
        error = ((observation[:, :217] - nominal[:, :217]) / self.error_scale).clamp(-10., 10.)
        phase = observation[:, -1:]
        angles = phase * self.frequencies
        encoding = torch.cat((phase, angles.sin(), angles.cos()), dim=1)
        correction = .5 * (self.feedback(torch.cat((error, encoding), dim=1))
                           - self.feedback(torch.cat((torch.zeros_like(error), encoding), dim=1))).tanh()
        action = (self.trajectory_actions[indices] + correction).clamp(-1., 1.)
        return torch.where(selected[:, None], action, self.stand(observation))

    def load_student_state(self, state):
        self.load_state_dict(state)


def load_student(checkpoint, source_checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = StepStudent(source_checkpoint)
    if "trajectory_observations" in state:
        observation = state["trajectory_observations"]
        source = TrajectoryStudent(source, observation, state["trajectory_actions"], observation)
    source.load_student_state(state)
    return source.eval()


def run(args):
    if (args.out.exists() or args.epochs < 1 or args.learning_rate <= 0
            or args.trajectory_feedback and (args.samples_run is None or args.calibrated_features)):
        raise ValueError("Require a new output directory and positive training settings")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    teacher_path = args.teacher_run / "report.json"
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    motion_path = args.teacher_run / "step_reference.npz"
    search = None
    if args.teacher_search is not None:
        search_path = args.teacher_search / "report.json"
        search = json.loads(search_path.read_text(encoding="utf-8"))
        samples_path = args.teacher_search / "best_teacher_samples.npz"
        native = search["best"]["native"]
        if (not native.get("touchdown_retained", False) or not native["passed"]
                or search["source_report_sha256"] != sha256(teacher_path)
                or search["source_motion_sha256"] != sha256(motion_path)
                or search["teacher_samples_sha256"] != sha256(samples_path)):
            raise ValueError("Require a passing, hash-bound native teacher search")
    else:
        samples_path = args.teacher_run / "teacher_samples.npz"
        native = teacher["native_evaluation"]
        if teacher["teacher_samples_sha256"] != sha256(samples_path):
            raise ValueError("Teacher samples hash mismatch")
    if (not native.get("touchdown_retained", False)
            or not native["passed"]
            or teacher["reference_sha256"] != sha256(motion_path)
            or teacher["model_sha256"] != sha256(rig.path)):
        raise ValueError("Require an intact native-passing teacher run")
    if args.samples_run is not None:
        dataset_report_path = args.samples_run / "report.json"
        dataset = json.loads(dataset_report_path.read_text(encoding="utf-8"))
        dataset_path = args.samples_run / "teacher_samples.npz"
        if (not dataset.get("recovery_gate_passed", False)
                or dataset.get("model_sha256") != sha256(rig.path)
                or dataset.get("teacher_report_sha256") != sha256(teacher_path)
                or search is not None and dataset.get("controller_search_sha256")
                != sha256(args.teacher_search / "report.json")
                or dataset.get("teacher_samples_sha256") != sha256(dataset_path)):
            raise ValueError("Require a passing, hash-bound disturbed teacher dataset")
        samples_path = dataset_path
    best = json.loads(args.initialize_from.with_name("best.json").read_text(encoding="utf-8"))
    contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    if (best["checkpoint_sha256"] != sha256(args.initialize_from)
            or contract["model_sha256"] != sha256(rig.path)
            or contract["action_mode"] != "reference_relative_target_pd"):
        raise ValueError("Source actor or contract mismatch")
    with np.load(samples_path, allow_pickle=False) as arrays:
        observations = torch.as_tensor(arrays["observation"], dtype=torch.float32)
        actions = torch.as_tensor(arrays["action"], dtype=torch.float32)
    if (observations.ndim != 2 or actions.ndim != 2
            or len(observations) != len(actions)
            or observations.shape[1] != contract["num_obs"]
            or actions.shape[1] != len(contract["actuator_names"])
            or not torch.isfinite(observations).all()
            or not torch.isfinite(actions).all()):
        raise ValueError("Teacher samples do not match the source actor contract")
    with np.load(motion_path, allow_pickle=False) as arrays:
        reference = EntryReference(arrays["qpos"], rig, float(arrays["dt"]))
    step = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    step.reference = reference
    step.episode_seconds = reference.duration
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                      8, "cpu", engine_factory=NativeDummyEngine,
                      control_mode="target_pd", contact_mode="support_v2")
    student = StepStudent(args.initialize_from)
    quiet_x = quiet_y = None
    if args.samples_run is not None:
        _, quiet_trace = evaluate_stand(quiet, student, capture_actions=True)
        quiet_x = torch.as_tensor(quiet_trace["observations"].reshape(-1, observations.shape[1]))
        quiet_y = torch.as_tensor(quiet_trace["actions"].reshape(-1, actions.shape[1]))
    if args.calibrated_features:
        calibration = (torch.cat((observations, quiet_x))
                       if quiet_x is not None else observations)
        student.calibrate_features(calibration)
    if args.trajectory_feedback:
        with np.load(samples_path, allow_pickle=False) as arrays:
            nominal = arrays["case_id"] == 0
        student = TrajectoryStudent(student, observations[nominal], actions[nominal], observations)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    student.to(device)
    train_x, train_y = observations.to(device), actions.to(device)
    if quiet_x is not None:
        quiet_x, quiet_y = quiet_x.to(device), quiet_y.to(device)
    validation = torch.arange(len(actions), device=device) % 5 == 0
    training = ~validation
    training_indices = torch.nonzero(training, as_tuple=True)[0]
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    history = []

    def review(epoch):
        actor = copy.deepcopy(student).cpu().eval()
        with torch.no_grad():
            validation_mse = float((actor(observations[::5]) - actions[::5]).square().mean())
            native, _, _ = evaluate_step(step, actor, teacher["requested_placement_m"],
                                         teacher["phase_seconds"]["touchdown"])
            standing, _ = evaluate_stand(quiet, actor)
        row = dict(epoch=epoch, validation_mse=validation_mse,
                   native={key: value for key, value in native.items() if key != "trace"},
                   quiet=standing, quiet_passed=standing_passed(standing))
        checkpoint = args.out / f"epoch-{epoch:04d}.pt"
        torch.save(actor.state_dict(), checkpoint)
        row["checkpoint_sha256"] = sha256(checkpoint)
        history.append(row)
        print(json.dumps(dict(epoch=epoch, validation_mse=validation_mse,
                              native_passed=native["passed"],
                              native_survival=native["survival_seconds"],
                              quiet_passed=row["quiet_passed"])), flush=True)
        return row

    args.out.mkdir(parents=True)
    initial = review(0)
    checkpoints = {0, args.epochs // 2, args.epochs}
    for epoch in range(1, args.epochs + 1):
        student.train()
        for indices in torch.randperm(len(training_indices), device=device).split(128):
            selection = training_indices[indices]
            loss = (student(train_x[selection]) - train_y[selection]).square().mean()
            if quiet_x is not None:
                quiet_indices = torch.randint(len(quiet_x), (len(selection),), device=device)
                loss += .5 * (student(quiet_x[quiet_indices]) - quiet_y[quiet_indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
        if epoch in checkpoints:
            review(epoch)
    candidate_path = args.out / "candidate.pt"
    torch.save(copy.deepcopy(student).cpu().state_dict(), candidate_path)
    selected = min((row for row in history if row["epoch"] > 0 and row["native"]["passed"]
                    and row["quiet_passed"]),
                   key=lambda row: row["validation_mse"], default=None)
    report = dict(schema=SCHEMA, teacher_report_sha256=sha256(teacher_path),
                  teacher_search_sha256=(sha256(args.teacher_search / "report.json")
                                         if search is not None else None),
                  teacher_samples_sha256=sha256(samples_path),
                  samples_run_sha256=(sha256(args.samples_run / "report.json")
                                      if args.samples_run is not None else None),
                  quiet_regularization_weight=.5 if args.samples_run is not None else 0.,
                  input_features=("trajectory_feedback_v1" if args.trajectory_feedback else
                                  "stand_plus_calibrated_all_v1"
                                  if args.calibrated_features else "frozen_stand"),
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  candidate_sha256=sha256(candidate_path), epochs=args.epochs,
                  learning_rate=args.learning_rate, seed=args.seed,
                  history=history, selected=selected,
                  passed=selected is not None, admitted_for_deployment=False,
                  scope="Native teacher imitation with closed-loop landing and quiet validation; no ball or deployment admission")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--teacher-search", type=Path)
    parser.add_argument("--samples-run", type=Path)
    parser.add_argument("--calibrated-features", action="store_true")
    parser.add_argument("--trajectory-feedback", action="store_true")
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=260926)
    run(parser.parse_args())

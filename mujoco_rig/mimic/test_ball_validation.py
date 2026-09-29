"""Coverage, contact evidence and policy-selection regressions for ball validation."""
from collections import Counter
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .ball_protocol import TARGET_BODIES, VALIDATION_EPISODES, VALIDATION_SCHEMA, validation_cases
from .checkpoints import BestCheckpoint, checkpoint_rank, ball_regressions, recovery_step_rank
from .evaluate_stand import evaluate
from .godot_ball import compare
from .runtime import activate
from .rig import Rig


class BallValidationTests(unittest.TestCase):
    def test_recovered_steps_cannot_outweigh_lost_recoveries_or_quiet(self):
        rows = [{**c, "hit": True, "survived": True, "recovered": True,
                 "recovered_with_step": False} for c in validation_cases()]
        baseline = self.metrics(rows)
        baseline["recovered_with_step"] = 0
        stepped = copy.deepcopy(baseline)
        stepped["cases"][0]["recovered_with_step"] = True
        stepped["recovered_with_step"] = 1
        self.assertGreater(recovery_step_rank(stepped), recovery_step_rank(baseline))
        self.assertLess(recovery_step_rank({**stepped, "quiet_standing_passed": False}), recovery_step_rank(baseline))
        fewer = {**stepped, "recovered_hits": stepped["recovered_hits"]-1}
        self.assertLess(recovery_step_rank(fewer), recovery_step_rank(baseline))
        with self.assertRaises(ValueError):
            recovery_step_rank({**stepped, "recovered_with_step": 2})

    @staticmethod
    def metrics(cases=None):
        if cases is None:
            cases = [{**c, "hit": True, "survived": False, "recovered": False} for c in validation_cases()]
        return dict(episodes=len(cases), successes=sum(c["survived"] for c in cases),
                    confirmed_hits=sum(c["hit"] for c in cases),
                    survived_hits=sum(c["hit"] and c["survived"] for c in cases),
                    recovered_hits=sum(c["recovered"] for c in cases), cases=cases,
                    quiet_standing_passed=True, evaluation_schema=VALIDATION_SCHEMA,
                    mean_survival_seconds=4., mean_root_tracking_error_m=.1)

    def test_every_body_gets_each_direction_and_speed(self):
        cases = validation_cases()
        counts = Counter((case["target_body"], case["direction"], case["speed"]) for case in cases)
        self.assertEqual(len(cases), 96)
        for body in TARGET_BODIES:
            for direction in range(4):
                for speed in (1., 2.5):
                    self.assertEqual(counts[body, direction, speed], 1)

    def test_cannot_silently_score_only_first_eight_worlds(self):
        task = SimpleNamespace(validation_cases=validation_cases(), get_num_envs=lambda: 8)
        # evaluate imports the pinned environment API before validating the count.
        self.activate()
        with self.assertRaisesRegex(ValueError, "one world per scored case"):
            evaluate(task, None)

    @staticmethod
    def activate():
        checkout = os.environ.get("MIMICKIT_PATH")
        if not checkout:
            raise unittest.SkipTest("Set MIMICKIT_PATH for native task integration")
        activate(Path(checkout))

    def test_native_schedule_and_returned_observation_match_protocol(self):
        self.activate()
        from envs.base_env import EnvMode
        from .ball_task import BallTask
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        task = BallTask(rig, root / "assets/stand_reference.npz", VALIDATION_EPISODES, "cpu", speed_max=2.5)
        task.set_mode(EnvMode.TEST)
        obs, _ = task.reset()
        torch.testing.assert_close(obs, task.observations())
        for i, spec in enumerate(validation_cases()):
            self.assertEqual(int(task.shot_target_body[i]), rig.model.body(spec["target_body"]).id - 1)
            self.assertEqual(float(task.shot_speeds[i]), spec["speed"])
            self.assertAlmostEqual(float(task.offset[i]), spec["phase"])
            np.testing.assert_allclose(task.directions[i].numpy(),
                                       [np.cos(spec["direction"] * np.pi / 2), np.sin(spec["direction"] * np.pi / 2), 0], atol=2e-7)
        task.steps[1] = 17
        task.reset(torch.tensor([0, 95]))
        self.assertEqual(int(task.steps[1]), 17)
        self.assertEqual(int(task.steps[95]), 0)

    def test_selection_requires_consistent_recovery_counts_and_suite(self):
        metrics = dict(episodes=96, successes=50, confirmed_hits=90, survived_hits=44, recovered_hits=20,
                       quiet_standing_passed=True, evaluation_schema=VALIDATION_SCHEMA,
                       mean_survival_seconds=4., mean_root_tracking_error_m=.1)
        self.assertGreater(checkpoint_rank({**metrics, "recovered_hits": 21}), checkpoint_rank(metrics))
        with self.assertRaises(ValueError):
            checkpoint_rank({**metrics, "recovered_hits": 45})
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {})
            save = lambda path: Path(path).write_bytes(b"actor")
            measured = self.metrics()
            best.consider(save, measured, iteration=0, samples=0)
            with self.assertRaisesRegex(ValueError, "different suites"):
                best.consider(save, {**measured, "evaluation_schema": "old"}, iteration=1, samples=1)

    def test_focused_training_keeps_full_validation_and_partial_reset_isolation(self):
        self.activate()
        from envs.base_env import EnvMode
        from .ball_task import BallTask
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        task = BallTask(rig, root / "assets/stand_reference.npz", 96, "cpu",
                        curriculum="front_back_torso_v1", reward_mode="recovery_v1")
        task.set_mode(EnvMode.TRAIN)
        task.reset()
        torso = {rig.model.body(n).id - 1 for n in ("Chest", "Spine", "Pelvis")}
        self.assertEqual(set(task.shot_target_body.tolist()), torso)
        self.assertTrue(bool((task.directions[:, 0].abs() >= np.cos(np.deg2rad(10))).all()))
        self.assertTrue(bool((task.directions[:, 0] > 0).any()))
        self.assertTrue(bool((task.directions[:, 0] < 0).any()))
        self.assertTrue(bool(task.shot_enabled.any() & ~task.shot_enabled.all()))
        peer = task.directions[1:].clone()
        task.steps[1] = 17
        task.reset(torch.tensor([0]))
        torch.testing.assert_close(task.directions[1:], peer, rtol=0, atol=0)
        self.assertEqual(int(task.steps[1]), 17)
        task.set_mode(EnvMode.TEST)
        task.reset()
        self.assertEqual(task.validation_cases, validation_cases(1., 2.))
        for i, case in enumerate(task.validation_cases):
            self.assertEqual(int(task.shot_target_body[i]), rig.model.body(case["target_body"]).id - 1)
            np.testing.assert_allclose(task.directions[i].numpy(),
                [np.cos(case["direction"] * np.pi / 2), np.sin(case["direction"] * np.pi / 2), 0], atol=2e-7)
        self.assertEqual(task.experiment_contract()["training_curriculum"]["schema"], "front_back_torso_v1")

    def test_front_back_review_excludes_misses_and_separates_quiet_and_delayed_falls(self):
        from .review_ball_recovery import review_cases, summarize
        cases = [dict(c, hit=False, survived=False, recovered=False, survival_seconds=1.)
                 for c in review_cases(30., 1 / 60)]
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c["quiet"] for c in cases), 8)
        self.assertTrue(all(c["trial_steps"] == 1800 for c in cases))
        cases[0].update(hit=True, survived=True, recovered=True, survival_seconds=30.)
        cases[1].update(hit=True, survival_seconds=10.)
        cases[2].update(survived=True, survival_seconds=30.)  # A miss is not recovery.
        cases[-1].update(survived=True, survival_seconds=30.)
        summary = summarize(dict(cases=cases))
        self.assertEqual(summary["confirmed_hits"], 2)
        self.assertEqual(summary["survived_hits"], 1)
        self.assertEqual(summary["recovered_hits"], 1)
        self.assertEqual(summary["delayed_falls_after_5s"], 1)
        self.assertEqual(summary["quiet_survived"], 1)
        self.assertEqual(summary["directions"]["forward"]["recovered"], 1)
        for seconds in (float("nan"), 4.):
            with self.assertRaises(ValueError):
                review_cases(seconds, 1 / 60)

    def test_mixed_curriculum_distribution_geometry_and_validation_isolation(self):
        self.activate()
        from envs.base_env import EnvMode
        from .ball_task import BallTask
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        with self.assertRaisesRegex(ValueError, "25% quiet"):
            BallTask(rig, root / "assets/stand_reference.npz", 96, "cpu",
                     curriculum="mixed_recovery_v1", quiet_fraction=.5)
        task = BallTask(rig, root / "assets/stand_reference.npz", 96, "cpu",
                        curriculum="mixed_recovery_v1", reward_mode="recovery_v1")
        task.set_mode(EnvMode.TRAIN)
        totals = torch.zeros(3)
        broad_targets = set()
        torso = {rig.model.body(n).id - 1 for n in ("Chest", "Spine", "Pelvis")}
        broad_side_shots = 0
        for _ in range(20):
            task.reset()
            focused, broad, quiet = task.focused_shot, task.shot_enabled & ~task.focused_shot, ~task.shot_enabled
            totals += torch.tensor([focused.sum(), broad.sum(), quiet.sum()])
            self.assertFalse(bool((focused & quiet).any()))
            self.assertTrue(set(task.shot_target_body[focused].tolist()) <= torso)
            self.assertTrue(bool((task.directions[focused, 0].abs() >= np.cos(np.deg2rad(10))).all()))
            broad_targets.update(task.shot_target_body[broad].tolist())
            broad_side_shots += int((task.directions[broad, 1].abs() > .8).sum())
        torch.testing.assert_close(totals / totals.sum(), torch.tensor([.5, .25, .25]), atol=.05, rtol=0)
        self.assertEqual(broad_targets, set(task.target_indices.tolist()))
        self.assertGreater(broad_side_shots, 0)
        peer = torch.stack((task.shot_enabled[1:], task.focused_shot[1:])).clone()
        task.reset(torch.tensor([0]))
        torch.testing.assert_close(torch.stack((task.shot_enabled[1:], task.focused_shot[1:])), peer)
        task.set_mode(EnvMode.TEST)
        task.reset()
        self.assertEqual(task.validation_cases, validation_cases(1., 2.))
        self.assertTrue(bool(task.shot_enabled.all()))
        self.assertFalse(bool(task.focused_shot.any()))
        for i, case in enumerate(task.validation_cases):
            self.assertEqual(int(task.shot_target_body[i]), rig.model.body(case["target_body"]).id - 1)
            np.testing.assert_allclose(task.directions[i].numpy(),
                [np.cos(case["direction"] * np.pi / 2), np.sin(case["direction"] * np.pi / 2), 0], atol=2e-7)
        self.assertEqual(task.experiment_contract()["training_curriculum"]["episode_probabilities"],
                         dict(focused_torso=.5, all_directions=.25, quiet=.25))

    def test_front_back_review_reset_applies_quiet_and_long_horizon(self):
        self.activate()
        from envs.base_env import EnvMode
        from .ball_task import BallTask
        from .review_ball_recovery import make_task
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        contract = BallTask(rig, root / "assets/stand_reference.npz", 8, "cpu").observation_contract()
        task = make_task(rig, root / "assets", contract, 30., 1., 2.)
        task.set_mode(EnvMode.TEST)
        obs, _ = task.reset()
        torch.testing.assert_close(obs, task.observations())
        self.assertEqual(task.episode_seconds, 30.)
        self.assertTrue(bool(task.shot_enabled[:12].all()))
        self.assertFalse(bool(task.shot_enabled[12:].any()))
        self.assertAlmostEqual(float(task.offset[-1]), .9)

    def test_directional_losses_cannot_be_hidden_by_aggregate_gains(self):
        initial = self.metrics()
        initial["cases"][25].update(survived=True, recovered=True)  # Backward chest.
        initial = self.metrics(initial["cases"])
        candidate = copy.deepcopy(initial["cases"])
        candidate[25].update(survived=False, recovered=False)
        candidate[1].update(survived=True, recovered=True)
        candidate[2].update(survived=True, recovered=True)
        candidate = self.metrics(candidate)
        self.assertGreater(checkpoint_rank(candidate), checkpoint_rank(initial))
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {})
            best.consider(lambda p: Path(p).write_bytes(b"initial"), initial, iteration=0, samples=0)
            self.assertFalse(best.consider(lambda p: self.fail("Regression saved"), candidate, iteration=1, samples=1))
            self.assertEqual(best.path.read_bytes(), b"initial")
            self.assertEqual(best.decisions[-1]["reason"], "directional_regression")
            # A real gain without any subgroup loss remains eligible.
            candidate["cases"][25].update(survived=True, recovered=True)
            self.assertTrue(best.consider(lambda p: Path(p).write_bytes(b"gain"),
                            self.metrics(candidate["cases"]), iteration=2, samples=2))

    def test_torso_and_settling_are_protected_even_when_direction_survival_is_unchanged(self):
        initial = self.metrics()["cases"]
        initial[25].update(survived=True, recovered=True)
        candidate = copy.deepcopy(initial)
        candidate[25].update(survived=False, recovered=False)
        candidate[28].update(survived=True, recovered=True)  # Same direction, arm.
        losses = ball_regressions(self.metrics(candidate), self.metrics(initial))
        self.assertTrue(any(r["group"] == "direction_2_torso" for r in losses))
        self.assertFalse(any(r["group"] == "direction_2" for r in losses))
        candidate = copy.deepcopy(initial)
        candidate[25]["recovered"] = False
        self.assertTrue(any(r["metric"] == "recovered_hits" for r in
                            ball_regressions(self.metrics(candidate), self.metrics(initial))))

    def test_rejects_missing_inconsistent_or_changed_case_evidence(self):
        initial = self.metrics()
        for modify in (lambda m: m.pop("cases"), lambda m: m.update(confirmed_hits=95),
                       lambda m: m["cases"][0].update(speed=2.),
                       lambda m: m["cases"][0].update(name=m["cases"][1]["name"])):
            candidate = copy.deepcopy(initial)
            modify(candidate)
            with self.assertRaises(ValueError):
                ball_regressions(candidate, initial)
        reordered = {**initial, "cases": list(reversed(initial["cases"]))}
        self.assertEqual(ball_regressions(reordered, initial), [])

    def test_fixed_speed_curriculum_keeps_both_start_phases(self):
        cases = [{**c, "hit": True, "survived": False, "recovered": False}
                 for c in validation_cases(2., 2.)]
        metrics = self.metrics(cases)
        self.assertEqual(ball_regressions(metrics, metrics), [])
        duplicate = copy.deepcopy(cases)
        duplicate[0] = copy.deepcopy(duplicate[1])
        with self.assertRaisesRegex(ValueError, "unique cases"):
            ball_regressions(self.metrics(duplicate), metrics)

    def test_recovery_requires_hit_final_settling_and_survival(self):
        self.activate()
        from envs.base_env import DoneFlags
        for hit, disturbed_at_end, fall, expected in (
                (True, False, False, 1), (False, False, False, 0),
                (True, True, False, 0), (True, False, True, 0)):
            with self.subTest(hit=hit, disturbed=disturbed_at_end, fall=fall):
                q = torch.zeros(1, 46)
                q[0, 2], q[0, 3] = .9, 1.
                task = SimpleNamespace(validation_cases=[validation_cases()[0]],
                    get_num_envs=lambda: 1, device="cpu", dt=.1, episode_seconds=1.,
                    ids=torch.tensor([0]), offset=torch.zeros(1), steps=torch.zeros(1, dtype=torch.long),
                    allowed_ground=torch.arange(4), set_mode=lambda mode: None,
                    observations=lambda: torch.zeros(1, 362))
                task.reference = SimpleNamespace(sample=lambda phase: (q, torch.zeros(1, 45)))
                task.engine = SimpleNamespace(hit=torch.tensor([False]), native_qpos=lambda: q,
                    get_root_pos=lambda _: q[:, :3], get_root_ang_vel=lambda _: torch.zeros(1, 3),
                    get_root_vel=lambda _: torch.ones(1, 3) if disturbed_at_end and task.steps[0] == 10 else torch.zeros(1, 3),
                    get_ground_contact_forces=lambda _: torch.full((1, 4, 3), 20.))
                def reset(ids=None):
                    if ids is None or len(ids):
                        task.steps.zero_()
                        task.engine.hit.zero_()
                def step(action):
                    task.steps += 1
                    task.engine.hit[:] = hit and task.steps[0] >= 2
                    done = DoneFlags.NULL.value
                    if task.steps[0] == 10:
                        done = DoneFlags.FAIL.value if fall else DoneFlags.TIME.value
                    return task.observations(), torch.ones(1), torch.tensor([done]), {}
                task.reset, task.step = reset, step
                metrics, _ = evaluate(task, lambda obs: torch.zeros(1, 30))
                self.assertEqual(metrics["recovered_hits"], expected)

    def test_parity_rejects_changed_actions_missing_frames_and_missed_hits(self):
        dt = .016668
        metrics = {"cases": [dict(name="case", first_hit_step=-1, survived=True, survival_seconds=2 * dt, launch_step=60)]}
        traces = {key: np.zeros((2, 1, width)) for key, width in
                  (("observations", 362), ("actions", 30), ("qpos_after", 46), ("ball_position", 3))}
        actual = [dict(name="case", first_hit_step=-1, survived=True, survival_seconds=2 * dt,
                       trace=[dict(step=i, observation=[0.] * 362, action=[0.] * 30,
                                   qpos_after=[0.] * 46, ball_position=[0.] * 3) for i in range(2)])]
        self.assertTrue(compare(metrics, traces, actual, dt)["passed"])
        changed = copy.deepcopy(actual)
        changed[0]["trace"][0]["action"][0] = .1
        self.assertFalse(compare(metrics, traces, changed, dt)["passed"])
        changed = copy.deepcopy(actual)
        changed[0]["trace"].pop()
        self.assertFalse(compare(metrics, traces, changed, dt)["passed"])
        changed = copy.deepcopy(actual)
        changed[0]["first_hit_step"] = 1
        self.assertFalse(compare(metrics, traces, changed, dt)["passed"])


if __name__ == "__main__":
    unittest.main()

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
from .checkpoints import BestCheckpoint, checkpoint_rank, ball_regressions
from .evaluate_stand import evaluate
from .godot_ball import compare
from .runtime import activate
from .rig import Rig


class BallValidationTests(unittest.TestCase):
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

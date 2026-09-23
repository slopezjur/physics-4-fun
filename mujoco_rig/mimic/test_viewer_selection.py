"""Automatic viewer selection preserves completed exports and supports rollback."""
import json
from pathlib import Path
import tempfile
import unittest

from .baseline import sha256
from .viewer_selection import read_selection, rollback, select_run, selection_path


class ViewerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def make_run(self, name, task="ball"):
        run = self.root / name
        bundle = run / "export"
        bundle.mkdir(parents=True)
        contract = {}
        for file, key in (("stand.onnx", "actor_sha256"), ("stand_reference.npz", "reference_sha256"),
                          ("reference_frames.json", "reference_frames_sha256")):
            (bundle / file).write_bytes(name.encode())
            contract[key] = sha256(bundle / file)
        (bundle / "contract.json").write_text(json.dumps(contract))
        (bundle / "experiment.json").write_text(json.dumps({"task": task}))
        (run / "best.pt").write_bytes(name.encode())
        (run / "report.json").write_text(json.dumps({"experiment": {"task": task},
            "onnx_max_action_error": 1e-7, "ball_gate_passed": False, "export_source": "best.pt",
            "selected_checkpoint": {"checkpoint_sha256": sha256(run / "best.pt")}}))
        return run

    def test_selection_rollback_and_repeat_selection_preserve_previous(self):
        old, new = self.make_run("old"), self.make_run("new")
        select_run(old, self.root)
        select_run(new, self.root)
        selected = select_run(new, self.root)
        self.assertEqual(selected["bundle"], "res://new/export")
        self.assertEqual(selected["previous"], "res://old/export")
        self.assertEqual(rollback("ball", self.root)["bundle"], "res://old/export")
        self.assertEqual(rollback("ball", self.root)["bundle"], "res://new/export")
        self.assertTrue((old / "export/stand.onnx").exists())

    def test_incomplete_or_corrupt_run_never_replaces_current_selection(self):
        select_run(self.make_run("old"), self.root)
        path = selection_path("ball", self.root)
        original = path.read_bytes()
        for name, corrupt_file in (("unfinished", "report.json"), ("bad_actor", "export/stand.onnx"),
                                   ("bad_checkpoint", "best.pt")):
            run = self.make_run(name)
            (run / corrupt_file).unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                select_run(run, self.root)
            self.assertEqual(path.read_bytes(), original)

    def test_tasks_stay_separate_and_missing_previous_is_an_error(self):
        ball = select_run(self.make_run("ball"), self.root)
        select_run(self.make_run("stand", "stand"), self.root)
        self.assertEqual(read_selection("ball", self.root), ball)
        with self.assertRaisesRegex(ValueError, "No previous"):
            rollback("ball", self.root)

    def test_zero_shot_diagnostic_cannot_replace_viewer_selection(self):
        select_run(self.make_run("old"), self.root)
        path = selection_path("ball", self.root)
        original = path.read_bytes()
        run = self.make_run("diagnostic")
        contract_path = run / "export/contract.json"
        contract = json.loads(contract_path.read_text())
        contract["diagnostic_only"] = True
        contract_path.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "Diagnostic"):
            select_run(run, self.root)
        self.assertEqual(original, path.read_bytes())

    def test_failed_export_check_is_rejected_but_behavioral_gate_is_not_required(self):
        run = self.make_run("review")
        # A successfully exported experimental actor can be reviewed before it passes recovery.
        select_run(run, self.root)
        report = json.loads((run / "report.json").read_text())
        report["onnx_max_action_error"] = .1
        (run / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "ONNX"):
            select_run(run, self.root)


if __name__ == "__main__":
    unittest.main()

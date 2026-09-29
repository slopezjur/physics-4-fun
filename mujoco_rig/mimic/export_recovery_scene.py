"""Export a native-gated recovery controller for the local MimicPerturb scene."""
import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .baseline import sha256
from .recovery_step_motion import support_schedule
from .rig import Rig


SCHEMA = "mimic_recovery_scene_bundle_v1"
ROOT = Path(__file__).resolve().parents[2]


def export(run, teacher_run, teacher_search, source_contract, out, scene_ball_mass=.25):
    if out.exists():
        raise FileExistsError(out)
    if not math.isfinite(scene_ball_mass) or not .1 <= scene_ball_mass <= 8.:
        raise ValueError("Scene ball mass must be 0.1–8 kg")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    checkpoint = run / "best.pt"
    source = teacher_run / "report.json"
    search = teacher_search / "report.json"
    motion = teacher_run / "step_reference.npz"
    character = ROOT / "mujoco_rig/dummy.xml"
    ball_world = ROOT / "mujoco_rig/dummy_ball.xml"
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    best = report["best"]
    if (not report["passed"] or not best["actor_changed"]
            or not all(row["passed"] for row in best["ball"])
            or not all(row["passed"] for row in best["disturbance"])
            or report["checkpoint_sha256"] != sha256(checkpoint)
            or protocol["model_sha256"] != sha256(character)
            or protocol["teacher_report_sha256"] != sha256(source)
            or protocol["teacher_search_sha256"] != sha256(search)
            or protocol["reference_sha256"] != sha256(motion)
            or protocol["observation_contract_sha256"] != sha256(source_contract)
            or contract["model_sha256"] != sha256(character)
            or contract["num_obs"] != 362
            or contract["observation_contract"] != "mimic_stand_target_pd_v2"
            or protocol["ball_mass_kg"] != .25):
        raise ValueError("Recovery scene export requires intact native-gated 0.25 kg sources")
    with np.load(motion, allow_pickle=False) as arrays:
        frames = {"dt": float(arrays["dt"]), "reference_sha256": sha256(motion),
                  "qpos": arrays["qpos"].astype(np.float32).tolist(),
                  "qvel": arrays["qvel"].astype(np.float32).tolist()}
        motion_arrays = {key: arrays[key].copy() for key in arrays.files}
    times, torques, centers, _ = support_schedule(Rig.load(character), motion_arrays)
    if (not np.isfinite(times).all() or not np.isfinite(torques).all()
            or not np.isfinite(centers).all()):
        raise ValueError("Recovery support schedule contains nonfinite values")
    out.mkdir(parents=True)
    for origin, name in ((checkpoint, "recovery.pt"), (motion, "motion.npz"),
                         (source, "source.json"), (search, "search.json"),
                         (character, "dummy.xml"), (run / "report.json", "training_report.json")):
        shutil.copyfile(origin, out / name)
    world = ET.parse(ball_world)
    projectile = world.getroot().find("./worldbody/body[@name='ball']/geom[@name='g_ball']")
    if projectile is None or float(projectile.attrib["mass"]) != 8.:
        raise ValueError("Unexpected source projectile")
    projectile.set("mass", format(scene_ball_mass, ".9g"))
    world.write(out / "dummy_ball.xml", encoding="unicode")
    np.savez_compressed(out / "support_schedule.npz", times=times,
                        torques=torques, centers=centers)
    (out / "reference_frames.json").write_text(json.dumps(frames), encoding="utf-8")
    contract.update(reference_sha256=sha256(out / "motion.npz"),
                    reference_frames_sha256=sha256(out / "reference_frames.json"),
                    reference_duration=(len(frames["qpos"]) - 1) * frames["dt"],
                    deployment_status="Experimental native recovery scene adapter")
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    manifest = dict(schema=SCHEMA, training_report_sha256=sha256(out / "training_report.json"),
                    checkpoint_sha256=sha256(out / "recovery.pt"),
                    motion_sha256=sha256(out / "motion.npz"),
                    source_sha256=sha256(out / "source.json"),
                    search_sha256=sha256(out / "search.json"),
                    contract_sha256=sha256(out / "contract.json"),
                    character_sha256=sha256(out / "dummy.xml"),
                    ball_world_sha256=sha256(out / "dummy_ball.xml"),
                    support_schedule_sha256=sha256(out / "support_schedule.npz"),
                    ball_mass_kg=scene_ball_mass,
                    trained_ball_mass_kg=protocol["ball_mass_kg"],
                    ball_speed_m_s=1.,
                    residual_scale=protocol["residual_scale"],
                    controller_settings=protocol["controller_settings"],
                    selected_iteration=best["iteration"])
    (out / "recovery.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "teacher-run", "teacher-search", "source-contract", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene-ball-mass", type=float, default=.25)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.teacher_run, args.teacher_search,
                            args.source_contract, args.out, args.scene_ball_mass)))

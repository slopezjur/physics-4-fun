"""Policy-independent rig checks and reproducible Python/C# sensor fixtures.

python mujoco_rig/rl/foundation_audit.py --json logs/foundation-audit.json \
    --native_fixture_dir logs/foundation-native
Set P4F_FOUNDATION_FIXTURES to that absolute directory before dotnet test.
"""
import argparse
import json
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from export_onnx import build_contract
from env_config import FALL_FRACTION
from observation_contract import FOUNDATION
from perturb_env import PerturbEnv


def audit():
    env = PerturbEnv(num_envs=1, observation_version=FOUNDATION)
    m, d = env.model, env.datas[0]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    env._park_ball(d)
    mujoco.mj_forward(m, d)
    bodies = np.flatnonzero(env.mass > 0)
    assert np.all(m.body_mass[bodies] > 0)
    assert np.all(m.body_inertia[bodies] > 0)
    inertias = np.sort(m.body_inertia[bodies], axis=1)
    assert np.all(inertias[:, 2] <= inertias[:, :2].sum(axis=1) + 1e-9)
    floor = m.geom('floor').id
    self_contacts = [tuple(map(int, c.geom)) for c in d.contact if floor not in c.geom]
    assert not self_contacts, f'Rest pose self-contact: {self_contacts}'
    # Generalized gravity demand is a useful scale check, not a static balance
    # proof: the floating root still needs a feasible distribution of floor forces.
    gravity_fraction = np.abs(d.qfrc_bias[env.vadr]) / (env.force_limit * env.authority)
    native = np.zeros(6)
    d.qvel[:6] = [.3, -.2, .1, 1., -.7, .4]
    mujoco.mj_forward(m, d)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, env.pelvis, native, 0)
    np.testing.assert_allclose(env._foot_velocity(d, env.pelvis), native[3:], atol=1e-12)

    # Contacts disabled solely for this isolated differential actuator probe.
    # Subtract +/- responses to cancel gravity, ligament and other bias forces.
    original_flags = m.opt.disableflags
    responses = []
    try:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        for action, actuator in enumerate(env.act_idx):
            acceleration = []
            for sign in (-1, 1):
                mujoco.mj_resetDataKeyframe(m, d, 0)
                d.ctrl[actuator] = sign * .1 * env.force_limit[action] * env.authority
                mujoco.mj_forward(m, d)
                np.testing.assert_allclose(d.actuator_force[actuator], d.ctrl[actuator], atol=1e-9)
                acceleration.append(d.qacc[env.vadr[action]])
                assert np.isfinite(d.qacc).all()
            response = acceleration[1] - acceleration[0]
            assert response > 0, f'Reversed or ineffective actuator: {env.act_names[action]}'
            responses.append(float(response))
    finally:
        m.opt.disableflags = original_flags

    mujoco.mj_resetDataKeyframe(m, d, 0)
    env._park_ball(d)
    first_fall = None
    for step in range(round(4 / env.dt)):
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        if first_fall is None and d.xpos[env.pelvis, 2] < FALL_FRACTION * env.rest_pelvis_z:
            first_fall = (step + 1) * env.dt
    assert first_fall is not None, 'Zero torque unexpectedly holds the dummy upright'
    return dict(body_mass_kg=float(env.total_mass), actuators_checked=len(responses),
                minimum_signed_acceleration_response=min(responses), rest_self_contacts=len(self_contacts),
                max_joint_gravity_fraction=float(gravity_fraction.max()),
                zero_torque_fall_seconds=first_fall, native_origin_velocity_matches=True)


def native_fixtures(directory):
    """Real model, native solver, full exported contract and several physical states."""
    directory.mkdir(parents=True, exist_ok=True)
    env = PerturbEnv(num_envs=1, observation_version=FOUNDATION)
    contract = build_contract('perturb', 'foundation_fixture',
                              dict(num_obs=140, num_actions=30, observation_version=FOUNDATION), env)
    (directory / 'contract.json').write_text(json.dumps(contract), encoding='utf-8')
    # Full-observation fixtures exercise transforms, floor filtering and both
    # sides of a rotating root. Store XML keyframes so C# uses its normal reset.
    for index, lift in enumerate((0., .01, .4)):
        d = env.datas[0]
        mujoco.mj_resetDataKeyframe(env.model, d, 0)
        env._park_ball(d)
        d.qpos[2] += lift
        d.qvel[:6] = [.3, -.2, .1, 1., -.7, .4]
        env.prev_action[:] = np.linspace(-.2, .3, 30)
        env.action_queue[0][:] = np.linspace(.4, -.1, 30)
        xml = directory / f'body{index}.xml'
        mujoco.mj_saveLastXML(str(xml), env.model)
        tree = ET.parse(xml)
        key = tree.find('./keyframe/key[@name="rest"]')
        key.set('qpos', ' '.join(map(str, d.qpos)))
        key.set('qvel', ' '.join(map(str, d.qvel)))
        tree.write(xml, encoding='utf-8')
        mujoco.mj_forward(env.model, d)
        observations = [env.get_observations()[0].tolist()]
        for _ in range(20):
            mujoco.mj_step(env.model, d)
            observations.append(env.get_observations()[0].tolist())
        payload = dict(xml=xml.name, previous=env.prev_action[0].tolist(),
                       pending=env.action_queue[0][0].tolist(), observations=observations)
        (directory / f'body{index}.json').write_text(json.dumps(payload), encoding='utf-8')
    # Independent heel/toe fixtures catch foot-origin proxy regressions.
    from test_foot_contacts import MODEL, contact_fixture
    from foot_contacts import FootContacts
    for index, angle in enumerate((0., -.55, .55)):
        m, d = contact_fixture(angle, pivot=angle != 0)
        tree = ET.ElementTree(ET.fromstring(MODEL))
        keys = ET.SubElement(tree.getroot(), 'keyframe')
        ET.SubElement(keys, 'key', name='rest', qpos=' '.join(map(str, d.qpos)), qvel=' '.join(map(str, d.qvel)))
        xml = directory / f'foot{index}.xml'
        tree.write(xml, encoding='utf-8')
        payload = dict(xml=xml.name, loads=FootContacts(m).read(d)[0].tolist())
        (directory / f'foot{index}.json').write_text(json.dumps(payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=pathlib.Path, required=True)
    parser.add_argument('--native_fixture_dir', type=pathlib.Path)
    args = parser.parse_args()
    result = audit()
    if args.native_fixture_dir:
        native_fixtures(args.native_fixture_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

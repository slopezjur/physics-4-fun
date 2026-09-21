"""Contact semantics checked against physics, including loaded heel/toe pivots."""
import pathlib
import sys
import unittest

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from foot_contacts import FootContacts, loaded_feet
from perturb_env import PerturbEnv
from recovery_metrics import HeightProxyRecoveryMetrics, RecoveryMetrics
from recovery_reward import contact_history, reward_terms
from test_recovery import fixture


MODEL = '''<mujoco><option timestep="0.004167"/>
<worldbody><geom name="floor" type="plane" size="10 10 .1"/>
<body name="Foot_L" pos="0 0 .04"><freejoint/>
<geom name="sole_L" type="box" size=".1 .05 .04" mass="1"/>
<body name="Toe_L" pos=".14 0 0"><geom name="toe_L" type="box" size=".04 .05 .04" mass=".2"/></body>
</body>
<body name="Foot_R" pos="1 0 1"><freejoint/>
<geom type="box" size=".1 .05 .04" mass="1"/>
<body name="Toe_R" pos=".14 0 0"><geom type="box" size=".04 .05 .04" mass=".2"/></body>
</body>
<body name="ball" pos="3 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>'''


def contact_fixture(angle=0., pivot=False):
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    mujoco.mj_forward(model, data)
    # Translate the lowest sole/toe vertex just through the plane.
    lowest = min(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3,3)[2]) @ model.geom_size[g]
                 for g in (model.geom('sole_L').id, model.geom('toe_L').id))
    data.qpos[2] -= lowest + .001
    mujoco.mj_forward(model, data)
    if pivot:
        point = data.contact[0].pos.copy()
        omega = np.array([0., 2., 0.])
        data.qvel[:3] = -np.cross(omega, point - data.xpos[model.body('Foot_L').id])
        data.qvel[3:6] = omega
        mujoco.mj_forward(model, data)
    return model, data


class ContactTests(unittest.TestCase):
    def test_flat_loaded_foot_and_unloaded_raised_foot(self):
        m,d = contact_fixture()
        load, slip = FootContacts(m).read(d)
        self.assertGreater(load[0], 5)
        self.assertEqual(load[1], 0)
        np.testing.assert_array_equal(slip, 0)

    def test_loaded_heel_and_toe_pivots_do_not_count_as_sliding(self):
        for angle in (-.55,.55):
            with self.subTest(angle=angle):
                m,d = contact_fixture(angle, pivot=True)
                load, slip = FootContacts(m).read(d)
                self.assertGreater(d.xpos[m.body('Foot_L').id,2], .04 + .025)
                self.assertGreater(load[0], 5)
                self.assertGreater(np.linalg.norm(d.qvel[:3]), .05)
                self.assertLess(slip[0], 1e-8)
                f = fixture()
                f['grounded'] = loaded_feet(np, load, np.zeros(2,dtype=bool))
                f['slip_speed_sq'] = slip
                f['foot_velocity'][0] = d.qvel[:3]
                self.assertAlmostEqual(reward_terms(np, **f)['sliding'], 0., places=7)
                if angle > 0:
                    # Only the toe touches: it must belong to the same foot's load.
                    self.assertTrue(all(m.geom_bodyid[c.geom2] == m.body('Toe_L').id
                                        for c in d.contact))

    def test_translation_and_twisting_cannot_cancel_slip(self):
        m,d = contact_fixture()
        d.qvel[0] = .5
        mujoco.mj_forward(m,d)
        _,slip = FootContacts(m).read(d)
        self.assertAlmostEqual(slip[0], .25, places=6)
        d.qvel[:] = 0
        d.qvel[5] = 3
        mujoco.mj_forward(m,d)
        _,slip = FootContacts(m).read(d)
        self.assertGreater(slip[0], .01)

    def test_ball_contact_is_not_floor_support(self):
        m,d = contact_fixture()
        d.qpos[2] = .4
        d.qpos[14:17] = [0,0,.28]
        mujoco.mj_forward(m,d)
        self.assertGreater(d.ncon, 0)
        load,slip = FootContacts(m).read(d)
        np.testing.assert_array_equal(load,0)
        np.testing.assert_array_equal(slip,0)

    def test_load_hysteresis_and_landing_history_match_numpy_torch(self):
        ground = np.zeros((1,2),dtype=bool)
        elapsed = np.full((1,2),.25)
        for force, expected in [(6,True),(3,True),(.5,False),(3,False),(6,True)]:
            load = np.array([[force,0.]])
            new = loaded_feet(np,load,ground)
            torch_new = loaded_feet(torch,torch.tensor(load),torch.tensor(ground))
            np.testing.assert_array_equal(new,torch_new.numpy())
            self.assertEqual(new[0,0],expected)
            ground,elapsed,rapid = contact_history(np,new,ground,elapsed,1/60)
        self.assertGreater(rapid[0],0)

    def test_partial_reset_does_not_retain_old_contact_measurements(self):
        env = PerturbEnv(num_envs=2)
        env.step(torch.zeros((2,env.num_actions)))
        env.datas[0].qpos[2] += 2
        mujoco.mj_forward(env.model,env.datas[0])
        self.assertFalse(env.recovery_features(0)['grounded'].any())
        other = env.foot_contacts.read(env.datas[1])[0].copy()
        env.reset_idx([0])
        np.testing.assert_array_equal(env.foot_contacts.read(env.datas[1])[0],other)
        fresh = env.foot_contacts.read(env.datas[0])[0]
        np.testing.assert_array_equal(env.recovery_features(0)['grounded'],loaded_feet(np,fresh,env.grounded[0]))

    def test_legacy_gate_does_not_consume_new_contact_flags_or_slip(self):
        historical = HeightProxyRecoveryMetrics(1,1/60,.4,[.04,.04])
        measured = RecoveryMetrics(1,1/60,.4)
        f = fixture()
        f['feet'][:,2] = .10
        for _ in range(90):
            for metrics in (historical,measured):
                metrics.record(0,f,np.zeros(3),True,1,np.zeros(30))
        self.assertEqual(historical.result(np.ones(1,bool),True)['recovered'],0)
        self.assertEqual(measured.result(np.ones(1,bool),True)['recovered'],100)


@unittest.skipUnless(torch.cuda.is_available(), 'requires the training GPU')
class ContactGpuTests(unittest.TestCase):
    def test_physical_cpu_gpu_load_slip_and_reward_parity(self):
        import mujoco_warp as mjw
        import warp as wp
        from foot_contacts_warp import FootContactsWarp
        m,d = contact_fixture()
        wm = mjw.put_model(m)
        wd = mjw.put_data(m,d,nworld=4,njmax=128,nconmax=32,naconmax=128)
        sensor = FootContactsWarp(m,wm,wd,4,'cuda')
        qpos,qvel = wp.to_torch(wd.qpos),wp.to_torch(wd.qvel)
        states = []
        for i,angle in enumerate((0.,-.55,.55,0.)):
            _,state = contact_fixture(angle,pivot=angle != 0)
            if i == 0:
                state.qvel[0] = .5
            if i == 3:
                state.qpos[2] += 1
            mujoco.mj_forward(m,state)
            states.append(state)
            qpos[i] = torch.as_tensor(state.qpos,device='cuda',dtype=torch.float32)
            qvel[i] = torch.as_tensor(state.qvel,device='cuda',dtype=torch.float32)
        mjw.forward(wm,wd)
        load,slip = sensor.read()
        expected = [FootContacts(m).read(state) for state in states]
        np.testing.assert_allclose(load.cpu(),np.array([x[0] for x in expected]),rtol=2e-3,atol=2e-3)
        np.testing.assert_allclose(slip.cpu(),np.array([x[1] for x in expected]),rtol=2e-3,atol=2e-6)
        for i, (cpu_load,cpu_slip) in enumerate(expected):
            f = fixture()
            f.update(grounded=loaded_feet(np,cpu_load,np.zeros(2,bool)),slip_speed_sq=cpu_slip)
            cpu_reward = sum(reward_terms(np,**f).values())
            f.update(grounded=loaded_feet(np,load[i].cpu().numpy(),np.zeros(2,bool)),
                     slip_speed_sq=slip[i].cpu().numpy())
            self.assertAlmostEqual(sum(reward_terms(np,**f).values()),cpu_reward,places=5)
        # Empty contact buffer after a loaded one: stale capacity slots must not leak.
        qpos[:,2] += 3
        qpos[:,9] += 3
        mjw.forward(wm,wd)
        load,slip = sensor.read()
        np.testing.assert_array_equal(load.cpu(),0)
        np.testing.assert_array_equal(slip.cpu(),0)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from .audit_motion_authority import envelope_excess
from .collect_recovery_examples import example_checks, articulated_step_feet


class RecoveryExampleTests(unittest.TestCase):
    def quality(self):
        return dict(articulation_first_5s=dict(hip=dict(range_rad=[.3,0.,0.,.3,0.,0.]),
                    knee=dict(range_rad=[.3,.3])),contact_slip_distance_by_foot_m=[.01,.01],
                    rapid_contact_reversals_by_foot=[0,0])

    def test_either_foot_can_supply_a_mechanically_filtered_reaction(self):
        for counts in ([1,0],[0,1]):
            case=dict(recovered=True,recovered_with_step=True,steps_by_foot=counts)
            self.assertTrue(all(example_checks(case,self.quality()).values()))

    def test_articulated_shuffling_and_failed_recovery_are_rejected(self):
        case=dict(recovered=True,recovered_with_step=True,steps_by_foot=[1,0])
        quality=self.quality()
        quality['contact_slip_distance_by_foot_m']=[.3,.2]
        quality['rapid_contact_reversals_by_foot']=[8,4]
        checks=example_checks(case,quality)
        self.assertTrue(checks['leg_excursion'])
        self.assertFalse(checks['bounded_contact_slip'])
        self.assertFalse(checks['bounded_reversals'])
        case['recovered']=False
        self.assertFalse(example_checks(case,self.quality())['recovered'])

    def test_pose_envelope_respects_residual_radius_and_joint_limits(self):
        result=envelope_excess(np.array([[.1],[.3]]),np.array([[.1],[.5]]),
                               np.array([[0.,.4]]),.25)
        np.testing.assert_allclose(result[:,:,0],[[0.,0.],[.15,.1]])

    def test_bilateral_coverage_does_not_count_an_unarticulated_other_foot(self):
        case=dict(recovered=True,recovered_with_step=True,steps_by_foot=[1,1])
        quality=self.quality()
        quality['articulation_first_5s']['knee']['range_rad'][1]=.01
        self.assertEqual(articulated_step_feet(case,quality),[0])


if __name__=='__main__':
    unittest.main()

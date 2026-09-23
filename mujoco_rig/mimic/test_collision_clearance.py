"""Conservative box distances remain stable at finite-difference perturbations."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .collision_clearance import box_separation


class CollisionClearanceTests(unittest.TestCase):
    def test_axis_aligned_separation_and_overlap(self):
        a, size, rot = np.zeros(3), np.full(3, .1), np.eye(3)
        self.assertAlmostEqual(box_separation(a, rot, size, np.array([.206, 0, 0]), rot, size), .006)
        self.assertAlmostEqual(box_separation(a, rot, size, np.array([.195, 0, 0]), rot, size), -.005)

    def test_diagonal_distance_is_a_conservative_lower_bound(self):
        gap = box_separation(np.zeros(3), np.eye(3), np.ones(3), np.array([5., 6., 0.]), np.eye(3), np.ones(3))
        self.assertAlmostEqual(gap, 4.)
        self.assertLessEqual(gap, 5.)  # Actual closest Euclidean distance is sqrt(3^2 + 4^2).

    def test_tiny_rotation_does_not_create_a_zero_distance_discontinuity(self):
        size = np.array([.0318, .0663, .0442])
        b = np.array([2 * size[0] + .00623, 0, 0])
        rotations = Rotation.from_rotvec(np.array([[0, angle, 0] for angle in (0., 1e-8, 1e-7)])).as_matrix()
        gaps = box_separation(np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)), np.tile(size, (3, 1)),
                              np.tile(b, (3, 1)), rotations, np.tile(size, (3, 1)))
        self.assertLess(np.ptp(gaps), 1e-7)
        self.assertTrue((gaps > .006).all())

    def test_rotated_pair_is_symmetric(self):
        a, b = Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix(), Rotation.from_euler("xyz", [.5, .1, -.2]).as_matrix()
        p, q = np.array([.3, .2, -.1]), np.array([.1, .4, .2])
        s, t = np.array([.1, .2, .1]), np.array([.2, .1, .3])
        self.assertAlmostEqual(box_separation(p, a, s, q, b, t), box_separation(q, b, t, p, a, s))


if __name__ == "__main__":
    unittest.main()

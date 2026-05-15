"""
UR3 Gymnasium environment with 6-DOF position and orientation tracking.
Identical to UR3OrientationEnv but uses an exponential reward for orientation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.ur3_orientation_env import UR3OrientationEnv, _get_site_quat, _quat_geodesic_error


class UR3OrientationExpEnv(UR3OrientationEnv):
    """UR3 environment with exponential orientation reward."""

    def _compute_reward(self, ee_pos, target_pos):
        # Get base position reward from UR3TrackingEnv (via super of super)
        # but we can just use the UR3OrientationEnv one and swap the orient term.
        base_reward = super(UR3OrientationEnv, self)._compute_reward(ee_pos, target_pos)

        if self.orient_weight <= 0.0:
            return base_reward

        ee_quat = _get_site_quat(self.data, self.ee_site_id)
        orient_error = _quat_geodesic_error(ee_quat, self._target_quat)

        # Exponential reward for orientation (similar to position)
        # Peak: 10.0, Scale: 0.2 rad (~11.5 deg)
        # Provides much sharper gradient than cos²(error/2)
        r_orient = self.orient_weight * 10.0 * np.exp(-orient_error / 0.2)

        return base_reward + r_orient

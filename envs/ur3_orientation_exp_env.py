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

    def __init__(self, *args, orient_reward_scale=1.0, **kwargs):
        self.orient_reward_scale = orient_reward_scale
        super().__init__(*args, **kwargs)

    def _compute_reward(self, ee_pos, target_pos):
        # Get base position reward from UR3TrackingEnv (via super of super)
        # but we can just use the UR3OrientationEnv one and swap the orient term.
        base_reward = super(UR3OrientationEnv, self)._compute_reward(ee_pos, target_pos)

        if self.orient_weight <= 0.0:
            return base_reward

        ee_quat = _get_site_quat(self.data, self.ee_site_id)
        orient_error = _quat_geodesic_error(ee_quat, self._target_quat)

        # Exponential reward for orientation (similar to position)
        # Peak: 10.0, Scale: self.orient_reward_scale
        # Large scale (e.g. 1.0) = better discovery of the objective.
        # Small scale (e.g. 0.2) = higher precision once close.
        r_orient = self.orient_weight * 10.0 * np.exp(-orient_error / self.orient_reward_scale)

        return base_reward + r_orient

    def set_difficulty(self, orient_reward_scale=None, **kwargs):
        if orient_reward_scale is not None:
            self.orient_reward_scale = float(orient_reward_scale)
        super().set_difficulty(**kwargs)


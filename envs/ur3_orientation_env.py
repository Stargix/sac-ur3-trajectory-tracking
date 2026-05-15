"""
UR3 Gymnasium environment with 6-DOF position and orientation tracking.

Extends UR3TrackingEnv by adding end-effector orientation to the observation
space and reward function. The target orientation is the home end-effector
pose (computed via forward kinematics at initialisation), representing a
"tool-down" / natural resting orientation that the arm should maintain
throughout the trajectory.

Observation space (35-dimensional):
    [0:27]   Identical to UR3TrackingEnv (position, velocity, target, error)
    [27:31]  Current EE orientation quaternion [w, x, y, z]
    [31:35]  Target orientation quaternion     [w, x, y, z]

All quaternion components are unit-normalised and lie in [-1, 1]; no
additional scaling is applied.

Reward:
    Identical to the base environment, plus an orientation term:
        r_orient = orient_weight * 5.0 * cos²(geodesic_error / 2)
    This term equals 5 * orient_weight when perfectly aligned and decreases
    smoothly to 0 as the angular error approaches 180°.

    orient_weight is set to 0.0 initially (curriculum phase 1 ignores
    orientation) and is increased by PhasedCurriculumCallback as training
    progresses.

Info dict additions (appended to base info):
    orient_error_rad : float   Geodesic angular error in radians
    orient_error_deg : float   Same in degrees
"""

import numpy as np
import mujoco
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.ur3_tracking_env import UR3TrackingEnv


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    mat : np.ndarray, shape (3, 3) or (9,)
        Rotation matrix (row-major, as stored by MuJoCo in site_xmat).

    Returns
    -------
    np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    """
    if mat.shape == (9,):
        mat = mat.reshape(3, 3)
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _get_site_quat(data, site_id: int) -> np.ndarray:
    """Return the orientation of a MuJoCo site as a quaternion [w,x,y,z]."""
    return _mat_to_quat(data.site_xmat[site_id].reshape(3, 3))


def _quat_geodesic_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angular distance between two unit quaternions (radians).

    Handles the sign ambiguity of quaternion representation (q and -q
    describe the same rotation) by using the absolute dot product.

    Parameters
    ----------
    q1, q2 : array_like, shape (4,)
        Unit quaternions in [w, x, y, z] (MuJoCo convention).

    Returns
    -------
    float
        Angle in [0, pi] radians.
    """
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


class UR3OrientationEnv(UR3TrackingEnv):
    """UR3 environment with simultaneous position and orientation tracking."""

    def __init__(
        self,
        xml_path="models/ur3.xml",
        render_mode=None,
        obs_noise_std=0.0,
        action_delay=0,
        traj_radius=0.08,
        traj_speed=0.3,
        episode_steps=1000,
        orient_weight=0.0,
        orient_reward_scale=0.2,
    ):
        """
        Parameters
        ----------
        orient_weight : float
            Weight of the orientation reward term in [0, 1].
        orient_reward_scale : float
            Scale (sharpness) of the exponential orientation reward.
            Large values (1.0) for discovery, small values (0.2) for precision.
        """
        self.orient_weight = orient_weight
        self.orient_reward_scale = orient_reward_scale
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # placeholder

        super().__init__(
            xml_path=xml_path,
            render_mode=render_mode,
            obs_noise_std=obs_noise_std,
            action_delay=action_delay,
            traj_radius=traj_radius,
            traj_speed=traj_speed,
            episode_steps=episode_steps,
        )

        # Record the home EE orientation as the target (already computed
        # by parent __init__ via FK at HOME_Q).
        self._target_quat = _get_site_quat(self.data, self.ee_site_id)

        # Extend observation space from 27 to 35 dimensions.
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(35,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Observation — append orientation to base obs
    # ------------------------------------------------------------------

    def _get_obs(self):
        base_obs = super()._get_obs()          # (27,)

        # Current EE quaternion [w, x, y, z], already unit-normalised.
        ee_quat = _get_site_quat(self.data, self.ee_site_id)

        obs = np.concatenate([base_obs, ee_quat, self._target_quat])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Reward — add orientation term on top of base reward
    # ------------------------------------------------------------------

    def _compute_reward(self, ee_pos, target_pos):
        base_reward = super()._compute_reward(ee_pos, target_pos)

        if self.orient_weight <= 0.0:
            return base_reward

        ee_quat = _get_site_quat(self.data, self.ee_site_id)
        orient_error = _quat_geodesic_error(ee_quat, self._target_quat)

        # Exponential reward for orientation (similar to position)
        # Peak: 10.0, Scale: self.orient_reward_scale
        # Large scale (1.0) = better discovery; small scale (0.2) = precision.
        r_orient = self.orient_weight * 10.0 * np.exp(-orient_error / self.orient_reward_scale)

        return base_reward + r_orient

    # ------------------------------------------------------------------
    # Step — augment info dict with orientation metrics
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ee_quat = _get_site_quat(self.data, self.ee_site_id)
        orient_error_rad = _quat_geodesic_error(ee_quat, self._target_quat)

        info["orient_error_rad"] = float(orient_error_rad)
        info["orient_error_deg"] = float(np.degrees(orient_error_rad))

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Curriculum support — extend base set_difficulty
    # ------------------------------------------------------------------

    def set_difficulty(self, orient_weight=None, orient_reward_scale=None, **kwargs):
        """Set orientation parameters in addition to base parameters."""
        if orient_weight is not None:
            self.orient_weight = float(orient_weight)
        if orient_reward_scale is not None:
            self.orient_reward_scale = float(orient_reward_scale)
        super().set_difficulty(**kwargs)

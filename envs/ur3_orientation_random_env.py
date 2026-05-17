"""
UR3 Gymnasium environment with per-episode randomised spline trajectories
and 6-DOF orientation tracking using exponential reward.

Inherits from UR3OrientationExpEnv and replaces the fixed lemniscate with a
smooth random path generated at every episode reset.
"""

import numpy as np
from scipy.interpolate import splprep, splev
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.ur3_orientation_exp_env import UR3OrientationExpEnv


class UR3OrientationRandomEnv(UR3OrientationExpEnv):
    """UR3 environment with random spline trajectories and orientation tracking."""

    def __init__(
        self,
        xml_path="models/ur3.xml",
        render_mode=None,
        obs_noise_std=0.0,
        action_delay=0,
        episode_steps=1000,
        workspace_radius=0.10,
        min_speed=0.2,
        max_speed=0.6,
        n_control_points_range=(4, 8),
        orient_reward_scale=1.0,
        **kwargs
    ):
        self.workspace_radius = workspace_radius
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.n_cp_min, self.n_cp_max = n_control_points_range

        self._spline_tck = None
        self._spline_length = 1.0
        self._path_speed = 0.3
        self._path_u = 0.0
        self._path_du = 0.0
        self._spline_offset = np.zeros(3)

        super().__init__(
            xml_path=xml_path,
            render_mode=render_mode,
            obs_noise_std=obs_noise_std,
            action_delay=action_delay,
            traj_radius=workspace_radius,
            traj_speed=0.3,
            episode_steps=episode_steps,
            orient_reward_scale=orient_reward_scale,
            **kwargs
        )

    def _generate_spline(self):
        n_cp = self.np_random.integers(self.n_cp_min, self.n_cp_max + 1)
        angles = np.sort(self.np_random.uniform(0, 2 * np.pi, n_cp))
        radii = self.np_random.uniform(0.03, self.workspace_radius, n_cp)

        cx = self.traj_center[0] + radii * np.cos(angles)
        cy = self.traj_center[1] + radii * np.sin(angles)
        cz = np.full(n_cp, self.traj_center[2])

        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])
        cz = np.append(cz, cz[0])

        tck, _ = splprep([cx, cy, cz], s=0, k=3, per=True)
        self._spline_tck = tck

        # Calculate offset to make the spline pass through traj_center at u=0
        p0_x, p0_y, p0_z = splev(0.0, tck)
        self._spline_offset = np.array([
            self.traj_center[0] - float(p0_x),
            self.traj_center[1] - float(p0_y),
            self.traj_center[2] - float(p0_z)
        ])

        u_dense = np.linspace(0, 1, 500)
        pts = np.array(splev(u_dense, tck)).T
        diffs = np.diff(pts, axis=0)
        self._spline_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        self._spline_length = max(self._spline_length, 1e-3)

        self._path_speed = float(
            self.np_random.uniform(self.min_speed, self.max_speed)
        )
        self._path_du = self._path_speed * self.dt / self._spline_length
        self._path_u = 0.0  # Start at u=0 (which is now at traj_center!)

    def _random_path(self, u):
        x, y, z = splev(u % 1.0, self._spline_tck)
        return np.array([float(x) + self._spline_offset[0], 
                         float(y) + self._spline_offset[1], 
                         float(z) + self._spline_offset[2]])

    def _lemniscate(self, t):
        if self._spline_tck is None:
            return self.traj_center.copy()
        # Map time t to spline parameter u
        u = self._initial_u + t * self._path_speed / self._spline_length
        return self._random_path(u)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._generate_spline()
        self._initial_u = self._path_u  # Save initial u for time mapping
        return obs, info

    def step(self, action):
        # We don't need to advance _path_u manually because _lemniscate uses t!
        # And t is advanced in super().step()!
        return super().step(action)

    def set_difficulty(self, workspace_radius=None, min_speed=None,
                       max_speed=None, n_control_points_range=None, **kwargs):
        if workspace_radius is not None:
            self.workspace_radius = workspace_radius
            self.traj_radius = workspace_radius
        if min_speed is not None:
            self.min_speed = min_speed
        if max_speed is not None:
            self.max_speed = max_speed
        if n_control_points_range is not None:
            self.n_cp_min, self.n_cp_max = n_control_points_range
        super().set_difficulty(**kwargs)

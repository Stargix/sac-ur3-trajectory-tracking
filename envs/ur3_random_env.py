"""
UR3 Gymnasium environment with randomised spline trajectories.

Inherits from UR3TrackingEnv and replaces the fixed lemniscate with a
smooth random path generated at every episode reset. The path is a
closed cubic B-spline through randomly sampled control points within
the arm's reachable workspace.

The observation space, reward function, action space, and all other
mechanics are identical to the base environment. Only the target
trajectory changes.

This environment is intended for fine-tuning a pre-trained lemniscate
policy towards generalisation over arbitrary moving targets.

Trajectory properties
---------------------
- Smooth C2-continuous cubic spline (no abrupt direction changes)
- Parameterised by arc length so the target moves at constant speed
- Random number of control points (4-8) at each episode reset
- Points sampled within a configurable radius around the EE home position
- Speed drawn uniformly from [min_speed, max_speed] each episode
"""

import numpy as np
from scipy.interpolate import splprep, splev
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.ur3_tracking_env import UR3TrackingEnv


class UR3RandomEnv(UR3TrackingEnv):
    """UR3 environment with per-episode randomised spline trajectories."""

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
    ):
        """
        Parameters
        ----------
        workspace_radius : float
            Maximum XY distance from the EE home position that control
            points can be placed. Keeps the trajectory reachable.
        min_speed : float
            Minimum trajectory speed (m/s along the path).
        max_speed : float
            Maximum trajectory speed (m/s along the path).
        n_control_points_range : tuple[int, int]
            Range for the number of spline control points per episode.
        """
        self.workspace_radius = workspace_radius
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.n_cp_min, self.n_cp_max = n_control_points_range

        # Spline state — must be initialised before super().__init__ because
        # the parent constructor calls reset(), which calls _lemniscate().
        self._spline_tck = None        # scipy spline representation
        self._spline_length = 1.0      # arc length of the spline (m)
        self._path_speed = 0.3         # current episode speed (m/s)
        self._path_u = 0.0             # arc-length parameter in [0, 1]
        self._path_du = 0.0            # increment per control step

        # Pass dummy traj params — they are overridden per episode
        super().__init__(
            xml_path=xml_path,
            render_mode=render_mode,
            obs_noise_std=obs_noise_std,
            action_delay=action_delay,
            traj_radius=workspace_radius,  # reused as workspace radius
            traj_speed=0.3,
            episode_steps=episode_steps,
        )

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_spline(self):
        """Sample random control points and fit a closed cubic spline."""
        n_cp = self.np_random.integers(self.n_cp_min, self.n_cp_max + 1)

        # Sample control points in XY, keeping Z fixed at home height
        angles = np.sort(self.np_random.uniform(0, 2 * np.pi, n_cp))
        radii = self.np_random.uniform(0.03, self.workspace_radius, n_cp)

        cx = self.traj_center[0] + radii * np.cos(angles)
        cy = self.traj_center[1] + radii * np.sin(angles)
        cz = np.full(n_cp, self.traj_center[2])

        # Close the loop by repeating the first point
        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])
        cz = np.append(cz, cz[0])

        # Fit closed periodic cubic spline
        tck, _ = splprep([cx, cy, cz], s=0, k=3, per=True)
        self._spline_tck = tck

        # Estimate arc length via dense sampling
        u_dense = np.linspace(0, 1, 500)
        pts = np.array(splev(u_dense, tck)).T
        diffs = np.diff(pts, axis=0)
        self._spline_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        self._spline_length = max(self._spline_length, 1e-3)  # safety

        # Draw a random speed for this episode
        self._path_speed = float(
            self.np_random.uniform(self.min_speed, self.max_speed)
        )

        # Arc-length increment per control step: speed * dt / total_length
        self._path_du = self._path_speed * self.dt / self._spline_length

        # Random starting phase
        self._path_u = float(self.np_random.uniform(0, 1))

    def _random_path(self, u):
        """Return the 3-D target position for spline parameter u in [0, 1]."""
        x, y, z = splev(u % 1.0, self._spline_tck)
        return np.array([float(x), float(y), float(z)])

    # ------------------------------------------------------------------
    # Override trajectory interface used by base class
    # ------------------------------------------------------------------

    def _lemniscate(self, t):
        """Replaced by the spline path; t is ignored (we use _path_u).

        Returns traj_center as a safe fallback when the spline has not yet
        been generated (i.e., during the initial reset call from the parent
        constructor, which happens before our own reset() can run).
        """
        if self._spline_tck is None:
            return self.traj_center.copy()
        return self._random_path(self._path_u)

    # ------------------------------------------------------------------
    # Reset — generate a new random trajectory each episode
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Generate spline AFTER super().reset() so np_random is seeded
        self._generate_spline()
        return obs, info

    # ------------------------------------------------------------------
    # Step — advance arc-length parameter instead of traj_t
    # ------------------------------------------------------------------

    def step(self, action):
        # Temporarily monkey-patch traj_t so the base class advance is
        # a no-op for our purposes; we manage _path_u directly.
        obs, reward, terminated, truncated, info = super().step(action)

        # Advance spline parameter by one step
        self._path_u = (self._path_u + self._path_du) % 1.0

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Curriculum support (inherits set_difficulty)
    # ------------------------------------------------------------------

    def set_difficulty(self, workspace_radius=None, min_speed=None,
                       max_speed=None, **kwargs):
        """Extended difficulty control for randomised trajectories."""
        if workspace_radius is not None:
            self.workspace_radius = workspace_radius
            self.traj_radius = workspace_radius
        if min_speed is not None:
            self.min_speed = min_speed
        if max_speed is not None:
            self.max_speed = max_speed
        # Forward remaining kwargs (noise, delay) to base class
        super().set_difficulty(**kwargs)

"""
UR3 Gymnasium environment for lemniscate trajectory tracking.

The environment implements a 6-DOF position-controlled UR3 arm tracking a
Bernoulli lemniscate (figure-eight) in the horizontal plane. The control
interface is a normalized delta-position action space; the underlying
MuJoCo actuators are PD servos (gaintype=fixed, biastype=affine).

Observation space (27-dimensional, normalized):
    [0:6]   Joint positions  / pi
    [6:12]  Joint velocities / 3.0
    [12:15] End-effector position (centered, scaled by 0.15)
    [15:18] End-effector Cartesian velocity / 0.5
    [18:21] Current target position (centered, scaled)
    [21:24] Lookahead target position (10 control steps ahead)
    [24:27] Position error vector / 0.10

Action space (6-dimensional, [-1, 1]):
    Normalized delta applied to each joint reference position.
    MAX_DELTA = 0.08 rad/step  =>  100 Hz control rate.

Reward (exponential formulation):
    r = 10 * exp(-dist / 0.01)   tracking component
      + 0.5                       alive bonus
      - 1.0 * delta_a^2           smoothness penalty
      - 0.1 * jerk                second-order smoothness
      + {0, 0.5, 2.0}             proximity bonus tiers
      - 20.0 (if OOB)             out-of-bounds penalty
"""

import gymnasium as gym
import numpy as np
import mujoco


class UR3TrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    HOME_Q = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    MAX_DELTA = 0.08  # maximum joint delta per control step (rad)

    def __init__(
        self,
        xml_path="models/ur3.xml",
        render_mode=None,
        obs_noise_std=0.0,
        action_delay=0,
        traj_radius=0.08,
        traj_speed=0.3,
        episode_steps=1000,
    ):
        """
        Parameters
        ----------
        xml_path : str
            Path to the MuJoCo XML scene file.
        render_mode : str or None
            Set to "rgb_array" to enable frame rendering.
        obs_noise_std : float
            Standard deviation of Gaussian noise added to joint observations.
            Mirrors sensor noise for sim-to-real transfer.
        action_delay : int
            Number of control steps to delay the applied action.
        traj_radius : float
            Semi-axis of the lemniscate trajectory (metres).
        traj_speed : float
            Angular rate of the trajectory parameter (rad / control-step-second).
        episode_steps : int
            Maximum number of control steps per episode.
        """
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.obs_noise_std = obs_noise_std
        self.action_delay = action_delay
        self.traj_radius = traj_radius
        self.traj_speed = traj_speed
        self.episode_steps = episode_steps
        self.dt = self.model.opt.timestep * 5  # 5 physics substeps -> 100 Hz control

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(27,), dtype=np.float32
        )

        # Resolve model element IDs
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_visual"
        )

        # Compute the home end-effector position via forward kinematics and
        # centre the trajectory on it so the arm starts on the path.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.HOME_Q.copy()
        self.data.ctrl[:6] = self.HOME_Q.copy()
        mujoco.mj_forward(self.model, self.data)
        self.traj_center = self.data.site_xpos[self.ee_site_id].copy()

        # Observation normalisation constants
        self._scale_pos = 0.15   # metres — half-range around trajectory centre
        self._scale_vel = 0.5    # m/s    — typical maximum EE speed
        self._scale_err = 0.10   # metres — typical maximum tracking error

        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.reset()

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def _lemniscate(self, t):
        """Return the 3-D target position for trajectory parameter t."""
        a = self.traj_radius
        d = 1.0 + np.sin(t) ** 2
        return np.array([
            self.traj_center[0] + a * np.cos(t) / d,
            self.traj_center[1] + a * np.sin(t) * np.cos(t) / d,
            self.traj_center[2],
        ])

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_ee_vel(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
        return jacp @ self.data.qvel

    def _is_out_of_bounds(self, pos):
        return (
            np.linalg.norm(pos[:2]) > 0.48
            or pos[2] < 0.05
            or pos[2] > 0.65
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        q = self.data.qpos[:6].copy()
        dq = self.data.qvel[:6].copy()

        if self.obs_noise_std > 0:
            q += np.random.normal(0, self.obs_noise_std, 6)
            dq += np.random.normal(0, self.obs_noise_std * 5, 6)

        ee_pos = self._get_ee_pos()
        if self.obs_noise_std > 0:
            ee_pos += np.random.normal(0, self.obs_noise_std, 3)
        ee_vel = self._get_ee_vel()

        target_now = self._lemniscate(self.traj_t)
        target_next = self._lemniscate(self.traj_t + self.traj_speed * self.dt * 10)
        error = ee_pos - target_now

        obs = np.concatenate([
            q / np.pi,
            dq / 3.0,
            (ee_pos - self.traj_center) / self._scale_pos,
            ee_vel / self._scale_vel,
            (target_now - self.traj_center) / self._scale_pos,
            (target_next - self.traj_center) / self._scale_pos,
            error / self._scale_err,
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, ee_pos, target_pos):
        dist = np.linalg.norm(ee_pos - target_pos)

        r_track = 10.0 * np.exp(-dist / 0.01)
        r_alive = 0.5
        r_smooth = -1.0 * np.linalg.norm(self._last_action - self._prev_action) ** 2
        r_jerk = -0.1 * np.linalg.norm(
            self._last_action - 2 * self._prev_action + self._prev_prev_action
        )
        r_bonus = 2.0 if dist < 0.005 else (0.5 if dist < 0.02 else 0.0)
        r_oob = -20.0 if self._is_out_of_bounds(ee_pos) else 0.0

        return r_track + r_alive + r_smooth + r_jerk + r_bonus + r_oob

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        q_init = self.HOME_Q.copy()
        if self.obs_noise_std > 0:
            q_init += np.random.normal(0, 0.02, 6)

        self.data.qpos[:6] = q_init
        self.data.ctrl[:6] = q_init
        mujoco.mj_forward(self.model, self.data)

        # Randomise the trajectory phase so the agent cannot memorise a single
        # fixed sequence of motor commands.
        self.traj_t = np.random.uniform(0, 2 * np.pi)
        self.step_count = 0

        self.action_buffer = np.zeros((max(self.action_delay, 1), 6))
        self._last_action = np.zeros(6)
        self._prev_action = np.zeros(6)
        self._prev_prev_action = np.zeros(6)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        if self.action_delay > 0:
            delayed = self.action_buffer[0].copy()
            if self.action_delay > 1:
                self.action_buffer[:-1] = self.action_buffer[1:]
            self.action_buffer[-1] = action
        else:
            delayed = action

        q_ref = self.data.ctrl[:6].copy() + delayed * self.MAX_DELTA
        q_ref = np.clip(q_ref, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
        self.data.ctrl[:6] = q_ref

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.traj_t = (self.traj_t + self.traj_speed * self.dt) % (2 * np.pi)
        target_pos = self._lemniscate(self.traj_t)
        self.model.body_pos[self.target_body_id] = target_pos

        ee_pos = self._get_ee_pos()
        self._prev_prev_action = self._prev_action.copy()
        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()

        reward = self._compute_reward(ee_pos, target_pos)
        dist = np.linalg.norm(ee_pos - target_pos)

        self.step_count += 1
        truncated = self.step_count >= self.episode_steps

        info = {
            "dist": dist,
            "dist_mm": dist * 1000,
            "ee_pos": ee_pos,
            "target_pos": target_pos,
        }

        return self._get_obs(), reward, False, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="fixed")
            return self.renderer.render()

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

    # ------------------------------------------------------------------
    # Curriculum support
    # ------------------------------------------------------------------

    def set_difficulty(
        self,
        traj_radius=None,
        traj_speed=None,
        action_delay=None,
        obs_noise_std=None,
    ):
        """Adjust difficulty parameters without recreating the environment.

        Intended to be called by CurriculumCallback during training.
        """
        if traj_radius is not None:
            self.traj_radius = traj_radius
        if traj_speed is not None:
            self.traj_speed = traj_speed
        if obs_noise_std is not None:
            self.obs_noise_std = obs_noise_std
        if action_delay is not None:
            self.action_delay = action_delay
            self.action_buffer = np.zeros((max(self.action_delay, 1), 6))

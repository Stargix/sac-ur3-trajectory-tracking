"""
Training callbacks for UR3 SAC trajectory tracking.

DebugCallback
    Periodically runs a deterministic evaluation episode and writes a
    structured snapshot to disk:
        debugs/stepXXXXXXX/
            evaluation_details.png   — evaluation plot (adds orientation panel
                                       automatically when the environment
                                       exposes orient_error_deg in its info)
            training_progress.png    — reward and error curves over training
            observation_analysis.png — observation distribution histograms
            evaluation.mp4           — rendered episode video
            metrics.json             — tabulated numerical metrics
                                       (includes orient_error_deg fields when
                                       orientation data is available)

Curriculum management is handled by PhasedCurriculumCallback
(training/phased_curriculum.py), which replaces the old CurriculumCallback.
"""

import json
import os
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DebugCallback(BaseCallback):
    """Periodic evaluation with plot and metric generation."""

    def __init__(self, eval_env, debug_dir, eval_freq=25_000,
                 n_eval_steps=1000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.debug_dir = debug_dir
        self.eval_freq = eval_freq
        self.n_eval_steps = n_eval_steps

        self._ep_rewards = []
        self._ep_mean_dists = []
        self._cur_ep_reward = 0.0
        self._cur_ep_dists = []

        os.makedirs(debug_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0.0])[0]
        info = self.locals.get("infos", [{}])[0]
        done = self.locals.get("dones", [False])[0]

        self._cur_ep_reward += reward
        if "dist" in info:
            self._cur_ep_dists.append(info["dist"])

        if done:
            self._ep_rewards.append(self._cur_ep_reward)
            if self._cur_ep_dists:
                self._ep_mean_dists.append(np.mean(self._cur_ep_dists) * 1000)
            self._cur_ep_reward = 0.0
            self._cur_ep_dists = []

        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            self._write_snapshot()

        return True

    # ------------------------------------------------------------------

    def _write_snapshot(self):
        step_dir = os.path.join(self.debug_dir, f"step{self.num_timesteps:07d}")
        os.makedirs(step_dir, exist_ok=True)

        if self.verbose:
            print(f"\n[debug] step {self.num_timesteps:,} — writing snapshot to {step_dir}")

        # ---- run evaluation episode ----
        obs, _ = self.eval_env.reset()
        done = False
        ee_list, tgt_list, rew_list, act_list, obs_list = [], [], [], [], []
        orient_errors_deg = []   # populated only when env exposes orient_error_deg
        frames = []

        for _ in range(self.n_eval_steps):
            if done:
                break
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated

            ee_list.append(info["ee_pos"])
            tgt_list.append(info["target_pos"])
            rew_list.append(reward)
            act_list.append(action)
            obs_list.append(obs.copy())
            if "orient_error_deg" in info:
                orient_errors_deg.append(info["orient_error_deg"])

            if self.eval_env.render_mode == "rgb_array":
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)

        ee_pos = np.array(ee_list)
        tgt_pos = np.array(tgt_list)
        acts = np.array(act_list)
        obs_arr = np.array(obs_list)
        orient_arr = np.array(orient_errors_deg) if orient_errors_deg else None
        dists_mm = np.linalg.norm(ee_pos - tgt_pos, axis=1) * 1000
        jerk = np.linalg.norm(np.diff(acts, n=2, axis=0), axis=1) if len(acts) > 2 else np.zeros(1)
        t = np.arange(len(dists_mm)) * 0.01

        # ---- training progress plot ----
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._ep_rewards:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle(f"Training — step {self.num_timesteps:,}", fontsize=12)

            def rolling(arr, w):
                return np.convolve(arr, np.ones(w) / w, mode="valid")

            ax = axes[0]
            ax.plot(self._ep_rewards, alpha=0.25, color="#534AB7", linewidth=0.6)
            if len(self._ep_rewards) > 20:
                w = min(50, len(self._ep_rewards) // 4)
                ax.plot(range(w - 1, len(self._ep_rewards)),
                        rolling(self._ep_rewards, w),
                        color="#ff4444", linewidth=1.8, label=f"rolling mean ({w})")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            ax.set_title("Reward per episode")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[1]
            if self._ep_mean_dists:
                ax.plot(self._ep_mean_dists, alpha=0.25, color="#0f6e56", linewidth=0.6)
                if len(self._ep_mean_dists) > 20:
                    w = min(50, len(self._ep_mean_dists) // 4)
                    ax.plot(range(w - 1, len(self._ep_mean_dists)),
                            rolling(self._ep_mean_dists, w),
                            color="#ff4444", linewidth=1.8)
                ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10 mm target")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean error (mm)")
            ax.set_title("Tracking error per episode")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, "training_progress.png"), dpi=150)
            plt.close()

        # ---- evaluation detail plot ----
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle(f"Evaluation — step {self.num_timesteps:,}", fontsize=12)

        axes[0, 0].plot(t, dists_mm, color="#534AB7", linewidth=0.8)
        axes[0, 0].axhline(10, color="red", linestyle="--", alpha=0.5, label="10 mm")
        axes[0, 0].axhline(np.mean(dists_mm), color="orange", linestyle="--",
                           label=f"mean {np.mean(dists_mm):.1f} mm")
        axes[0, 0].set(xlabel="Time (s)", ylabel="Error (mm)", title="Position error")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(tgt_pos[:, 0], tgt_pos[:, 1], "--", color="#534AB7",
                        alpha=0.5, label="Target")
        axes[0, 1].plot(ee_pos[:, 0], ee_pos[:, 1], color="#ba7517",
                        linewidth=1.2, label="End-effector")
        axes[0, 1].set(xlabel="X (m)", ylabel="Y (m)", title="XY trajectory")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].set_aspect("equal")
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].plot(t, rew_list, color="#0f6e56", linewidth=0.8)
        axes[0, 2].axhline(np.mean(rew_list), color="orange", linestyle="--",
                           label=f"mean {np.mean(rew_list):.2f}")
        axes[0, 2].set(xlabel="Time (s)", ylabel="Reward", title="Instantaneous reward")
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].grid(alpha=0.3)

        for j in range(6):
            axes[1, 0].hist(acts[:, j], bins=30, alpha=0.45, label=f"J{j}")
        axes[1, 0].set(xlabel="Action (normalised)", ylabel="Count",
                       title="Action distribution")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].hist(dists_mm, bins=50, color="#534AB7", alpha=0.7, edgecolor="white")
        axes[1, 1].axvline(np.mean(dists_mm), color="red", linestyle="--",
                           label=f"mean {np.mean(dists_mm):.1f} mm")
        axes[1, 1].set(xlabel="Error (mm)", ylabel="Count", title="Error distribution")
        axes[1, 1].legend(fontsize=8)

        axes[1, 2].plot(t[2:], jerk, color="#0f6e56", linewidth=0.8)
        axes[1, 2].set(xlabel="Time (s)", ylabel="Jerk",
                       title=f"Action jerk (mean {np.mean(jerk):.4f})")
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(step_dir, "evaluation_details.png"), dpi=150)
        plt.close()

        # ---- orientation plot (only when env exposes orientation data) ----
        if orient_arr is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Orientation tracking — step {self.num_timesteps:,}", fontsize=12)

            axes[0].plot(t[:len(orient_arr)], orient_arr, color="#7289da", linewidth=0.8)
            axes[0].axhline(5, color="red", linestyle="--", alpha=0.5, label="5° target")
            axes[0].axhline(np.mean(orient_arr), color="orange", linestyle="--",
                           label=f"mean {np.mean(orient_arr):.1f}°")
            axes[0].set(xlabel="Time (s)", ylabel="Error (°)", title="Orientation error")
            axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

            axes[1].hist(orient_arr, bins=40, color="#7289da", alpha=0.75, edgecolor="white")
            axes[1].axvline(np.mean(orient_arr), color="red", linestyle="--",
                           label=f"mean {np.mean(orient_arr):.1f}°")
            axes[1].set(xlabel="Error (°)", ylabel="Count", title="Orientation error distribution")
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, "orientation_details.png"), dpi=150)
            plt.close()

        # ---- observation distribution ----
        if obs_arr.shape[0] > 0:
            fig, axes = plt.subplots(3, 3, figsize=(14, 11))
            fig.suptitle(f"Observation distributions — step {self.num_timesteps:,}", fontsize=12)
            labels = ["q0/π", "q1/π", "q2/π", "dq0/3", "dq1/3", "dq2/3",
                      "ee_x", "ee_y", "ee_z"]
            for i, ax in enumerate(axes.flat):
                ax.hist(obs_arr[:, i], bins=30, color="#7289da", alpha=0.75)
                ax.set_title(labels[i], fontsize=9)
                ax.axvline(0, color="red", linestyle="--", alpha=0.4)
                ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(step_dir, "observation_analysis.png"), dpi=150)
            plt.close()

        # ---- video ----
        if frames:
            try:
                import imageio
                imageio.mimsave(os.path.join(step_dir, "evaluation.mp4"), frames, fps=50)
            except Exception as exc:
                if self.verbose:
                    print(f"  [warn] could not save video: {exc}")

        # ---- metrics JSON ----
        eval_metrics = {
            "mean_dist_mm": float(np.mean(dists_mm)),
            "median_dist_mm": float(np.median(dists_mm)),
            "rmse_mm": float(np.sqrt(np.mean(dists_mm ** 2))),
            "max_dist_mm": float(np.max(dists_mm)),
            "pct_under_5mm": float(np.mean(dists_mm < 5) * 100),
            "pct_under_10mm": float(np.mean(dists_mm < 10) * 100),
            "pct_under_20mm": float(np.mean(dists_mm < 20) * 100),
            "mean_reward": float(np.mean(rew_list)),
            "mean_jerk": float(np.mean(jerk)),
        }
        if orient_arr is not None:
            eval_metrics["mean_orient_error_deg"] = float(np.mean(orient_arr))
            eval_metrics["pct_under_5deg"] = float(np.mean(orient_arr < 5) * 100)
            eval_metrics["pct_under_10deg"] = float(np.mean(orient_arr < 10) * 100)
            eval_metrics["pct_under_15deg"] = float(np.mean(orient_arr < 15) * 100)

        metrics = {
            "step": int(self.num_timesteps),
            "timestamp": datetime.now().isoformat(),
            "eval": eval_metrics,
            "training": {
                "total_episodes": len(self._ep_rewards),
                "last_20_mean_reward": float(np.mean(self._ep_rewards[-20:]))
                if len(self._ep_rewards) >= 20 else None,
                "last_20_mean_dist_mm": float(np.mean(self._ep_mean_dists[-20:]))
                if len(self._ep_mean_dists) >= 20 else None,
            },
        }

        with open(os.path.join(step_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        if self.verbose:
            orient_str = (
                f"  |  orient: {eval_metrics['mean_orient_error_deg']:.1f}°"
                if orient_arr is not None else ""
            )
            print(
                f"  RMSE {eval_metrics['rmse_mm']:.1f} mm  |  "
                f"<10 mm: {eval_metrics['pct_under_10mm']:.1f}%  |  "
                f"jerk: {eval_metrics['mean_jerk']:.4f}{orient_str}"
            )


# CurriculumCallback has been replaced by PhasedCurriculumCallback
# (training/phased_curriculum.py), which adds per-phase early stopping.

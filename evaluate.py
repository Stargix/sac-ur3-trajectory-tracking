"""
Evaluate a trained UR3 SAC policy and generate performance plots.

Usage
-----
    # Evaluate the best saved model (default)
    python evaluate.py

    # Specify a model file explicitly
    python evaluate.py --model weights/ur3_sac_final.zip

    # Run more evaluation episodes
    python evaluate.py --episodes 10

    # Change output directory
    python evaluate.py --output my_results

Outputs (written to --output directory)
----------------------------------------
    evaluation.png      Six-panel per-episode performance plot
    cross_episode.png   Error curves and bar chart across all episodes
    best_episode.mp4    Rendered video of the best-scoring episode
    metrics.json        Numerical summary of all episodes
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_orientation_exp_env import UR3OrientationExpEnv
from stable_baselines3 import SAC


# ---------------------------------------------------------------------------

def run(model_path, output_dir, n_episodes):
    os.makedirs(output_dir, exist_ok=True)

    model = SAC.load(model_path)
    env = UR3OrientationExpEnv(
        obs_noise_std=0.0,
        action_delay=0,
        traj_speed=0.4,
        traj_radius=0.08,
        orient_weight=1.0,
        render_mode="rgb_array",
    )

    all_dists, all_rewards = [], []
    best_ep, best_reward = None, -np.inf

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ee_list, tgt_list, rew_list, act_list, dist_list, frames = [], [], [], [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ee_list.append(info["ee_pos"])
            tgt_list.append(info["target_pos"])
            rew_list.append(reward)
            act_list.append(action)
            dist_list.append(info["dist"])
            frames.append(env.render())

        ep_reward = float(np.sum(rew_list))
        ep_dists = np.array(dist_list) * 1000

        all_dists.append(ep_dists)
        all_rewards.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_ep = dict(
                ee=np.array(ee_list), tgt=np.array(tgt_list),
                rew=np.array(rew_list), act=np.array(act_list),
                dists=ep_dists, frames=frames,
            )

        print(
            f"  Episode {ep + 1}/{n_episodes}  "
            f"mean error: {np.mean(ep_dists):.1f} mm  "
            f"total reward: {ep_reward:.1f}"
        )

    # ---- best episode plots ----
    ee, tgt, rew, acts, dists = (
        best_ep["ee"], best_ep["tgt"], best_ep["rew"],
        best_ep["act"], best_ep["dists"],
    )
    t = np.arange(len(dists)) * 0.01
    jerk = np.linalg.norm(np.diff(acts, n=2, axis=0), axis=1) if len(acts) > 2 else np.zeros(1)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("UR3 — Trajectory tracking evaluation", fontsize=13)

    axes[0, 0].plot(t, dists, color="#534ab7", linewidth=0.8)
    axes[0, 0].axhline(10, color="red", linestyle="--", alpha=0.5, label="10 mm target")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Error (mm)", title="Position error")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(tgt[:, 0], tgt[:, 1], "--", color="#534ab7", alpha=0.5, label="Target")
    axes[0, 1].plot(ee[:, 0], ee[:, 1], color="#ba7517", linewidth=1.2, label="End-effector")
    axes[0, 1].set(xlabel="X (m)", ylabel="Y (m)", title="XY trajectory")
    axes[0, 1].legend(fontsize=8); axes[0, 1].set_aspect("equal"); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(t, rew, color="#0f6e56", linewidth=0.8)
    axes[0, 2].set(xlabel="Time (s)", ylabel="Reward", title=f"Reward (total {np.sum(rew):.0f})")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].hist(dists, bins=50, color="#534ab7", alpha=0.7, edgecolor="white")
    axes[1, 0].axvline(np.mean(dists), color="red", linestyle="--",
                       label=f"mean {np.mean(dists):.1f} mm")
    axes[1, 0].set(xlabel="Error (mm)", ylabel="Count", title="Error distribution")
    axes[1, 0].legend(fontsize=8)

    for j in range(3):
        axes[1, 1].plot(t, acts[:, j], linewidth=0.8, alpha=0.7, label=f"J{j}")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Action", title="Joint commands (first 3)")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(t[2:], jerk, color="#0f6e56", linewidth=0.8)
    axes[1, 2].set(xlabel="Time (s)", ylabel="Jerk",
                   title=f"Action jerk (mean {np.mean(jerk):.4f})")
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation.png"), dpi=150)
    plt.close()

    # ---- cross-episode summary ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Cross-episode summary ({n_episodes} episodes)", fontsize=12)

    for i, d in enumerate(all_dists):
        axes[0].plot(np.arange(len(d)) * 0.01, d, alpha=0.5, linewidth=0.7,
                     label=f"Ep {i + 1}")
    axes[0].axhline(10, color="red", linestyle="--", alpha=0.5)
    axes[0].set(xlabel="Time (s)", ylabel="Error (mm)", title="Error per episode")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    means = [float(np.mean(d)) for d in all_dists]
    axes[1].bar(range(1, n_episodes + 1), means, color="#534ab7", alpha=0.75)
    axes[1].axhline(10, color="red", linestyle="--", label="10 mm target")
    axes[1].set(xlabel="Episode", ylabel="Mean error (mm)", title="Mean error per episode")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_episode.png"), dpi=150)
    plt.close()

    # ---- video ----
    if best_ep["frames"]:
        try:
            import imageio
            imageio.mimsave(os.path.join(output_dir, "best_episode.mp4"),
                            best_ep["frames"], fps=50)
        except Exception as exc:
            print(f"  Could not save video: {exc}")

    # ---- metrics ----
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "n_episodes": n_episodes,
        "best_episode": {
            "rmse_mm": float(np.sqrt(np.mean(dists ** 2))),
            "mean_error_mm": float(np.mean(dists)),
            "max_error_mm": float(np.max(dists)),
            "pct_under_5mm": float(np.mean(dists < 5) * 100),
            "pct_under_10mm": float(np.mean(dists < 10) * 100),
            "pct_under_20mm": float(np.mean(dists < 20) * 100),
            "mean_jerk": float(np.mean(jerk)),
            "total_reward": float(np.sum(rew)),
        },
        "all_episodes": {
            "mean_rewards": [float(r) for r in all_rewards],
            "mean_errors_mm": means,
        },
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nResults written to {output_dir}/")
    print(f"  RMSE:     {metrics['best_episode']['rmse_mm']:.1f} mm")
    print(f"  < 10 mm:  {metrics['best_episode']['pct_under_10mm']:.1f}%")
    print(f"  Jerk:     {metrics['best_episode']['mean_jerk']:.4f}")


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="UR3 SAC — evaluation")
    p.add_argument("--model", default="weights/best_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.output, args.episodes)

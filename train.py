"""
Train a Soft Actor-Critic agent on the UR3 position and orientation
tracking task.

The agent controls the 6-DOF UR3 arm to keep its end-effector on a
Bernoulli lemniscate trajectory while simultaneously maintaining a fixed
tool orientation (the home end-effector pose computed at startup via FK).

Curriculum
----------
Training is divided into four phases managed by PhasedCurriculumCallback.
Each phase advances automatically (early stopping) when its performance
thresholds are sustained for the required number of consecutive evaluation
intervals; if the thresholds are never met, the phase ends at its hard
step limit and training proceeds to the next phase regardless.

Phase 1 — Position bootstrap (target: ≤5 mm mean error, 4 intervals)
    radius=0.04 m  speed=0.2  noise=0  delay=0  orient_weight=0.0
    Orientation is in the observation but carries no reward weight yet.
    The agent replicates the proven position-tracking curriculum without
    any extra complexity.

Phase 2 — Full trajectory, light orientation (target: ≤3 mm + ≤25°, 3)
    radius=0.08 m  speed=0.3  noise=0  delay=0  orient_weight=0.2
    Full-size lemniscate. Orientation reward introduced at 20% weight so
    the agent learns to begin coordinating wrist joints without losing
    position accuracy.

Phase 3 — Speed and noise, heavier orientation (target: ≤2 mm + ≤12°, 4)
    radius=0.08 m  speed=0.4  noise=0.0003  delay=0  orient_weight=0.6
    Real operating speed. Sensor noise added for sim-to-real robustness.
    Orientation reward raised to 60% weight.

Phase 4 — Full difficulty (target: ≤2 mm + ≤8°, 5)
    radius=0.08 m  speed=0.4  noise=0.0005  delay=1  orient_weight=1.0
    Action delay simulates real hardware latency. Full orientation weight.
    Training stops when thresholds are sustained, or at --steps steps.

Observation space: 35 dimensions
    [0:27]   Identical to position-only baseline (normalised)
    [27:31]  Current EE orientation quaternion [w, x, y, z]
    [31:35]  Target orientation quaternion     [w, x, y, z]

Usage
-----
    python train.py                 # 1.2 M step budget, early stopping active
    python train.py --steps 800000  # custom step budget
    python train.py --resume        # resume from latest checkpoint
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_orientation_env import UR3OrientationEnv
from training.callbacks import DebugCallback
from training.phased_curriculum import PhasedCurriculumCallback

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DIR_CHECKPOINTS = "checkpoints"
DIR_WEIGHTS     = "weights"
DIR_LOGS        = "logs"
DIR_DEBUGS      = "debugs"

EVAL_FREQ  = 25_000
SAVE_FREQ  = 50_000

# ---------------------------------------------------------------------------
# Curriculum phases
# Designed from empirical data: the position-only model converged to
# 99% <5mm at step 200K with these env parameters.
# ---------------------------------------------------------------------------
PHASES = [
    {
        "name": "Phase 1 — Position Bootstrap",
        "max_steps": 200_000,
        "env_kwargs": {
            "traj_radius":   0.04,
            "traj_speed":    0.2,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 0.0,
            "orient_reward_scale": 1.0,
        },
        "thresholds": {
            "eval.mean_dist_mm": (5.0, "below"),
        },
        "patience": 4,
    },
    {
        "name": "Phase 2 — Full Trajectory (Position Only)",
        "max_steps": 450_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 0.0,
            "orient_reward_scale": 1.0,
        },
        "thresholds": {
            "eval.mean_dist_mm": (4.0, "below"),
        },
        "patience": 3,
    },
    {
        "name": "Phase 3 — Orientation Discovery (Wide Scale)",
        "max_steps": 750_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 1.0,
            "orient_reward_scale": 1.0,  # Wide curve for discovery
        },
        "thresholds": {
            "eval.mean_dist_mm":        (4.0,  "below"),
            "eval.mean_orient_error_deg": (15.0, "below"),
        },
        "patience": 3,
    },
    {
        "name": "Phase 4 — Orientation Refinement (Medium Scale)",
        "max_steps": 1_000_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 1.0,
            "orient_reward_scale": 0.5,  # Tightening the net
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.0,  "below"),
            "eval.mean_orient_error_deg": (8.0, "below"),
        },
        "patience": 4,
    },
    {
        "name": "Phase 5 — Speed & Noise (High Precision)",
        "max_steps": 1_300_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.4,
            "obs_noise_std": 0.0003,
            "action_delay":  0,
            "orient_weight": 1.0,
            "orient_reward_scale": 0.3,  # Sharpening further
        },
        "thresholds": {
            "eval.mean_dist_mm":        (2.5,  "below"),
            "eval.mean_orient_error_deg": (6.0, "below"),
        },
        "patience": 4,
    },
    {
        "name": "Phase 6 — Full Difficulty with Action Delay",
        "max_steps": None,          # runs until step budget is exhausted
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.4,
            "obs_noise_std": 0.0005,
            "action_delay":  1,
            "orient_weight": 1.0,
            "orient_reward_scale": 0.2,  # Full precision
        },
        "thresholds": {
            "eval.mean_dist_mm":        (2.5, "below"),
            "eval.mean_orient_error_deg": (5.0, "below"),
        },
        "patience": 5,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None
    zips = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if not zips:
        return None
    zips.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, zips[-1])


def parse_args():
    p = argparse.ArgumentParser(
        description="UR3 position + orientation tracking — SAC training"
    )
    p.add_argument("--steps", type=int, default=1_500_000,
                   help="Total training timesteps (default: 1 500 000)")
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    p.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    for d in (DIR_CHECKPOINTS, DIR_WEIGHTS, DIR_LOGS, DIR_DEBUGS):
        os.makedirs(d, exist_ok=True)

    print("=" * 62)
    print("  UR3 SAC — Position + Orientation Tracking")
    print("=" * 62)
    print(f"  Steps:       {args.steps:,}")
    print(f"  Obs dims:    35  (27 position + 8 orientation)")
    print(f"  Eval freq:   {args.eval_freq:,}")
    print(f"  Phases:      {len(PHASES)} (with per-phase early stopping)")
    print("=" * 62 + "\n")

    # ---- Environments (Phase 1 settings at startup) ----
    p1 = PHASES[0]["env_kwargs"]

    train_env = UR3OrientationEnv(
        obs_noise_std=p1["obs_noise_std"],
        action_delay=p1["action_delay"],
        traj_radius=p1["traj_radius"],
        traj_speed=p1["traj_speed"],
        orient_weight=p1["orient_weight"],
        orient_reward_scale=p1["orient_reward_scale"],
    )
    eval_env = UR3OrientationEnv(
        obs_noise_std=0.0,
        action_delay=0,
        traj_radius=p1["traj_radius"],
        traj_speed=p1["traj_speed"],
        orient_weight=p1["orient_weight"],
        orient_reward_scale=p1["orient_reward_scale"],
        render_mode="rgb_array",
    )

    check_env(train_env, warn=True)

    # ---- Model ----
    if args.resume:
        ckpt = latest_checkpoint(DIR_CHECKPOINTS)
        if ckpt:
            print(f"Resuming from {ckpt}\n")
            model = SAC.load(ckpt, env=train_env)
        else:
            print("No checkpoint found — starting from scratch.\n")
            args.resume = False

    if not args.resume:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            verbose=1,
            tensorboard_log=DIR_LOGS,
        )

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=DIR_CHECKPOINTS,
            name_prefix="ur3_sac",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=DIR_WEIGHTS,
            log_path=DIR_LOGS,
            eval_freq=args.eval_freq,
            n_eval_episodes=5,
            deterministic=True,
        ),
        DebugCallback(
            eval_env=eval_env,
            debug_dir=DIR_DEBUGS,
            eval_freq=args.eval_freq,
        ),
        PhasedCurriculumCallback(
            train_env=train_env,
            eval_env=eval_env,
            phases=PHASES,
            debug_dir=DIR_DEBUGS,
            check_freq=args.eval_freq,
        ),
    ]

    # ---- Train ----
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    model.save(os.path.join(DIR_WEIGHTS, "ur3_sac_final"))

    print(f"\n{'=' * 62}")
    print(f"  Training complete ({elapsed / 3600:.1f} h)")
    print(f"  Final model: {DIR_WEIGHTS}/ur3_sac_final.zip")
    print(f"  Best model:  {DIR_WEIGHTS}/best_model.zip")
    print(f"  Debug plots: {DIR_DEBUGS}/")
    print("=" * 62)


if __name__ == "__main__":
    main()

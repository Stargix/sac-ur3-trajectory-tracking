"""
Train a Soft Actor-Critic agent on the UR3 lemniscate tracking task.

Usage
-----
    # Full 1 M-step training run (default)
    python train.py

    # Custom step count
    python train.py --steps 500000

    # Resume from the most recent checkpoint
    python train.py --resume

    # Disable curriculum (fixed difficulty throughout)
    python train.py --no-curriculum

Directory layout produced
-------------------------
    checkpoints/    Periodic model snapshots (every --save-freq steps)
    weights/        best_model.zip updated whenever eval improves
    logs/           TensorBoard event files
    debugs/         Per-interval evaluation snapshots (plots + metrics)
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_tracking_env import UR3TrackingEnv
from training.callbacks import CurriculumCallback, DebugCallback

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TOTAL_STEPS = 1_000_000
EVAL_FREQ = 25_000
SAVE_FREQ = 50_000

DIR_CHECKPOINTS = "checkpoints"
DIR_WEIGHTS = "weights"
DIR_LOGS = "logs"
DIR_DEBUGS = "debugs"


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
    p = argparse.ArgumentParser(description="UR3 SAC trajectory tracking — training")
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    p.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    for d in (DIR_CHECKPOINTS, DIR_WEIGHTS, DIR_LOGS, DIR_DEBUGS):
        os.makedirs(d, exist_ok=True)

    # Environments
    # Training starts at phase-1 difficulty; CurriculumCallback updates it.
    train_env = UR3TrackingEnv(
        obs_noise_std=0.0,
        action_delay=0,
        traj_radius=0.04,
        traj_speed=0.2,
    )
    eval_env = UR3TrackingEnv(
        obs_noise_std=0.0,
        action_delay=0,
        traj_radius=0.04,
        traj_speed=0.2,
        render_mode="rgb_array",
    )

    check_env(train_env, warn=True)

    # Callbacks
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
    ]

    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(train_env, eval_env))

    # Model
    if args.resume:
        ckpt = latest_checkpoint(DIR_CHECKPOINTS)
        if ckpt:
            print(f"Resuming from {ckpt}")
            model = SAC.load(ckpt, env=train_env)
        else:
            print("No checkpoint found — starting from scratch.")
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

    print(f"\nTraining for {args.steps:,} steps — debug snapshots every {args.eval_freq:,} steps\n")
    t0 = time.time()

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    model.save(os.path.join(DIR_WEIGHTS, "ur3_sac_final"))
    print(f"\nTraining complete ({elapsed / 3600:.1f} h). Model saved to {DIR_WEIGHTS}/")


if __name__ == "__main__":
    main()

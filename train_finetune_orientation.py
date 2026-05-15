"""
Fine-tune the trained UR3 agent specifically for high-precision orientation.

This script loads a pre-trained model (from weights/best_model.zip) and
subjects it to a new, sharper orientation reward signal using
UR3OrientationExpEnv.

Fine-tuning Strategy:
1. Load model with 35-dim observation space.
2. Use a reduced learning rate (e.g., 5e-5) to preserve position tracking.
3. Skip position bootstrap; start directly with full lemniscate and high
   orientation weight.
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

sys.path.insert(0, os.path.dirname(__file__))
from envs import UR3OrientationExpEnv
from training.callbacks import DebugCallback
from training.phased_curriculum import PhasedCurriculumCallback

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DIR_CHECKPOINTS = "checkpoints_finetune"
DIR_WEIGHTS     = "weights_finetune"
DIR_LOGS        = "logs_finetune"
DIR_DEBUGS      = "debugs_finetune"

EVAL_FREQ  = 25_000
SAVE_FREQ  = 50_000

# ---------------------------------------------------------------------------
# Fine-tune Phases
# ---------------------------------------------------------------------------
PHASES = [
    {
        "name": "FT Phase 1 — Orientation Focus (Exponential Reward)",
        "max_steps": 150_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 0.8,
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.0,  "below"),
            "eval.mean_orient_error_deg": (8.0, "below"),
        },
        "patience": 4,
    },
    {
        "name": "FT Phase 2 — Full Difficulty Refinement",
        "max_steps": None,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.4,
            "obs_noise_std": 0.0005,
            "action_delay":  1,
            "orient_weight": 1.0,
        },
        "thresholds": {
            "eval.mean_dist_mm":        (2.5, "below"),
            "eval.mean_orient_error_deg": (5.0, "below"),
        },
        "patience": 5,
    },
]

def latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None
    zips = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if not zips:
        return None
    zips.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, zips[-1])


def main():
    parser = argparse.ArgumentParser(description="UR3 Orientation Fine-tuning")
    parser.add_argument("--model", default="weights/best_model.zip", help="Base model to load")
    parser.add_argument("--steps", type=int, default=300_000, help="Total FT steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for FT")
    parser.add_argument("--resume", action="store_true", help="Resume from latest FT checkpoint")
    args = parser.parse_args()

    for d in (DIR_CHECKPOINTS, DIR_WEIGHTS, DIR_LOGS, DIR_DEBUGS):
        os.makedirs(d, exist_ok=True)

    print("=" * 62)
    print("  UR3 SAC — ORIENTATION FINE-TUNING")
    print("=" * 62)
    print(f"  Base Model: {args.model}")
    print(f"  Target LR:  {args.lr}")
    print(f"  Phases:     {len(PHASES)}")
    print("=" * 62 + "\n")

    # Initialise Env with Phase 1 settings
    p1 = PHASES[0]["env_kwargs"]
    train_env = UR3OrientationExpEnv(**p1)
    eval_env = UR3OrientationExpEnv(**p1, render_mode="rgb_array")

    # Load model
    if args.resume:
        ckpt = latest_checkpoint(DIR_CHECKPOINTS)
        if ckpt:
            print(f"Resuming FT from {ckpt}...")
            model = SAC.load(ckpt, env=train_env)
        else:
            print("No FT checkpoint found — starting from base model.")
            args.resume = False

    if not args.resume:
        if not os.path.exists(args.model):
            print(f"Error: Base model {args.model} not found.")
            return
        print(f"Loading weights from base model {args.model}...")
        model = SAC.load(args.model, env=train_env)
    
    # Update learning rate for fine-tuning
    model.learning_rate = args.lr
    
    # Re-initialise optimizer with the new learning rate
    # (SAC.load preserves the original LR from the zip unless we force it)
    model.ent_coef = "auto" # ensure auto-entropy is still active

    # Callbacks
    callbacks = [
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=DIR_CHECKPOINTS, name_prefix="ur3_ft"),
        EvalCallback(eval_env, best_model_save_path=DIR_WEIGHTS, log_path=DIR_LOGS,
                     eval_freq=EVAL_FREQ, n_eval_episodes=5, deterministic=True),
        DebugCallback(eval_env=eval_env, debug_dir=DIR_DEBUGS, eval_freq=EVAL_FREQ),
        PhasedCurriculumCallback(train_env=train_env, eval_env=eval_env, phases=PHASES,
                                  debug_dir=DIR_DEBUGS, check_freq=EVAL_FREQ),
    ]

    # Train
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    
    elapsed = time.time() - t0
    model.save(os.path.join(DIR_WEIGHTS, "ur3_sac_ft_final"))
    
    print(f"\nFine-tuning complete in {elapsed/3600:.1f} hours.")
    print(f"Final model saved to {DIR_WEIGHTS}/")

if __name__ == "__main__":
    main()

"""
Fine-tune the UR3 agent for high-precision orientation without aggressive perturbations.

This script loads the checkpoint from step 700,000 (where position tracking was
perfect and orientation was being discovered) and applies a softer curriculum
focused on orientation precision, without action delay and with minimal noise.
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

sys.path.insert(0, os.path.dirname(__file__))
from envs import UR3OrientationEnv
from training.callbacks import DebugCallback
from training.phased_curriculum import PhasedCurriculumCallback

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DIR_CHECKPOINTS = "checkpoints_soft"
DIR_WEIGHTS     = "weights_soft"
DIR_LOGS        = "logs_soft"
DIR_DEBUGS      = "debugs_soft"

EVAL_FREQ  = 20_000
SAVE_FREQ  = 40_000

# ---------------------------------------------------------------------------
# Soft Fine-tune Phases
# ---------------------------------------------------------------------------
PHASES = [
    {
        "name": "Soft FT Phase 1 — Orientation Refinement (Scale 0.6)",
        "max_steps": 150_000,
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0,
            "action_delay":  0,
            "orient_weight": 1.0,
            "orient_reward_scale": 0.6,  # Bridge between discovery (1.0) and sharp (0.2)
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.5,  "below"),
            "eval.mean_orient_error_deg": (10.0, "below"),
        },
        "patience": 3,
    },
    {
        "name": "Soft FT Phase 2 — High Precision (Scale 0.3)",
        "max_steps": None,          # runs until budget is exhausted
        "env_kwargs": {
            "traj_radius":   0.08,
            "traj_speed":    0.3,
            "obs_noise_std": 0.0001, # minimal noise for robustness
            "action_delay":  0,
            "orient_weight": 1.0,
            "orient_reward_scale": 0.3,  # Sharper for fine control
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.0, "below"),
            "eval.mean_orient_error_deg": (5.0, "below"),
        },
        "patience": 4,
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
    parser = argparse.ArgumentParser(description="UR3 Soft Orientation Fine-tuning")
    parser.add_argument("--model", default="checkpoints/ur3_sac_1200000_steps.zip", 
                        help="Path to the checkpoint before action delay (step 1.2M)")
    parser.add_argument("--steps", type=int, default=300_000, help="Total FT steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for FT")
    parser.add_argument("--resume", action="store_true", help="Resume from latest soft FT checkpoint")
    args = parser.parse_args()

    for d in (DIR_CHECKPOINTS, DIR_WEIGHTS, DIR_LOGS, DIR_DEBUGS):
        os.makedirs(d, exist_ok=True)

    print("=" * 62)
    print("  UR3 SAC — SOFT ORIENTATION FINE-TUNING")
    print("=" * 62)
    print(f"  Base Model: {args.model}")
    print(f"  Target LR:  {args.lr}")
    print(f"  Phases:     {len(PHASES)}")
    print("=" * 62 + "\n")

    # Initialise Env with Phase 1 settings
    p1 = PHASES[0]["env_kwargs"]
    train_env = UR3OrientationEnv(
        traj_radius=p1["traj_radius"],
        traj_speed=p1["traj_speed"],
        obs_noise_std=p1["obs_noise_std"],
        action_delay=p1["action_delay"],
        orient_weight=p1["orient_weight"],
        orient_reward_scale=p1["orient_reward_scale"],
    )
    eval_env = UR3OrientationEnv(
        traj_radius=p1["traj_radius"],
        traj_speed=p1["traj_speed"],
        obs_noise_std=0.0,
        action_delay=0,
        orient_weight=p1["orient_weight"],
        orient_reward_scale=p1["orient_reward_scale"],
        render_mode="rgb_array",
    )

    # Load model
    if args.resume:
        ckpt = latest_checkpoint(DIR_CHECKPOINTS)
        if ckpt:
            print(f"Resuming FT from {ckpt}...")
            model = SAC.load(ckpt, env=train_env)
        else:
            print("No soft FT checkpoint found — starting from base model.")
            args.resume = False

    if not args.resume:
        if not os.path.exists(args.model):
            print(f"Error: Base model {args.model} not found.")
            print("Please ensure checkpoints/ur3_sac_700000_steps.zip exists.")
            return
        print(f"Loading weights from {args.model}...")
        model = SAC.load(args.model, env=train_env)
    
    # Update learning rate for fine-tuning
    model.learning_rate = args.lr
    model.ent_coef = "auto"

    # Callbacks
    callbacks = [
        CheckpointCallback(save_freq=SAVE_FREQ, save_path=DIR_CHECKPOINTS, name_prefix="ur3_soft_ft"),
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
    model.save(os.path.join(DIR_WEIGHTS, "ur3_sac_soft_ft_final"))
    
    print(f"\nFine-tuning complete in {elapsed/3600:.1f} hours.")
    print(f"Final model saved to {DIR_WEIGHTS}/")

if __name__ == "__main__":
    main()

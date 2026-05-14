"""
Fine-tune a pre-trained UR3 SAC policy on randomised spline trajectories.

This script loads an existing model (trained on the lemniscate) and
continues training in an environment where the target path changes at
every episode reset. The goal is to transfer the arm's control knowledge
to generalise over arbitrary smooth trajectories.

Training stops automatically when the policy achieves the configured
performance threshold for several consecutive evaluation intervals
(early stopping), or when the maximum step count is reached.

Usage
-----
    # Fine-tune from best_model with default settings
    python train_finetune.py

    # Specify source model and increase max steps
    python train_finetune.py --model weights/ur3_sac_final.zip --steps 400000

    # Disable early stopping and run for a fixed budget
    python train_finetune.py --no-early-stop --steps 200000

    # Adjust early stopping target (default: 90% of steps within 10 mm)
    python train_finetune.py --stop-threshold 85.0 --stop-patience 4

Directory layout produced
-------------------------
    checkpoints_ft/    Periodic snapshots during fine-tuning
    weights/           best_model_ft.zip updated on eval improvement
    logs_ft/           TensorBoard event files
    debugs_ft/         Per-interval evaluation snapshots (plots + metrics)
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_random_env import UR3RandomEnv
from training.callbacks import DebugCallback
from training.early_stopping import EarlyStoppingCallback

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "weights/best_model.zip"
DEFAULT_STEPS = 250_000
EVAL_FREQ = 25_000
SAVE_FREQ = 50_000

DIR_CHECKPOINTS = "checkpoints_ft"
DIR_WEIGHTS = "weights"
DIR_LOGS = "logs_ft"
DIR_DEBUGS = "debugs_ft"

# Early stopping defaults — based on observed lemniscate performance:
# the base model achieves ~99% <5mm. A reasonable fine-tuning target for
# random trajectories is 90% <10mm sustained for 5 consecutive intervals.
DEFAULT_STOP_METRIC = "eval.pct_under_10mm"
DEFAULT_STOP_THRESHOLD = 90.0
DEFAULT_STOP_PATIENCE = 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune UR3 SAC on randomised spline trajectories"
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Source model to fine-tune (default: {DEFAULT_MODEL})"
    )
    p.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Maximum fine-tuning timesteps (default: {DEFAULT_STEPS:,})"
    )
    p.add_argument(
        "--eval-freq", type=int, default=EVAL_FREQ,
        help="Steps between evaluation snapshots"
    )
    p.add_argument(
        "--save-freq", type=int, default=SAVE_FREQ,
        help="Steps between checkpoint saves"
    )
    p.add_argument(
        "--no-early-stop", action="store_true",
        help="Disable early stopping — run for exactly --steps steps"
    )
    p.add_argument(
        "--stop-threshold", type=float, default=DEFAULT_STOP_THRESHOLD,
        help=(
            f"Early stopping threshold for {DEFAULT_STOP_METRIC} "
            f"(default: {DEFAULT_STOP_THRESHOLD})"
        )
    )
    p.add_argument(
        "--stop-patience", type=int, default=DEFAULT_STOP_PATIENCE,
        help=(
            "Number of consecutive intervals threshold must be met "
            f"(default: {DEFAULT_STOP_PATIENCE})"
        )
    )
    p.add_argument(
        "--workspace-radius", type=float, default=0.10,
        help="Maximum trajectory radius from EE home (default: 0.10 m)"
    )
    p.add_argument(
        "--min-speed", type=float, default=0.2,
        help="Minimum trajectory speed in m/s (default: 0.2)"
    )
    p.add_argument(
        "--max-speed", type=float, default=0.5,
        help="Maximum trajectory speed in m/s (default: 0.5)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: model file not found: {args.model}")
        sys.exit(1)

    for d in (DIR_CHECKPOINTS, DIR_WEIGHTS, DIR_LOGS, DIR_DEBUGS):
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  UR3 SAC Fine-tuning — Random Trajectory Generalisation")
    print("=" * 60)
    print(f"  Source model:      {args.model}")
    print(f"  Max steps:         {args.steps:,}")
    print(f"  Workspace radius:  {args.workspace_radius} m")
    print(f"  Speed range:       [{args.min_speed}, {args.max_speed}] m/s")
    if not args.no_early_stop:
        print(
            f"  Early stop:        {DEFAULT_STOP_METRIC} >= "
            f"{args.stop_threshold} for {args.stop_patience} intervals"
        )
    print("=" * 60 + "\n")

    # ---- Environments ----
    train_env = UR3RandomEnv(
        obs_noise_std=0.0003,
        action_delay=0,
        workspace_radius=args.workspace_radius,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
    )
    eval_env = UR3RandomEnv(
        obs_noise_std=0.0,
        action_delay=0,
        workspace_radius=args.workspace_radius,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        render_mode="rgb_array",
    )

    check_env(train_env, warn=True)

    # ---- Load and adapt model ----
    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, env=train_env)

    # Reduce learning rate for fine-tuning — large steps risk destroying
    # the pre-trained representation
    model.learning_rate = 1e-4
    model.learning_starts = 0   # start updating immediately

    # ---- Callbacks ----
    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=DIR_CHECKPOINTS,
            name_prefix="ur3_sac_ft",
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

    if not args.no_early_stop:
        callbacks.append(
            EarlyStoppingCallback(
                debug_dir=DIR_DEBUGS,
                metric=DEFAULT_STOP_METRIC,
                threshold=args.stop_threshold,
                mode="above",
                patience=args.stop_patience,
                check_freq=args.eval_freq,
            )
        )

    # ---- Train ----
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    model.save(os.path.join(DIR_WEIGHTS, "ur3_sac_ft_final"))

    print(f"\n{'=' * 60}")
    print(f"  Fine-tuning complete ({elapsed / 3600:.1f} h)")
    print(f"  Model saved to {DIR_WEIGHTS}/ur3_sac_ft_final.zip")
    print(f"  Evaluation snapshots in {DIR_DEBUGS}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

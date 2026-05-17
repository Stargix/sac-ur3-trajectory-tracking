"""
Fine-tune the trained UR3 agent on randomised spline trajectories
and orientation tracking.

This script loads the best model from the orientation-exp-finetune branch
(weights_finetune/best_model.zip) and trains it on random smooth paths
using UR3OrientationRandomEnv.
"""

import argparse
import os
import sys
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_orientation_random_env import UR3OrientationRandomEnv
from training.callbacks import DebugCallback
from training.phased_curriculum import PhasedCurriculumCallback

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DIR_CHECKPOINTS = "checkpoints_random"
DIR_WEIGHTS     = "weights_random"
DIR_LOGS        = "logs_random"
DIR_DEBUGS      = "debugs_random"

EVAL_FREQ  = 25_000
SAVE_FREQ  = 50_000

# ---------------------------------------------------------------------------
# Fine-tune Phases
# ---------------------------------------------------------------------------
PHASES = [
    {
        "name": "Random FT Phase 1 — Simplification (Slow & Small)",
        "max_steps": 150_000,
        "env_kwargs": {
            "workspace_radius": 0.06,
            "min_speed": 0.1,
            "max_speed": 0.2,
            "obs_noise_std": 0.0003,
            "action_delay": 0,
            "orient_reward_scale": 0.4,
            "n_control_points_range": (3, 5),
        },
        "thresholds": {
            "eval.mean_dist_mm":        (4.0,  "below"),
            "eval.mean_orient_error_deg": (10.0, "below"),
        },
        "patience": 3,
    },
    {
        "name": "Random FT Phase 2 — Medium Speed & Full Radius",
        "max_steps": 300_000,
        "env_kwargs": {
            "workspace_radius": 0.10,
            "min_speed": 0.2,
            "max_speed": 0.4,
            "obs_noise_std": 0.0003,
            "action_delay": 0,
            "orient_reward_scale": 0.4,
            "n_control_points_range": (4, 8),
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.5, "below"),
            "eval.mean_orient_error_deg": (8.0, "below"),
        },
        "patience": 3,
    },
    {
        "name": "Random FT Phase 3 — High Speed",
        "max_steps": 500_000,
        "env_kwargs": {
            "workspace_radius": 0.10,
            "min_speed": 0.3,
            "max_speed": 0.6,
            "obs_noise_std": 0.0003,
            "action_delay": 0,
            "orient_reward_scale": 0.4,
            "n_control_points_range": (4, 8),
        },
        "thresholds": {
            "eval.mean_dist_mm":        (3.0, "below"),
            "eval.mean_orient_error_deg": (7.0, "below"),
        },
        "patience": 4,
    },
]

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(DIR_CHECKPOINTS, exist_ok=True)
    os.makedirs(DIR_WEIGHTS, exist_ok=True)
    os.makedirs(DIR_LOGS, exist_ok=True)
    os.makedirs(DIR_DEBUGS, exist_ok=True)

    # Use a dummy initial config for the vector/eval envs;
    # the PhasedCurriculumCallback will apply the proper values immediately.
    dummy_kwargs = {
        "workspace_radius": 0.10,
        "min_speed": 0.2,
        "max_speed": 0.4,
        "obs_noise_std": 0.0003,
        "action_delay": 0,
        "orient_reward_scale": 0.4,
    }

    # Create environments
    # Note: For evaluation, we use fixed seeds or fixed trajectories to make it comparable!
    # But here we use the same environment class.
    env = UR3OrientationRandomEnv(render_mode=None, **dummy_kwargs)
    eval_env = UR3OrientationRandomEnv(render_mode=None, **dummy_kwargs)

    # Load the 300k fine-tuned model from the exp branch
    model_path = "weights_finetune/ur3_sac_ft_final.zip"
    print(f"Loading pre-trained model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        sys.exit(1)

    model = SAC.load(
        model_path,
        env=env,
        tensorboard_log=DIR_LOGS,
        learning_rate=1.5e-4,  # Increased LR for fine-tuning on random paths
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=DIR_WEIGHTS,
        log_path=DIR_LOGS,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True
    )
    
    debug_callback = DebugCallback(
        eval_env=eval_env,
        debug_dir=DIR_DEBUGS,
        eval_freq=EVAL_FREQ
    )
    
    curriculum_callback = PhasedCurriculumCallback(
        train_env=env,
        eval_env=eval_env,
        phases=PHASES,
        debug_dir=DIR_DEBUGS,
        check_freq=EVAL_FREQ,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=DIR_CHECKPOINTS,
        name_prefix="ur3_sac_random",
    )

    callbacks = [eval_callback, debug_callback, curriculum_callback, checkpoint_callback]

    # Train
    print(f"Starting fine-tuning on random trajectories for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        reset_num_timesteps=True,  # Reset timesteps for the new run
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(DIR_WEIGHTS, "ur3_sac_random_final.zip")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()

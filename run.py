"""
Interactive policy rollout with live MuJoCo viewer.

Opens a native MuJoCo window and runs the trained policy in real time.
The viewer is passive (read-only camera control); press Escape or close
the window to exit.

Usage
-----
    python run.py                                   # uses weights/best_model.zip
    python run.py --model weights/ur3_sac_final.zip
    python run.py --speed 0.5                       # slow-motion (0.5x real time)
    python run.py --episodes 3                      # stop after 3 episodes
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from envs.ur3_tracking_env import UR3TrackingEnv
from stable_baselines3 import SAC


def parse_args():
    p = argparse.ArgumentParser(description="UR3 SAC — interactive visualisation")
    p.add_argument("--model", default="weights/best_model.zip",
                   help="Path to the SAC model zip file")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (1.0 = real time, 0.5 = half speed)")
    p.add_argument("--episodes", type=int, default=0,
                   help="Number of episodes to run (0 = run until window is closed)")
    p.add_argument("--traj-radius", type=float, default=0.08,
                   help="Lemniscate radius in metres (default: 0.08)")
    p.add_argument("--traj-speed", type=float, default=0.4,
                   help="Trajectory angular speed (default: 0.4)")
    return p.parse_args()


def run(model_path, speed, max_episodes, traj_radius, traj_speed):
    print(f"Loading model: {model_path}")
    policy = SAC.load(model_path)

    env = UR3TrackingEnv(
        traj_radius=traj_radius,
        traj_speed=traj_speed,
        obs_noise_std=0.0,
        action_delay=0,
    )

    control_dt = env.dt          # seconds per control step (0.01 s → 100 Hz)
    sleep_dt = control_dt / max(speed, 1e-3)

    episode = 0
    print("Launching MuJoCo viewer — close the window or press Ctrl+C to exit.\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Adjust default camera for a better viewing angle
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 140

        while viewer.is_running():
            if max_episodes > 0 and episode >= max_episodes:
                break

            obs, _ = env.reset()
            done = False
            step = 0
            episode_dists = []

            print(f"Episode {episode + 1}" +
                  (f"/{max_episodes}" if max_episodes > 0 else "") + " started.")

            while not done and viewer.is_running():
                t_start = time.perf_counter()

                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_dists.append(info["dist_mm"])

                viewer.sync()
                step += 1

                # Maintain real-time pacing
                elapsed = time.perf_counter() - t_start
                remaining = sleep_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            if episode_dists:
                print(
                    f"  Steps: {step}  |  "
                    f"Mean error: {np.mean(episode_dists):.1f} mm  |  "
                    f"Max error: {np.max(episode_dists):.1f} mm"
                )
            episode += 1

    env.close()
    print("Viewer closed.")


if __name__ == "__main__":
    args = parse_args()
    run(
        model_path=args.model,
        speed=args.speed,
        max_episodes=args.episodes,
        traj_radius=args.traj_radius,
        traj_speed=args.traj_speed,
    )

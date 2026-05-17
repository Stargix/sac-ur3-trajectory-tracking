import os
import json
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import SAC
from envs import UR3OrientationExpEnv

def record():
    print("Loading best model...")
    # Load the best model as requested
    model = SAC.load("weights_finetune/best_model.zip")
    
    print("Initializing environment (video recording mode)...")
    # USE EXACTLY THE SAME PARAMETERS AS IN PHASE 1 (Best Model)
    env = UR3OrientationExpEnv(
        render_mode="rgb_array",
        traj_radius=0.08,
        traj_speed=0.3,
        obs_noise_std=0.0,
        action_delay=0,
        orient_weight=1.0,
        orient_reward_scale=1.0
    )
    obs, info = env.reset()
    
    trajectory = []
    ee_positions = []
    frames = []
    
    done = False
    step_count = 0
    # At speed 0.3, the period is 2*pi / 0.3 = 20.94 seconds.
    # At 100 Hz, we need exactly 2094 steps to complete one perfect loop!
    print("Waiting 2 seconds for the robot to reach the trajectory, then recording 2094 steps...")
    
    while len(trajectory) < 2094:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # WARM-UP: Skip the first 200 steps
        if step_count < 200:
            step_count += 1
            continue
        
        # Record joints for playback
        real_joints = env.unwrapped.data.qpos[:6].tolist()
        trajectory.append(real_joints)
        
        # Record EE Cartesian position (X, Y) for visualization
        ee_pos = env.unwrapped.data.site_xpos[env.unwrapped.ee_site_id][:2].tolist()
        ee_positions.append(ee_pos)
        
        # Capture EVERY frame for video to avoid "trompicones" (aliasing at 25fps)
        frame = env.render()
        frames.append(frame)
            
        if terminated:
            print("Environment terminated early (collision or out of bounds)!")
            break
            
        step_count += 1
        
    # Save the trajectory to a JSON file
    os.makedirs("results", exist_ok=True)
    with open("results/recorded_trajectory.json", "w") as f:
        json.dump(trajectory, f)
        
    print(f"Success! Recorded {len(trajectory)} points to results/recorded_trajectory.json")
    
    # Generate visualization
    print("Generating path plot...")
    plt.figure(figsize=(8, 6))
    x_coords = [p[0] for p in ee_positions]
    y_coords = [p[1] for p in ee_positions]
    plt.plot(x_coords, y_coords, label="Executed Path", color="blue")
    plt.scatter(x_coords[0], y_coords[0], color="green", label="Start", zorder=5)
    plt.scatter(x_coords[-1], y_coords[-1], color="red", label="End", zorder=5)
    plt.title("Recorded End-Effector Trajectory (XY Plane)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    
    plot_path = "results/recorded_trajectory_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Save video
    print("Saving video...")
    video_path = "results/recorded_trajectory.mp4"
    imageio.mimsave(video_path, frames, fps=100)
    print(f"Video saved to {video_path}")
    
    env.close()

if __name__ == "__main__":
    record()

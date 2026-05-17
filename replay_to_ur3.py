import socket
import time
import json

# IP address of the real UR3 robot
HOST = "10.10.73.23X"
# Port for the primary/secondary server
PORT = 30002

def replay():
    # 1. Load the recorded trajectory
    try:
        with open("results/recorded_trajectory.json", "r") as f:
            path = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found. Run record_trajectory.py first.")
        return
        
    print(f"Loaded trajectory with {len(path)} points (approx. {len(path)/100} seconds).")
    
    # 2. Connect via socket to the robot controller
    print(f"Connecting to {HOST}:{PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Connected! Initializing playback...")
    
    # 3. SAFETY FIRST: Move to the starting point slowly!
    # This prevents the robot from jerking if it's far from the start of the trajectory.
    first_point = path[0]
    print(f"Moving slowly to the start position: {first_point}")
    # We use movej for this initial, large movement
    command = f"movej({first_point}, a=0.2, v=0.2)\n"
    sock.send(command.encode())
    
    # Wait for the robot to reach the starting point (give it 5 seconds)
    print("Waiting 5 seconds for the robot to reach the start position...")
    time.sleep(5)
    
    # 4. High-frequency playback loop (100 Hz)
    print("Starting high-frequency trajectory tracking...")
    t0 = time.time()
    for i, joint_config in enumerate(path):
        # We use servoj for continuous and smooth high-frequency motion
        # t=0.01 tells the robot it has 10ms to reach this point
        command = f"servoj({joint_config}, t=0.01, lookahead_time=0.1, gain=300)\n"
        sock.send(command.encode())
        
        # Strict time control to maintain exactly 100 Hz (10ms)
        expected_time = t0 + (i + 1) * 0.01
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    print("Trajectory completed successfully.")
    
    # Close the connection
    sock.close()

if __name__ == "__main__":
    replay()

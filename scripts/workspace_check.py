"""
Workspace verification script for the UR3 lemniscate trajectory.

Checks that every point of the trajectory lies within the reachable workspace
of the arm at its configured home position. Generates results/workspace_check.png.

Usage
-----
    python scripts/workspace_check.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

XML_PATH = "models/ur3.xml"
HOME_Q = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
TRAJ_R = 0.08
N_SAMPLES = 50_000

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

mujoco.mj_resetData(model, data)
data.qpos[:6] = HOME_Q.copy()
mujoco.mj_forward(model, data)
ee_home = data.site_xpos[ee_id].copy()
center = ee_home.copy()

print(f"EE home: X={ee_home[0]:.4f}  Y={ee_home[1]:.4f}  Z={ee_home[2]:.4f}")

# Sample random configurations for workspace cloud
jnt_lo = model.jnt_range[:6, 0]
jnt_hi = model.jnt_range[:6, 1]
ws = []
for _ in range(N_SAMPLES):
    data.qpos[:6] = np.random.uniform(jnt_lo, jnt_hi)
    mujoco.mj_forward(model, data)
    ws.append(data.site_xpos[ee_id].copy())
ws = np.array(ws)

# Trajectory points
t_vals = np.linspace(0, 2 * np.pi, 500)
traj = []
for t in t_vals:
    d = 1.0 + np.sin(t) ** 2
    traj.append([
        center[0] + TRAJ_R * np.cos(t) / d,
        center[1] + TRAJ_R * np.sin(t) * np.cos(t) / d,
        center[2],
    ])
traj = np.array(traj)

max_nn = max(np.linalg.norm(ws - pt, axis=1).min() for pt in traj)
print(f"Max distance from trajectory to nearest workspace sample: {max_nn * 1000:.1f} mm")
if max_nn < 0.03:
    print("  Trajectory is well within the reachable workspace.")
elif max_nn < 0.06:
    print("  Warning: trajectory is near the workspace boundary.")
else:
    print("  Error: trajectory may be outside the workspace.")

# Plot
os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("UR3 workspace vs. lemniscate trajectory", fontsize=13, fontweight="bold")

ax = axes[0]
ax.scatter(ws[:, 0], ws[:, 1], s=1, alpha=0.05, c="#7289da", label="Workspace samples")
ax.plot(traj[:, 0], traj[:, 1], color="#ff4444", linewidth=2, label="Lemniscate")
ax.plot(ee_home[0], ee_home[1], "ko", markersize=7, label="EE home")
ax.set(xlabel="X (m)", ylabel="Y (m)", title="Top view (XY)")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(ws[:, 0], ws[:, 2], s=1, alpha=0.05, c="#7289da", label="Workspace samples")
ax.plot(traj[:, 0], traj[:, 2], color="#ff4444", linewidth=2.5, label="Lemniscate")
ax.axhline(ee_home[2], color="#ff4444", linestyle="--", alpha=0.5,
           label=f"Traj Z = {ee_home[2]:.3f} m")
ax.plot(ee_home[0], ee_home[2], "ko", markersize=7, label="EE home")
ax.set(xlabel="X (m)", ylabel="Z (m)", title="Side view (XZ)")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/workspace_check.png", dpi=150)
print("Plot saved to results/workspace_check.png")

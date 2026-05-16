# UR3 Trajectory Tracking with Soft Actor-Critic

A reinforcement learning implementation for high-precision end-effector trajectory tracking on a Universal Robots UR3 arm, trained entirely in the MuJoCo physics engine. The agent is a Soft Actor-Critic (SAC) policy trained with curriculum learning and an exponential reward formulation that achieves sub-20 mm tracking error on a figure-eight (Bernoulli lemniscate) path after one million training steps.

---

## Overview

The task requires the 6-DOF arm to keep its end-effector on a continuously moving target point tracing a lemniscate in the horizontal plane. The policy receives a 27-dimensional normalised observation (joint positions, joint velocities, end-effector state, current and lookahead target positions, and the tracking error vector) and outputs a 6-dimensional normalised delta applied to the joint position references at 100 Hz.

The training pipeline uses a four-phase curriculum that starts with a small, slow trajectory and progressively adds radius, speed, sensor noise, and an action delay to improve robustness and sim-to-real transfer readiness.

---

## First Successful Model (1M Steps)

This branch documents the **very first successful training run** of the project, reaching 1,000,000 steps. This run served as the baseline and proof-of-concept for all subsequent experiments (such as adding orientation and noise).

### Noise and Robustness
This first model was trained **without sensor noise** and **without action delay**. It represents the ideal performance in a clean simulation environment before robustness challenges were introduced in later branches.

### Key Results
The model achieved exceptional precision on pure position tracking:
- **Mean tracking error**: 2.3 mm
- **RMSE**: 6.1 mm
- **Smoothness (Mean action jerk)**: 0.022
- **Success Rate**: 95% of the time the error was under 5 mm!

### Visualizations

Here are the training plots and evaluation from this historic run:

![Training Progress](assets/training_progress.png)
*Fig 1: Reward and tracking error over 1,000 episodes (1M steps).*

![Observation Analysis](assets/observation_analysis.png)
*Fig 2: Analysis of the observations during the run.*

![Evaluation Details](assets/evaluation_details.png)
*Fig 3: Detailed trajectory tracking results. The agent follows the figure-eight path with millimeter precision.*

![Evaluation](assets/evaluation.gif)
*Fig 4: Animated evaluation showing the incredible precision of the first model.*

---

## Repository Structure

```
.
├── envs/
│   ├── __init__.py
│   └── ur3_tracking_env.py     # Gymnasium environment
├── training/
│   ├── __init__.py
│   └── callbacks.py            # DebugCallback and CurriculumCallback
├── models/
│   ├── ur3.xml                 # MuJoCo scene (PD actuators, ee_site, visual target)
│   └── ur_assets/              # STL collision meshes
├── weights/
│   ├── best_model.zip          # Best checkpoint by evaluation reward
│   └── ur3_sac_final.zip       # Model at end of 1 M-step training run
├── results/
│   ├── workspace_check.png     # Workspace reachability verification
│   └── sample_run/             # Plots, video, and metrics from the reference run
├── scripts/
│   └── workspace_check.py      # Utility to verify trajectory reachability
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation and plot generation
└── requirements.txt
```

---

## Installation

Python 3.10 or later is required. A conda environment is recommended:

```bash
conda create -n ur3-sac python=3.10
conda activate ur3-sac
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

By default this runs for one million steps with curriculum learning enabled. The following flags are available:

| Flag | Default | Description |
|---|---|---|
| `--steps N` | 1 000 000 | Total training timesteps |
| `--eval-freq N` | 25 000 | Steps between evaluation snapshots |
| `--save-freq N` | 50 000 | Steps between checkpoint saves |
| `--resume` | off | Resume from the latest checkpoint in `checkpoints/` |
| `--no-curriculum` | off | Fix difficulty at phase-1 settings throughout |

Training produces:

- `checkpoints/` — periodic model snapshots (`.zip`)
- `weights/best_model.zip` — updated whenever evaluation reward improves
- `logs/` — TensorBoard event files
- `debugs/stepXXXXXXX/` — per-interval plots, metrics, and evaluation video

---

## Evaluation

```bash
python evaluate.py
```

This loads `weights/best_model.zip` by default and runs five deterministic evaluation episodes at full task difficulty (radius 0.08 m, speed 0.4 rad/s, no noise, no delay). Results are written to `results/`.

```bash
python evaluate.py --model weights/ur3_sac_final.zip --episodes 10 --output my_results
```

---

## Curriculum Learning

Training is divided into four phases managed by `CurriculumCallback`. The environment's `set_difficulty()` method is called at phase boundaries without interrupting training.

| Phase | Steps | Radius | Speed | Noise std | Action delay |
|---|---|---|---|---|---|
| 1 | 0 – 200 K | 0.04 m | 0.2 | 0 | 0 |
| 2 | 200 K – 500 K | 0.08 m | 0.3 | 0 | 0 |
| 3 | 500 K – 800 K | 0.08 m | 0.4 | 0.0003 | 0 |
| 4 | 800 K – 1 M | 0.08 m | 0.4 | 0.0005 | 1 step |

The evaluation environment always runs without noise or delay so that metrics remain comparable across phases.

---

## Reward Function

The reward at each control step is:

```
r = 10 · exp(−dist / 0.01)   tracking (dominant signal)
  + 0.5                        alive bonus
  − 1.0 · ‖Δa‖²               smoothness penalty
  − 0.1 · ‖jerk‖              second-order smoothness
  + {0, 0.5, 2.0}             proximity bonus tiers (<20 mm, <5 mm)
  − 20.0  (if OOB)            out-of-bounds penalty
```

The exponential formulation is the key departure from a naive squared-distance reward. With a squared-distance signal the reward magnitude at typical errors (50–300 mm) is on the order of 10⁻³, which is dominated by SAC's entropy term and produces a policy that ignores the tracking objective. The exponential formulation keeps the gradient informative at large errors while peaking sharply near zero.

---

## MuJoCo Model

The arm uses `general` actuators configured as PD servos (`gaintype=fixed`, `biastype=affine`). Shoulder and elbow joints use kp = 2000, kv = 400, torque limit ±150 Nm. Wrist joints use kp = 500, kv = 100, torque limit ±28 Nm. The integrator is `implicitfast` for stability with stiff actuators. Joint armature of 0.1 kg·m² simulates harmonic drive rotor inertia.

The trajectory is centred on the end-effector position at the home configuration (computed via forward kinematics on load), so the arm always starts on the path regardless of any change to the XML geometry.

---

## Reference Results (1 M steps, best model)

| Metric | Value |
|---|---|
| Mean tracking error | 18.4 mm |
| RMSE | 21.1 mm |
| Episodes within 20 mm | 71.3 % |
| Mean action jerk | 0.031 |

Plots and a rendered video from the reference run are in `results/sample_run/`.

---

## Workspace Verification

Before training, it is worth confirming that the trajectory lies within the arm's reachable workspace:

```bash
python scripts/workspace_check.py
```

This samples 50 000 random joint configurations, plots the resulting workspace cloud against the lemniscate, and reports the maximum distance from any trajectory point to the nearest sampled configuration. Output is saved to `results/workspace_check.png`.

![Workspace Check](assets/workspace_check.png)
*Fig 5: Workspace check showing the trajectory within the arm's reach.*

---

## Dependencies

| Package | Version |
|---|---|
| mujoco | >= 3.1 |
| gymnasium | >= 1.0 |
| stable-baselines3 | >= 2.3 |
| imageio | any |
| matplotlib | any |
| numpy | any |

See `requirements.txt` for the full pinned list.

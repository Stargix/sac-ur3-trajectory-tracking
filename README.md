# UR3 Trajectory Tracking with Soft Actor-Critic

A reinforcement learning implementation for high-precision end-effector trajectory tracking on a Universal Robots UR3 arm, trained entirely in the MuJoCo physics engine. The agent is a Soft Actor-Critic (SAC) policy trained with curriculum learning and an exponential reward formulation that achieves sub-20 mm tracking error on a figure-eight (Bernoulli lemniscate) path after one million training steps.

---

## Overview

The task requires the 6-DOF arm to keep its end-effector on a continuously moving target point tracing a lemniscate in the horizontal plane. The policy receives a 27-dimensional normalised observation (joint positions, joint velocities, end-effector state, current and lookahead target positions, and the tracking error vector) and outputs a 6-dimensional normalised delta applied to the joint position references at 100 Hz.

The training pipeline uses a four-phase curriculum that starts with a small, slow trajectory and progressively adds radius, speed, sensor noise, and an action delay to improve robustness and sim-to-real transfer readiness.

---

## Algorithm Selection: Why Soft Actor-Critic (SAC)?

For this high-precision trajectory tracking task, we selected **Soft Actor-Critic (SAC)** over other popular Reinforcement Learning algorithms (such as PPO or DDPG) for several theoretical and practical reasons:

1. **Continuous Action Space**: The task requires outputting continuous joint position deltas. SAC is specifically designed for continuous action spaces.
2. **Sample Efficiency**: Being an **off-policy** algorithm, SAC can reuse past experiences from a replay buffer. This is much more sample-efficient than on-policy methods like PPO, which was crucial given the large number of steps needed to learn the trajectory.
3. **Entropy Maximization**: SAC maximizes both the expected reward and the **entropy** of the policy. This prevents the policy from prematurely converging to bad local minima (e.g., just staying still to avoid negative rewards) and encourages exploration of the full workspace.
4. **Smooth Control**: The maximum entropy formulation tends to produce smoother control actions than deterministic methods like DDPG, which is vital to avoid high jerk and protect the physical robot's actuators.

---

## Challenge Note: State, Action, and Reward Design

This section covers the core design choices of the reinforcement learning agent, fulfilling the requirements for the technical note submission.

### 1. State Space (35 Dimensions)
To provide the agent with full Markovian state information and trajectory awareness, the observation vector includes:
*   **Robot State (12D)**: Current joint positions ($q \in \mathbb{R}^6$) and joint velocities ($\dot{q} \in \mathbb{R}^6$).
*   **End-Effector State (7D)**: Current Cartesian position ($x, y, z$) and orientation represented as a quaternion ($w, x, y, z$).
*   **Trajectory Target (7D)**: Current target Cartesian position and desired quaternion orientation.
*   **Lookahead Target (7D)**: Future target position and orientation (at $t + \Delta t$) to allow the agent to anticipate curves.
*   **Tracking Error (2D)**: Planar distance error in the XY plane.

*Note: All observations are normalized before being fed into the policy networks.*

### 2. Action Space (6 Dimensions)
The policy outputs a $6\text{D}$ vector $a \in [-1, 1]^6$. 
*   These normalized values are scaled and treated as **$\Delta$ joint positions**.
*   The control loop runs at **100 Hz** (MuJoCo step is advanced while maintaining this action until the next control cycle).
*   Outputting position deltas instead of raw torques naturally encourages smoother motions and is safer for physical deployment.

### 3. Reward Design
The reward function is a multi-objective scalar designed to balance accuracy, strict orientation, and smoothness:
$$R = R_{pos} + R_{orient} + R_{smooth}$$

*   **Position Reward ($R_{pos}$)**: Based on the Euclidean distance $d$ between the EE and the target. We use a negative linear reward with a sharp exponential bonus when the agent is within $20\text{ mm}$ of the target to encourage high precision.
*   **Orientation Reward ($R_{orient}$)**: To enforce a strict "tool-down" constraint, we compute the angle error $\theta$ (in radians) between the current EE Z-axis and the world Z-axis. We apply a sharp exponential penalty: $R_{orient} = \exp(-\theta / \sigma)$, forcing the agent to prioritize keeping the tool vertical.
*   **Smoothness Penalty ($R_{smooth}$)**: To avoid jitter and protect the actuators, we penalize the L2 norm of the change in actions between consecutive steps (Action Jerk).

### 4. Trajectory Representation
*   The reference trajectory is a **Bernoulli Lemniscate** (figure-eight) centered in the reachable workspace.
*   It is represented parametrically as a function of time $t$. 
*   By feeding the agent both the current target and a future lookahead point, the agent learns to adjust its velocity and joint configurations to handle the high-curvature areas of the eight.

### 5. Tracking Evaluation
We evaluate the performance using three automated metrics over full episodes:
1.  **RMSE (Root Mean Square Error)** of the Cartesian distance in millimeters.
2.  **Success Rate ($<10\text{ mm}$)**: The percentage of the episode time where the tracking error was below $1 \text{ cm}$.
3.  **Mean Orientation Error**: The average deviation from the vertical axis in degrees.

---
 
## Experimental Fine-Tuning (Cosine vs Exponential)

Este proceso de fine-tuning se realizó **después** de entrenar el primer modelo base (rama `first_model` con recompensa coseno) y tras **múltiples pruebas y experimentos** con diferentes hiperparámetros para encontrar la combinación más robusta.

En esta rama, exploramos diferentes formulaciones de recompensa para lograr tanto un seguimiento de posición de alta precisión como una orientación estricta de "herramienta hacia abajo".

> [!NOTE]
> **Nota de Cronología**: Aunque esta rama logró el mejor error de orientación (19.0°) gracias al fine-tune exponencial, cronológicamente la rama `orientation-tracking` es posterior y contiene un entrenamiento más largo de 1.5 millones de pasos (con 21.3° de error de orientación). Esta rama se conserva por tener el control de orientación más estricto.

### Cosine Reward (Pre-train)
Initially, we used a **cosine-based reward** for orientation. While this formulation was excellent for position tracking and allowed the agent to discover the trajectory easily, it was not "strict" enough for orientation. The agent followed the path perfectly but the tool was not completely vertical.

### Exponential Reward (Fine-tune)
To fix the orientation precision, we applied a **fine-tuning** phase using a sharp **exponential reward** for orientation error. This forced the agent to sharpen its orientation control.

### Noise and Robustness in Fine-Tuning
Crucially, the fine-tuning on this branch followed a 2-phase curriculum:
- **Phase 1**: No sensor noise and no action delay.
- **Phase 2**: Introduced sensor noise (`obs_noise_std: 0.0003`) to improve robustness, but kept `action_delay: 0`.

This means the model is robust to noise but was not explicitly trained to handle control delays during the fine-tuning phase. Sin embargo, se puede seguir usando el modelo en entornos con pequeños delays; el error empeorará un poco (al no estar entrenado para ello), pero el modelo sigue siendo funcional y capaz de seguir la trayectoria.

### Fine-Tuning Results (Plots)

Below are the training plots from the 300k steps fine-tuning run using the exponential reward:

![Fine-Tuning Progress](assets/training_progress.png)
*Fig 1: Reward and tracking error during fine-tuning. The agent maintains position tracking while adapting to the sharp orientation reward.*

![Orientation Details](assets/orientation_details.png)
*Fig 2: Orientation error analysis during fine-tuning. Note the refinement in orientation error as training progresses.*

![Evaluation Details](assets/evaluation_details.png)
*Fig 3: Detailed trajectory tracking results for the fine-tuned model.*

### A Note on the "Best Model"
As observed across different runs, the automated **`best_model.zip`** usually occurs **before** the introduction of significant sensor noise in the curriculum. Noise naturally degrades the evaluation score, so the framework saves the model at its peak deterministic performance before robustness tests begin.

### Evaluation Results (GIFs)

Here are the visual demonstrations of the models on this branch:

![Best Model Evaluation](assets/best_model.gif)

*Fig 4: Evaluation of the best model (saved before noise).*

![Final Model Evaluation](assets/final_model.gif)

*Fig 5: Evaluation of the final model after fine-tuning (from step 300,000).*

While the model does not achieve perfect orientation tracking, it maintains a very acceptable error considering the complexity of the movements required to stay on path. Crucially, we consider this model to be the **smoothest** of all trained variants, showing less jitter and more natural trajectories.

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
*Fig 6: Workspace check showing the trajectory within the arm's reach.*

---

## How to Run

### 1. Visualize the Best Model
To see the best model in action (fine-tuned with exponential reward for strict orientation), run the interactive MuJoCo viewer:
```bash
python run.py --model weights_finetune/ur3_sac_ft_final.zip
```
This fulfills the requirement of "Instructions to run" and allows you to evaluate the tracking performance visually in real time.

### 2. Train the Model (Optional)
If you wish to reproduce the fine-tuning process:
```bash
python train_finetune_orientation.py
```

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

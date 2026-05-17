"""
PhasedCurriculumCallback — unified curriculum and per-phase early stopping.

Manages a sequence of training phases, each with its own environment
difficulty settings, performance thresholds, and patience counter.
When a phase's thresholds are sustained for `patience` consecutive
evaluation intervals, the callback automatically advances to the next
phase. When the final phase's thresholds are met, training stops.

If no threshold is met, training continues until the global step budget
defined in model.learn(total_timesteps=...) is exhausted.

Phase configuration
-------------------
Each phase is a dict with the following keys:

    name          : str   — display label
    max_steps     : int   — hard upper bound for this phase (None = no limit)
    env_kwargs    : dict  — passed to env.set_difficulty()
    thresholds    : dict  — keys are dot-paths into metrics.json
                            (e.g. "eval.mean_dist_mm")
                            values are (target_value, mode) where mode is
                            "below" or "above"
    patience      : int   — consecutive intervals threshold must be met

Example phase list
------------------
    PHASES = [
        {
            "name": "Phase 1 — position bootstrap",
            "max_steps": 250_000,
            "env_kwargs": {"traj_radius": 0.04, "traj_speed": 0.2,
                           "obs_noise_std": 0.0, "action_delay": 0,
                           "orient_weight": 0.0},
            "thresholds": {"eval.mean_dist_mm": (5.0, "below")},
            "patience": 4,
        },
        ...
    ]
"""

import json
import os

from stable_baselines3.common.callbacks import BaseCallback


class PhasedCurriculumCallback(BaseCallback):
    """Per-phase curriculum with integrated early stopping."""

    def __init__(self, train_env, eval_env, phases, debug_dir,
                 check_freq=25_000, verbose=1):
        """
        Parameters
        ----------
        train_env : UR3OrientationEnv
            Training environment (difficulty updated at phase transitions).
        eval_env : UR3OrientationEnv
            Evaluation environment (difficulty updated at phase transitions;
            noise and delay are always forced to 0 so metrics are comparable).
        phases : list[dict]
            Ordered list of phase configurations (see module docstring).
        debug_dir : str
            Directory where DebugCallback writes stepXXXXXXX/metrics.json.
        check_freq : int
            How often (steps) to read metrics and check thresholds. Should
            match DebugCallback.eval_freq so checks happen after each snapshot.
        """
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.phases = phases
        self.debug_dir = debug_dir
        self.check_freq = check_freq

        self._phase_idx = -1          # -1 = not yet started
        self._consecutive = 0         # intervals current phase thresholds met
        self._last_checked_dir = None # avoid re-checking same snapshot

    # ------------------------------------------------------------------
    # Metrics reading
    # ------------------------------------------------------------------

    def _latest_metrics(self):
        """Return the metrics dict from the most recent snapshot, or None."""
        if not os.path.isdir(self.debug_dir):
            return None

        dirs = sorted(
            d for d in os.listdir(self.debug_dir)
            if d.startswith("step") and
            os.path.isfile(os.path.join(self.debug_dir, d, "metrics.json"))
        )
        if not dirs:
            return None

        latest = dirs[-1]
        if latest == self._last_checked_dir:
            return None  # already processed this snapshot

        path = os.path.join(self.debug_dir, latest, "metrics.json")
        try:
            with open(path) as fh:
                data = json.load(fh)
            self._last_checked_dir = latest
            return data
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _get_nested(d, dot_path):
        """Retrieve a value from a nested dict using a dot-separated key."""
        keys = dot_path.split(".")
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _activate_phase(self, idx):
        """Apply environment settings for the given phase index."""
        self._phase_idx = idx
        self._consecutive = 0
        phase = self.phases[idx]

        train_kwargs = dict(phase["env_kwargs"])
        self.train_env.set_difficulty(**train_kwargs)

        # Eval env always runs without noise or delay for comparable metrics.
        eval_kwargs = dict(phase["env_kwargs"])
        eval_kwargs["obs_noise_std"] = 0.0
        eval_kwargs["action_delay"] = 0
        self.eval_env.set_difficulty(**eval_kwargs)

        if self.verbose:
            kw_str = "  ".join(f"{k}={v}" for k, v in phase["env_kwargs"].items())
            print(f"\n[curriculum] {phase['name']}  ({kw_str})")

    def _thresholds_met(self, metrics, phase):
        """Return True if all thresholds in `phase` are satisfied."""
        for dot_path, (target, mode) in phase["thresholds"].items():
            value = self._get_nested(metrics, dot_path)
            if value is None:
                return False  # metric not yet available
            if mode == "below" and value > target:
                return False
            if mode == "above" and value < target:
                return False
        return True

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._activate_phase(0)

    def _on_step(self) -> bool:
        # Hard phase step limit — advance without checking thresholds
        if self._phase_idx >= 0:
            max_steps = self.phases[self._phase_idx].get("max_steps")
            if max_steps and self.num_timesteps >= max_steps:
                return self._try_advance()

        if self.num_timesteps % self.check_freq != 0 or self.num_timesteps == 0:
            return True

        metrics = self._latest_metrics()
        if metrics is None:
            return True

        phase = self.phases[self._phase_idx]
        met = self._thresholds_met(metrics, phase)

        if met:
            self._consecutive += 1
            if self.verbose:
                threshold_str = "  ".join(
                    f"{k}={self._get_nested(metrics, k):.2f}"
                    for k in phase["thresholds"]
                )
                print(
                    f"[curriculum] {phase['name']} thresholds met "
                    f"[{self._consecutive}/{phase['patience']}]  {threshold_str}"
                )
            if self._consecutive >= phase["patience"]:
                return self._try_advance()
        else:
            if self._consecutive > 0 and self.verbose:
                print(
                    f"[curriculum] {phase['name']} streak reset "
                    f"(was {self._consecutive})"
                )
            self._consecutive = 0

        return True

    def _try_advance(self) -> bool:
        """Advance to next phase or stop if this was the last phase."""
        next_idx = self._phase_idx + 1
        if next_idx >= len(self.phases):
            if self.verbose:
                print(
                    f"\n[curriculum] All phases complete at step "
                    f"{self.num_timesteps:,} — stopping training."
                )
            return False  # signals SB3 to stop
        self._activate_phase(next_idx)
        return True

"""
EarlyStoppingCallback — stops training when a performance threshold is
sustained for a configurable number of consecutive evaluation intervals.

The callback reads the most recent metrics.json written by DebugCallback
rather than maintaining its own evaluation loop, so it adds zero overhead
and is consistent with what you see in the debug snapshots.
"""

import json
import os

from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a metric stays above/below a threshold for N intervals.

    Parameters
    ----------
    debug_dir : str
        Directory where DebugCallback writes stepXXXXXXX/metrics.json files.
    metric : str
        Dot-separated path into the metrics JSON.
        Examples: "eval.pct_under_10mm", "eval.mean_dist_mm", "eval.rmse_mm"
    threshold : float
        Target value for the metric.
    mode : str
        "above" — stop when metric >= threshold for `patience` intervals.
        "below" — stop when metric <= threshold for `patience` intervals.
    patience : int
        Number of consecutive intervals the threshold must be met before stopping.
    check_freq : int
        How often (in steps) to check. Should match DebugCallback.eval_freq.
    verbose : int
    """

    def __init__(
        self,
        debug_dir,
        metric="eval.pct_under_10mm",
        threshold=90.0,
        mode="above",
        patience=5,
        check_freq=25_000,
        verbose=1,
    ):
        super().__init__(verbose)
        self.debug_dir = debug_dir
        self.metric_path = metric.split(".")
        self.threshold = threshold
        self.mode = mode
        self.patience = patience
        self.check_freq = check_freq
        self._consecutive = 0

    def _read_latest_metric(self):
        """Return the metric value from the most recent metrics.json, or None."""
        if not os.path.isdir(self.debug_dir):
            return None

        dirs = sorted(
            d for d in os.listdir(self.debug_dir)
            if d.startswith("step") and
            os.path.isfile(os.path.join(self.debug_dir, d, "metrics.json"))
        )
        if not dirs:
            return None

        path = os.path.join(self.debug_dir, dirs[-1], "metrics.json")
        try:
            with open(path) as fh:
                data = json.load(fh)
            for key in self.metric_path:
                data = data[key]
            return float(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0 or self.num_timesteps == 0:
            return True

        value = self._read_latest_metric()
        if value is None:
            return True

        met = (
            value >= self.threshold if self.mode == "above"
            else value <= self.threshold
        )

        if met:
            self._consecutive += 1
            if self.verbose:
                print(
                    f"[early_stop] {'.'.join(self.metric_path)} = {value:.2f}  "
                    f"({'above' if self.mode == 'above' else 'below'} {self.threshold})  "
                    f"[{self._consecutive}/{self.patience}]"
                )
            if self._consecutive >= self.patience:
                print(
                    f"\n[early_stop] Threshold met for {self.patience} consecutive "
                    f"intervals — stopping training at step {self.num_timesteps:,}."
                )
                return False  # signals SB3 to stop
        else:
            if self._consecutive > 0 and self.verbose:
                print(
                    f"[early_stop] {'.'.join(self.metric_path)} = {value:.2f}  "
                    f"streak reset (was {self._consecutive})"
                )
            self._consecutive = 0

        return True

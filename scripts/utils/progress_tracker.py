"""
Progress tracking for benchmark runs.

Simple utility to track progress and estimate remaining time.

Created: 16Jan26
"""

import time
from datetime import datetime
from typing import Optional


class ProgressTracker:
    """Track progress of benchmark experiments with time estimation."""

    def __init__(self, total_experiments: int):
        """
        Initialize progress tracker.

        Parameters
        ----------
        total_experiments : int
            Total number of experiments to run
        """
        self.total = total_experiments
        self.completed = 0
        self.start_time = time.time()
        self.experiment_times = []

    def update(self, experiment_name: str, elapsed_sec: float, success: bool = True):
        """
        Update progress after completing an experiment.

        Parameters
        ----------
        experiment_name : str
            Name of completed experiment (e.g., "erbf_friedman1")
        elapsed_sec : float
            Time taken for this experiment (seconds)
        success : bool
            Whether the experiment succeeded
        """
        self.completed += 1
        self.experiment_times.append(elapsed_sec)

        # Calculate ETA
        total_elapsed = time.time() - self.start_time
        avg_time = total_elapsed / self.completed
        remaining = (self.total - self.completed) * avg_time

        # Format ETA
        if remaining > 3600:
            eta_str = f"{remaining/3600:.1f}h"
        elif remaining > 60:
            eta_str = f"{remaining/60:.1f}min"
        else:
            eta_str = f"{remaining:.0f}s"

        status = "OK" if success else "FAILED"
        print(
            f"[{self.completed}/{self.total}] {experiment_name} "
            f"({elapsed_sec:.1f}s) [{status}] - ETA: {eta_str}",
            flush=True
        )

    def get_summary(self) -> dict:
        """Get summary statistics."""
        total_elapsed = time.time() - self.start_time
        return {
            'completed': self.completed,
            'total': self.total,
            'total_time_sec': total_elapsed,
            'avg_time_sec': total_elapsed / max(self.completed, 1),
            'min_time_sec': min(self.experiment_times) if self.experiment_times else 0,
            'max_time_sec': max(self.experiment_times) if self.experiment_times else 0,
        }

    def print_summary(self):
        """Print final summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Benchmark Complete")
        print("="*60)
        print(f"  Completed: {summary['completed']}/{summary['total']} experiments")
        print(f"  Total time: {summary['total_time_sec']/60:.1f} minutes")
        print(f"  Avg per experiment: {summary['avg_time_sec']:.1f}s")
        print(f"  Range: {summary['min_time_sec']:.1f}s - {summary['max_time_sec']:.1f}s")
        print("="*60)


def format_timestamp() -> str:
    """Return current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_date() -> str:
    """Return current date in YYYYMMDD format."""
    return datetime.now().strftime("%Y%m%d")

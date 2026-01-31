"""MLflow helpers for experiment setup and artifact logging."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.figure
import mlflow


def setup_experiment(name: str) -> str:
    """Configure MLflow tracking URI and set/create experiment.

    Returns the experiment ID.
    """
    project_root = Path(__file__).resolve().parents[2]
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    tracking_uri = mlruns_dir.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(name)
    return experiment.experiment_id


def log_plots(figs: Dict[str, matplotlib.figure.Figure]) -> None:
    """Save matplotlib figures to temp PNGs and log as MLflow artifacts.

    Closes each figure after logging to free memory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for fig_name, fig in figs.items():
            path = os.path.join(tmpdir, f"{fig_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(path)
            matplotlib.pyplot.close(fig)

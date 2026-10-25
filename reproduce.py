"""Reproduce a previous MLflow run by re-training with its exact logged params."""
import argparse
import tempfile

import mlflow
import yaml
from mlflow import MlflowClient

from train import main

TRACKING_URI = "sqlite:///mlflow.db"


def _config_from_params(params: dict) -> dict:
    return {
        "seed": int(params["seed"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "epochs": int(params["epochs"]),
        "checkpoint_dir": "artifacts/checkpoints",
        "model_path": "artifacts/model.pt",
    }


def reproduce(run_id: str | None, best: bool) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    if best:
        runs = mlflow.search_runs(
            experiment_names=["mnist"],
            order_by=["metrics.best_accuracy DESC"],
        )
        run_id = runs.iloc[0]["run_id"]
        print(f"Best run: {run_id}")

    params = client.get_run(run_id).data.params
    cfg = _config_from_params(params)
    print(f"Reproducing run {run_id} with params: {cfg}")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        config_path = f.name

    main(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="ID of the run to reproduce")
    group.add_argument("--best", action="store_true", help="reproduce the best-scoring run")
    args = parser.parse_args()
    reproduce(args.run_id, args.best)

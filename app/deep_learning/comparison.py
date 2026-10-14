
from __future__ import annotations

import logging
from datetime import datetime, timezone

import mlflow

from app.config import MLFLOW_TRACKING_URI
from app.deep_learning.trainer import train_model

logger = logging.getLogger(__name__)

COMPARISON_EXPERIMENT = "ModelAutopsy-Model-Comparison"


def compare_models(
    n_samples: int = 1500,
    n_features: int = 10,
    epochs: int = 15,
    random_state: int = 42,
) -> dict:
    """Train and compare several neural-network configurations."""

    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")
    if n_features < 4:
        raise ValueError("n_features must be at least 4")
    if epochs < 2:
        raise ValueError("epochs must be at least 2")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(COMPARISON_EXPERIMENT)

    configurations = [
        {
            "name": "baseline",
            "hidden_units": 32,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "batch_size": 64,
        },
        {
            "name": "larger_network",
            "hidden_units": 64,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "batch_size": 64,
        },
        {
            "name": "lower_learning_rate",
            "hidden_units": 32,
            "learning_rate": 0.0005,
            "dropout": 0.2,
            "batch_size": 64,
        },
        {
            "name": "higher_dropout",
            "hidden_units": 32,
            "learning_rate": 0.001,
            "dropout": 0.4,
            "batch_size": 64,
        },
    ]

    results = []

    with mlflow.start_run(
        run_name="neural-network-model-comparison"
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        mlflow.log_params(
            {
                "comparison_count": len(configurations),
                "n_samples": n_samples,
                "n_features": n_features,
                "requested_epochs": epochs,
                "random_state": random_state,
            }
        )

        for config in configurations:
            logger.info(
                "Training comparison candidate: %s",
                config["name"],
            )

            try:
                with mlflow.start_run(
                    run_name=config["name"],
                    nested=True,
                ):
                    result = train_model(
                        epochs=epochs,
                        batch_size=config["batch_size"],
                        learning_rate=config["learning_rate"],
                        random_state=random_state,
                        n_samples=n_samples,
                        n_features=n_features,
                        hidden_units=config["hidden_units"],
                        dropout=config["dropout"],
                    )

                    summary = result["training_summary"]
                    history = result["history"]

                    best_epoch = summary["best_epoch"]
                    best_index = best_epoch - 1

                    best_validation_accuracy = history[
                        "val_accuracy"
                    ][best_index]

                    best_train_accuracy = history[
                        "train_accuracy"
                    ][best_index]

                    accuracy_gap = (
                        best_train_accuracy
                        - best_validation_accuracy
                    )

                    candidate = {
                        "name": config["name"],
                        "config": config,
                        "best_validation_loss": float(
                            summary["best_validation_loss"]
                        ),
                        "best_validation_accuracy": float(
                            best_validation_accuracy
                        ),
                        "train_accuracy_at_best_epoch": float(
                            best_train_accuracy
                        ),
                        "accuracy_gap": float(accuracy_gap),
                        "best_epoch": int(best_epoch),
                        "epochs_completed": int(
                            summary["epochs_completed"]
                        ),
                        "early_stopping_triggered": bool(
                            summary["early_stopping_triggered"]
                        ),
                        "training_time_seconds": float(
                            summary["training_time_seconds"]
                        ),
                    }

                    mlflow.log_params(config)
                    mlflow.log_metrics(
                        {
                            "best_validation_loss": (
                                candidate["best_validation_loss"]
                            ),
                            "best_validation_accuracy": (
                                candidate[
                                    "best_validation_accuracy"
                                ]
                            ),
                            "train_accuracy_at_best_epoch": (
                                candidate[
                                    "train_accuracy_at_best_epoch"
                                ]
                            ),
                            "accuracy_gap": candidate["accuracy_gap"],
                            "best_epoch": float(best_epoch),
                            "epochs_completed": float(
                                summary["epochs_completed"]
                            ),
                            "training_time_seconds": float(
                                summary["training_time_seconds"]
                            ),
                            "early_stopping_triggered": float(
                                summary["early_stopping_triggered"]
                            ),
                        }
                    )

                    # Log the checkpoint before the next candidate
                    # overwrites the shared local checkpoint path.
                    mlflow.log_artifact(
                        summary["checkpoint_path"],
                        artifact_path="checkpoints",
                    )

                    results.append(candidate)

            except Exception:
                logger.exception(
                    "Model comparison candidate failed: %s",
                    config["name"],
                )
                raise

        # Lower validation loss wins. Accuracy breaks ties.
        results.sort(
            key=lambda item: (
                item["best_validation_loss"],
                -item["best_validation_accuracy"],
            )
        )

        for rank, candidate in enumerate(results, start=1):
            candidate["rank"] = rank

        best_model = results[0]

        mlflow.log_metrics(
            {
                "winning_validation_loss": (
                    best_model["best_validation_loss"]
                ),
                "winning_validation_accuracy": (
                    best_model["best_validation_accuracy"]
                ),
            }
        )

        report = {
            "experiment": COMPARISON_EXPERIMENT,
            "run_id": parent_run_id,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "ranking_criteria": (
                "Lowest validation loss, then highest validation accuracy"
            ),
            "models_compared": len(results),
            "best_model": best_model,
            "leaderboard": results,
            "limitations": [
                (
                    "All candidates use the same synthetic dataset "
                    "generation settings and random seed."
                ),
                (
                    "Validation performance is not a substitute "
                    "for evaluation on an independent test set."
                ),
                (
                    "The best configuration is only best among "
                    "the candidates tested."
                ),
            ],
        }

        mlflow.log_dict(
            report,
            "model_comparison_report.json",
        )

        logger.info(
            "Model comparison completed. Best candidate: %s",
            best_model["name"],
        )

        return report

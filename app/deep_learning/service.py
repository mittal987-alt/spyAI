
import logging
from datetime import datetime, timezone

import mlflow

from app.config import MLFLOW_TRACKING_URI
from app.deep_learning.trainer import train_model
from app.deep_learning.diagnostics import analyze_training_history
from app.deep_learning.prediction_analysis import (
    analyze_predictions,
    generate_prediction_findings,
)
from app.deep_learning.explainability import explain_predictions


logger = logging.getLogger(__name__)

DEEP_LEARNING_EXPERIMENT = "ModelAutopsy-DeepLearning-Diagnostics"


def run_deep_learning_diagnosis(
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    n_samples: int = 3000,
    n_features: int = 20,
    hidden_units: int = 64,
    dropout: float = 0.2,
    random_state: int = 42,
    explain: bool = True,
) -> dict:
    """Train a neural network and generate diagnostic reports."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(DEEP_LEARNING_EXPERIMENT)

    with mlflow.start_run(
        run_name="neural-network-diagnostics"
    ) as active_run:
        run_id = active_run.info.run_id

        # 1. Train the model with early stopping and checkpointing.
        result = train_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_samples=n_samples,
            n_features=n_features,
            hidden_units=hidden_units,
            dropout=dropout,
            random_state=random_state,
        )

        training_summary = result["training_summary"]

        # 2. Analyze training behavior.
        training_report = analyze_training_history(
            result["history"]
        )

        # 3. Analyze validation predictions.
        prediction_report = analyze_predictions(
            result["model"],
            result["X_val"],
            result["y_val"],
        )

        prediction_findings = generate_prediction_findings(
            prediction_report
        )

        # 4. Explain predictions using SHAP when requested.
        explanation_report = None

        if explain:
            feature_names = [
                f"feature_{i}"
                for i in range(n_features)
            ]

            explanation_report = explain_predictions(
                model=result["model"],
                background=result["X_train"],
                samples=result["X_val"],
                feature_names=feature_names,
                max_background=50,
                max_samples=20,
            )

        # 5. Combine all diagnostic results.
        diagnostics = {
            **training_report,
            "prediction_analysis": prediction_report,
            "prediction_findings": prediction_findings,
            "explainability": explanation_report,
        }

        # 6. Log parameters to MLflow.
        mlflow.log_params(result["config"])

        mlflow.log_params(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "diagnostic_type": (
                    "training_prediction_explainability"
                ),
                "shap_enabled": explain,
            }
        )

        # 7. Log training, prediction, and efficiency metrics.
        metrics = {
            "final_train_loss": training_report[
                "final_train_loss"
            ],
            "final_validation_loss": training_report[
                "final_validation_loss"
            ],
            "final_train_accuracy": training_report[
                "final_train_accuracy"
            ],
            "final_validation_accuracy": training_report[
                "final_validation_accuracy"
            ],
            "accuracy_gap": training_report["accuracy_gap"],
            "best_validation_loss": training_report[
                "best_validation_loss"
            ],
            "best_validation_epoch": float(
                training_report["best_validation_epoch"]
            ),
            "overfitting_detected": float(
                training_report["overfitting_detected"]
            ),
            "training_instability_detected": float(
                training_report[
                    "training_instability_detected"
                ]
            ),
            "prediction_accuracy": prediction_report[
                "accuracy"
            ],
            "prediction_mean_confidence": prediction_report[
                "confidence"
            ]["mean"],
            "prediction_error_rate": prediction_report[
                "errors"
            ]["error_rate"],
            "high_confidence_error_count": float(
                prediction_report["errors"][
                    "high_confidence_error_count"
                ]
            ),
            "prediction_finding_count": float(
                len(prediction_findings)
            ),
            "epochs_completed": float(
                training_summary["epochs_completed"]
            ),
            "best_epoch": float(
                training_summary["best_epoch"]
            ),
            "training_time_seconds": float(
                training_summary["training_time_seconds"]
            ),
            "early_stopping_triggered": float(
                training_summary[
                    "early_stopping_triggered"
                ]
            ),
        }

        mlflow.log_metrics(metrics)

        # 8. Log SHAP feature importance metrics.
        if explanation_report is not None:
            top_features = explanation_report[
                "global_feature_importance"
            ]

            mlflow.log_metric(
                "shap_explained_sample_count",
                float(
                    explanation_report[
                        "explained_sample_count"
                    ]
                ),
            )

            for rank, item in enumerate(
                top_features[:10],
                start=1,
            ):
                mlflow.log_metric(
                    f"shap_importance_rank_{rank}",
                    float(item["mean_absolute_shap"]),
                )

        # 9. Add searchable MLflow tags.
        has_findings = bool(
            training_report["findings"]
            or prediction_findings
        )

        mlflow.set_tag(
            "diagnostic_status",
            (
                "findings_detected"
                if has_findings
                else "no_findings"
            ),
        )

        mlflow.set_tag(
            "explainability_status",
            (
                "completed"
                if explanation_report is not None
                else "disabled"
            ),
        )

        mlflow.set_tag(
            "early_stopping",
            str(
                training_summary[
                    "early_stopping_triggered"
                ]
            ).lower(),
        )

        # 10. Log the best-model checkpoint.
        mlflow.log_artifact(
            training_summary["checkpoint_path"],
            artifact_path="checkpoints",
        )

        # 11. Build the API response.
        output = {
            "experiment": DEEP_LEARNING_EXPERIMENT,
            "run_id": run_id,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "dataset": result["dataset"],
            "config": result["config"],
            "training_summary": training_summary,
            "diagnostics": diagnostics,
            "training_history": result["history"],
        }

        # 12. Store the complete diagnostic report.
        mlflow.log_dict(
            output,
            "deep_learning_diagnosis.json",
        )

        logger.info(
            "Deep learning diagnosis completed. "
            "Run ID: %s; early stopping: %s; SHAP enabled: %s",
            run_id,
            training_summary["early_stopping_triggered"],
            explain,
        )

        return output

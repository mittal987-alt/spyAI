
import json
import logging
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn

from app.config import (
    DRIFT_ALPHA,
    DRIFT_EFFECT_THRESHOLD,
    DRIFT_MEDIUM_THRESHOLD,
    DRIFT_HIGH_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
)
from app.data.dataset import load_datasets
from app.diagnostics.drift import detect_feature_drift
from app.evaluation.metrics import evaluate_model, compare_metrics
from app.models.train import train_and_save_model
from app.monitoring.alerts import (
    generate_drift_alerts,
    save_alerts,
)
from app.validation.data_quality import (
    validate_dataset,
    require_valid_dataset,
)

logger = logging.getLogger(__name__)


def run_experiment(save_report: bool = True) -> dict:
    """
    Execute a complete feature-drift investigation.

    Workflow:
    1. Initialize MLflow tracking.
    2. Load datasets and validate data quality.
    3. Train a baseline model.
    4. Compare baseline and drifted dataset performance.
    5. Detect feature drift with multiple-testing correction.
    6. Generate and persist actionable alerts.
    7. Log metrics, parameters, artifacts, and the report to MLflow.
    8. Optionally save the report locally.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="synthetic-feature-drift"
    ) as active_run:
        run_id = active_run.info.run_id

        logger.info(
            "Starting drift experiment. MLflow run ID: %s",
            run_id,
        )

        # ---------------------------------------------------------
        # 1. Load datasets
        # ---------------------------------------------------------
        data = load_datasets()

        required_keys = {
            "X_train",
            "y_train",
            "X_reference",
            "y_reference",
            "X_drifted",
            "y_drifted",
            "feature_names",
        }

        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Dataset loader is missing keys: {sorted(missing_keys)}"
            )

        feature_names = list(data["feature_names"])

        # ---------------------------------------------------------
        # 2. Validate data quality
        # ---------------------------------------------------------
        validation_reports = {}

        datasets_to_validate = [
            ("training", "X_train"),
            ("reference", "X_reference"),
            ("drifted", "X_drifted"),
        ]

        for dataset_name, dataset_key in datasets_to_validate:
            validation_report = validate_dataset(
                data[dataset_key],
                feature_names,
                expected_features=len(feature_names),
            )

            validation_reports[dataset_name] = validation_report

            mlflow.log_metric(
                f"{dataset_name}_data_valid",
                float(validation_report["valid"]),
            )
            mlflow.log_metric(
                f"{dataset_name}_duplicate_rate",
                float(
                    validation_report["summary"]["duplicate_rate"]
                ),
            )

            mlflow.log_dict(
                validation_report,
                f"validation/{dataset_name}.json",
            )

            # Fail the experiment if the dataset violates hard rules.
            require_valid_dataset(validation_report)

        logger.info("All datasets passed data-quality validation.")

        # ---------------------------------------------------------
        # 3. Train the baseline model
        # ---------------------------------------------------------
        model = train_and_save_model(
            data["X_train"],
            data["y_train"],
        )

        # ---------------------------------------------------------
        # 4. Evaluate baseline and drifted data
        # ---------------------------------------------------------
        baseline_metrics = evaluate_model(
            model,
            data["X_reference"],
            data["y_reference"],
        )

        changed_metrics = evaluate_model(
            model,
            data["X_drifted"],
            data["y_drifted"],
        )

        metric_deltas = compare_metrics(
            baseline_metrics,
            changed_metrics,
        )

        # ---------------------------------------------------------
        # 5. Detect statistical feature drift
        # ---------------------------------------------------------
        drift_results = detect_feature_drift(
            data["X_reference"],
            data["X_drifted"],
            feature_names,
            alpha=DRIFT_ALPHA,
            effect_threshold=DRIFT_EFFECT_THRESHOLD,
            medium_threshold=DRIFT_MEDIUM_THRESHOLD,
            high_threshold=DRIFT_HIGH_THRESHOLD,
        )

        drifted_features = [
            result["feature"]
            for result in drift_results
            if result["drift_detected"]
        ]

        severity_counts = {
            severity: sum(
                result["drift_detected"]
                and result["severity"] == severity
                for result in drift_results
            )
            for severity in ("high", "medium", "low")
        }

        # ---------------------------------------------------------
        # 6. Generate and persist alerts
        # ---------------------------------------------------------
        alerts = generate_drift_alerts(
            drift_results,
            run_id=run_id,
        )

        saved_alert_count = save_alerts(alerts)

        logger.info(
            "Alert processing completed: %d generated, %d saved.",
            len(alerts),
            saved_alert_count,
        )

        # ---------------------------------------------------------
        # 7. Log parameters to MLflow
        # ---------------------------------------------------------
        mlflow.log_params(
            {
                "random_state": 42,
                "training_samples": len(data["y_train"]),
                "reference_samples": len(data["y_reference"]),
                "drifted_samples": len(data["y_drifted"]),
                "feature_count": len(feature_names),
                "drift_alpha": DRIFT_ALPHA,
                "drift_effect_threshold": DRIFT_EFFECT_THRESHOLD,
                "drift_medium_threshold": DRIFT_MEDIUM_THRESHOLD,
                "drift_high_threshold": DRIFT_HIGH_THRESHOLD,
                "model_type": "LogisticRegression",
                "preprocessing": "StandardScaler",
                "validation_enabled": True,
                "multiple_testing_correction": "Benjamini-Hochberg",
            }
        )

        # ---------------------------------------------------------
        # 8. Log evaluation metrics to MLflow
        # ---------------------------------------------------------
        metrics_to_log = {}

        for name, value in baseline_metrics.items():
            if name != "sample_count":
                metrics_to_log[f"baseline_{name}"] = float(value)

        for name, value in changed_metrics.items():
            if name != "sample_count":
                metrics_to_log[f"changed_{name}"] = float(value)

        for name, value in metric_deltas.items():
            metrics_to_log[f"delta_{name}"] = float(value)

        metrics_to_log["drifted_feature_count"] = float(
            len(drifted_features)
        )
        metrics_to_log["generated_alert_count"] = float(len(alerts))
        metrics_to_log["saved_alert_count"] = float(saved_alert_count)

        for severity, count in severity_counts.items():
            metrics_to_log[f"{severity}_drift_count"] = float(count)

        for result in drift_results:
            # Sanitize feature names for MLflow metric keys.
            feature_name = (
                str(result["feature"])
                .replace("/", "_")
                .replace(" ", "_")
            )

            mlflow.log_metric(
                f"drift_ks_{feature_name}",
                float(result["ks_statistic"]),
            )
            mlflow.log_metric(
                f"drift_p_value_{feature_name}",
                float(result["p_value"]),
            )
            mlflow.log_metric(
                f"drift_adjusted_p_{feature_name}",
                float(result["adjusted_p_value"]),
            )
            mlflow.log_metric(
                f"drift_effect_{feature_name}",
                float(result["effect_size"]),
            )

            mlflow.set_tag(
                f"drift_severity_{feature_name}",
                result["severity"],
            )

        mlflow.log_metrics(metrics_to_log)

        mlflow.set_tag(
            "alert_status",
            "alerts_generated" if alerts else "no_drift_alerts",
        )
        mlflow.set_tag(
            "experiment_status",
            "completed",
        )

        # ---------------------------------------------------------
        # 9. Log the trained model
        # ---------------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            name="baseline_model",
            input_example=data["X_reference"][:5],
        )

        # ---------------------------------------------------------
        # 10. Build the experiment report
        # ---------------------------------------------------------
        report = {
            "experiment": "synthetic_feature_drift",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "training_samples": int(len(data["y_train"])),
                "reference_samples": int(len(data["y_reference"])),
                "drifted_samples": int(len(data["y_drifted"])),
                "feature_count": len(feature_names),
            },
            "data_validation": validation_reports,
            "baseline_metrics": baseline_metrics,
            "changed_metrics": changed_metrics,
            "metric_deltas": metric_deltas,
            "drift_summary": {
                "alpha": DRIFT_ALPHA,
                "effect_threshold": DRIFT_EFFECT_THRESHOLD,
                "medium_threshold": DRIFT_MEDIUM_THRESHOLD,
                "high_threshold": DRIFT_HIGH_THRESHOLD,
                "multiple_testing_correction": "Benjamini-Hochberg",
                "drifted_feature_count": len(drifted_features),
                "drifted_features": drifted_features,
                "severity_counts": severity_counts,
            },
            "feature_drift": drift_results,
            "alert_summary": {
                "generated": len(alerts),
                "saved": saved_alert_count,
                "by_severity": {
                    severity: sum(
                        alert["severity"] == severity
                        for alert in alerts
                    )
                    for severity in ("high", "medium", "low")
                },
            },
            "alerts": alerts,
            "limitations": [
                "The dataset is synthetic, not a real production incident.",
                "Feature drift does not establish causation.",
                "Drifted labels are retained from reference samples.",
                (
                    "Benjamini-Hochberg controls the false discovery rate "
                    "under its applicable statistical assumptions; it does "
                    "not eliminate all false positives."
                ),
                (
                    "Duplicate rows generate warnings rather than "
                    "automatic rejection when below validation limits."
                ),
                (
                    "Severity thresholds are configurable heuristics, "
                    "not calibrated business-impact estimates."
                ),
                (
                    "A drift alert alone does not prove that model "
                    "performance has degraded."
                ),
            ],
        }

        # ---------------------------------------------------------
        # 11. Save report and MLflow artifacts
        # ---------------------------------------------------------
        mlflow.log_dict(
            report,
            "experiment_report.json",
        )
        mlflow.log_dict(
            alerts,
            "alerts/drift_alerts.json",
        )

        if save_report:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)

            report_path = REPORTS_DIR / "latest_experiment.json"
            report_path.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )

            mlflow.log_artifact(
                str(report_path),
                artifact_path="reports",
            )

            report["report_path"] = str(report_path)

        logger.info(
            "Experiment completed successfully. Run ID: %s",
            run_id,
        )

        return report

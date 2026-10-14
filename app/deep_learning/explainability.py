
from __future__ import annotations

import numpy as np
import shap
import torch


def explain_predictions(
    model: torch.nn.Module,
    background: torch.Tensor,
    samples: torch.Tensor,
    feature_names: list[str] | None = None,
    max_background: int = 50,
    max_samples: int = 20,
) -> dict:
    """
    Explain neural-network predictions using SHAP GradientExplainer.

    Returns global feature importance and local feature attributions.
    Explanations are relative to the selected background samples.
    """
    if background.ndim != 2 or samples.ndim != 2:
        raise ValueError("background and samples must be 2D tensors")

    if len(background) == 0 or len(samples) == 0:
        raise ValueError("background and samples must not be empty")

    if background.shape[1] != samples.shape[1]:
        raise ValueError("background and samples need matching features")

    if not torch.isfinite(background).all():
        raise ValueError("background contains non-finite values")

    if not torch.isfinite(samples).all():
        raise ValueError("samples contains non-finite values")

    if max_background < 1 or max_samples < 1:
        raise ValueError("max_background and max_samples must be positive")

    feature_count = samples.shape[1]

    if feature_names is None:
        feature_names = [
            f"feature_{i}" for i in range(feature_count)
        ]

    if len(feature_names) != feature_count:
        raise ValueError("feature_names count does not match input features")

    # Use a bounded subset to keep explanations manageable.
    background = background[:max_background].detach().cpu()
    explained_samples = samples[:max_samples].detach().cpu()

    model.eval()

    try:
        explainer = shap.GradientExplainer(model, background)
        raw_values = explainer.shap_values(explained_samples)
    except Exception as exc:
        raise RuntimeError(
            f"SHAP could not explain this model: {exc}"
        ) from exc

    # SHAP versions may return either a list per output class or
    # a single array with an output-class dimension.
    if isinstance(raw_values, list):
        class_values = [
            np.asarray(values, dtype=float)
            for values in raw_values
        ]
    else:
        values = np.asarray(raw_values, dtype=float)

        if values.ndim == 2:
            # Single-output model: samples x features.
            class_values = [values]
        elif values.ndim == 3:
            # Newer SHAP: samples x features x outputs.
            class_values = [
                values[:, :, class_index]
                for class_index in range(values.shape[2])
            ]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    if not class_values:
        raise ValueError("SHAP returned no feature attributions")

    class_importance = []
    all_importance = []

    for class_index, values in enumerate(class_values):
        if values.shape != (
            len(explained_samples),
            feature_count,
        ):
            raise ValueError(
                f"Unexpected attribution shape for output {class_index}: "
                f"{values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError("SHAP returned non-finite attributions")

        importance = np.mean(np.abs(values), axis=0)
        all_importance.append(importance)

        class_importance.append({
            "class_index": class_index,
            "features": [
                {
                    "feature": feature_names[index],
                    "mean_absolute_shap": float(importance[index]),
                }
                for index in np.argsort(importance)[::-1]
            ],
        })

    global_importance = np.mean(
        np.stack(all_importance, axis=0),
        axis=0,
    )

    ranked_indices = np.argsort(global_importance)[::-1]

    local_explanations = []
    for sample_index in range(len(explained_samples)):
        sample_classes = []

        for class_index, values in enumerate(class_values):
            contributions = values[sample_index]
            top_indices = np.argsort(
                np.abs(contributions)
            )[::-1][:min(10, feature_count)]

            sample_classes.append({
                "class_index": class_index,
                "top_features": [
                    {
                        "feature": feature_names[index],
                        "shap_value": float(contributions[index]),
                        "direction": (
                            "positive"
                            if contributions[index] > 0
                            else "negative"
                            if contributions[index] < 0
                            else "neutral"
                        ),
                    }
                    for index in top_indices
                ],
            })

        local_explanations.append({
            "sample_index": sample_index,
            "class_explanations": sample_classes,
        })

    return {
        "method": "SHAP GradientExplainer",
        "background_sample_count": len(background),
        "explained_sample_count": len(explained_samples),
        "feature_count": feature_count,
        "global_feature_importance": [
            {
                "feature": feature_names[index],
                "mean_absolute_shap": float(
                    global_importance[index]
                ),
            }
            for index in ranked_indices
        ],
        "class_feature_importance": class_importance,
        "local_explanations": local_explanations,
        "limitations": [
            (
                "SHAP values explain model outputs relative to the "
                "selected background samples."
            ),
            (
                "Attributions indicate model behavior, not causation."
            ),
            (
                "Attributions for different output classes should be "
                "interpreted separately."
            ),
        ],
    }

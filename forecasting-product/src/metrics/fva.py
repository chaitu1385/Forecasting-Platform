"""
Forecast Value Add (FVA) computation engine.

Measures how much accuracy each forecast layer contributes:
  L1: Naive baseline (SeasonalNaive)
  L2: Statistical (best of AutoARIMA, AutoETS, or Croston/TSB for sparse)
  L3: ML (best of LightGBM, XGBoost)
  L4: Post-override (planner-adjusted forecast, if applicable)

FVA = error reduction from the previous layer. Positive = improvement.
"""

from typing import Dict, List

import polars as pl

from .definitions import mae, normalized_bias, wmape

# FVA classification thresholds (in WMAPE percentage points)
FVA_ADDS_VALUE_THRESHOLD = 0.02      # > 2pp improvement
FVA_DESTROYS_VALUE_THRESHOLD = -0.02  # > 2pp degradation


def classify_fva(fva_value: float) -> str:
    """Classify an FVA value as ADDS_VALUE, NEUTRAL, or DESTROYS_VALUE."""
    if fva_value > FVA_ADDS_VALUE_THRESHOLD:
        return "ADDS_VALUE"
    elif fva_value < FVA_DESTROYS_VALUE_THRESHOLD:
        return "DESTROYS_VALUE"
    return "NEUTRAL"


def compute_layer_metrics(
    actual: pl.Series,
    forecast: pl.Series,
) -> Dict[str, float]:
    """Compute WMAPE, bias, and MAE for a single layer."""
    return {
        "wmape": wmape(actual, forecast),
        "bias": normalized_bias(actual, forecast),
        "mae": mae(actual, forecast),
    }


def compute_fva_between_layers(
    actual: pl.Series,
    parent_forecast: pl.Series,
    child_forecast: pl.Series,
) -> Dict[str, float]:
    """
    Compute FVA from parent layer to child layer.

    FVA = parent_error - child_error (positive = child is better).
    """
    parent_metrics = compute_layer_metrics(actual, parent_forecast)
    child_metrics = compute_layer_metrics(actual, child_forecast)

    # WMAPE can be None when actuals are all zero on the window.
    p_w = parent_metrics["wmape"]
    c_w = child_metrics["wmape"]
    fva_val = (p_w - c_w) if (p_w is not None and c_w is not None) else None

    return {
        "parent_wmape": p_w,
        "child_wmape": c_w,
        "fva_wmape": fva_val,
        "parent_bias": parent_metrics["bias"],
        "child_bias": child_metrics["bias"],
        "fva_bias": (
            abs(parent_metrics["bias"]) - abs(child_metrics["bias"])
            if parent_metrics["bias"] is not None and child_metrics["bias"] is not None else None
        ),
        "fva_class": classify_fva(fva_val) if fva_val is not None else "undefined",
    }


def compute_fva_cascade(
    actual: pl.Series,
    forecasts: Dict[str, pl.Series],
) -> List[Dict[str, float]]:
    """
    Compute FVA across a full layer cascade.

    Parameters
    ----------
    actual:
        Actual values.
    forecasts:
        Dict mapping layer name to forecast Series.
        Expected order: {"naive": ..., "statistical": ..., "ml": ...}
        Optional: {"override": ...}

    Returns
    -------
    List of dicts, one per layer transition.
    """
    layer_order = ["naive", "statistical", "ml", "override"]
    available = [lyr for lyr in layer_order if lyr in forecasts]

    if not available:
        return []

    results = []

    # First layer: absolute metrics (no parent)
    first = available[0]
    first_metrics = compute_layer_metrics(actual, forecasts[first])
    results.append({
        "layer": first,
        "parent_layer": None,
        "wmape": first_metrics["wmape"],
        "bias": first_metrics["bias"],
        "mae": first_metrics["mae"],
        "fva_wmape": 0.0,
        "fva_bias": 0.0,
        "fva_class": "BASELINE",
    })

    # Subsequent layers: FVA vs previous
    for i in range(1, len(available)):
        parent = available[i - 1]
        child = available[i]
        fva = compute_fva_between_layers(
            actual, forecasts[parent], forecasts[child]
        )
        child_metrics = compute_layer_metrics(actual, forecasts[child])

        results.append({
            "layer": child,
            "parent_layer": parent,
            "wmape": child_metrics["wmape"],
            "bias": child_metrics["bias"],
            "mae": child_metrics["mae"],
            "fva_wmape": fva["fva_wmape"],
            "fva_bias": fva["fva_bias"],
            "fva_class": fva["fva_class"],
        })

    return results


def compute_total_fva(
    actual: pl.Series,
    forecasts: Dict[str, pl.Series],
) -> float:
    """
    Compute total FVA: WMAPE reduction from baseline (naive) to final layer.
    """
    layer_order = ["naive", "statistical", "ml", "override"]
    available = [lyr for lyr in layer_order if lyr in forecasts]

    if len(available) < 2:
        return 0.0

    baseline_wmape = wmape(actual, forecasts[available[0]])
    final_wmape = wmape(actual, forecasts[available[-1]])

    # WMAPE is undefined when actuals are all zero. Treat as "no value added".
    if baseline_wmape is None or final_wmape is None:
        return 0.0

    return baseline_wmape - final_wmape

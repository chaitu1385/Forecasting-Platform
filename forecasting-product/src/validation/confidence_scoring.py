"""
Confidence Scoring — aggregate 4-layer validation into A-F letter grades.

Takes outputs from:
    Layer 1: structural_validator.run_structural_checks()
    Layer 2: logical_validator.run_logical_checks()
    Layer 3: business_rules.validate_business_rules()
    Layer 4: simpsons_paradox (check or report)

Produces a 0-100 numeric score and a letter grade (A-F) with
explanations and recommendations.

Usage:
    from src.validation.confidence_scoring import (
        score_confidence, format_confidence_badge,
    )

    badge = score_confidence(
        structural=structural_result,
        logical=logical_result,
        business=business_result,
        paradox=paradox_result,
    )
    print(format_confidence_badge(badge))
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (50, "D"),
    (0, "F"),
]

_GRADE_DESCRIPTIONS = {
    "A": "High confidence — findings are well-supported by clean, consistent data.",
    "B": "Good confidence — minor issues noted but unlikely to affect conclusions.",
    "C": "Moderate confidence — some data quality or consistency issues. Validate key findings manually.",
    "D": "Low confidence — significant issues detected. Treat findings as directional only.",
    "F": "Very low confidence — critical data issues. Do not rely on these findings for decisions.",
}

_GRADE_RECOMMENDATIONS = {
    "A": [],
    "B": ["Review warnings in the validation report before sharing."],
    "C": [
        "Investigate warning-level issues before sharing.",
        "Cross-reference key numbers with a second data source.",
    ],
    "D": [
        "Resolve blocker-level issues before continuing.",
        "Consider a smaller, cleaner data slice for analysis.",
        "Have a data engineer review the source data.",
    ],
    "F": [
        "HALT — do not present these results.",
        "Fix critical data issues first.",
        "Re-run validation after fixes.",
    ],
}

# ---------------------------------------------------------------------------
# Factor weights (sum = 100)
# ---------------------------------------------------------------------------

_FACTOR_WEIGHTS = {
    "structural": 30,
    "logical": 25,
    "business": 25,
    "paradox": 20,
}

# ---------------------------------------------------------------------------
# Per-factor scoring
# ---------------------------------------------------------------------------

def _score_layer(result: Optional[dict], layer_name: str) -> dict:
    """Score a single validation layer result.

    Returns dict with ``score`` (0-100), ``factor_name``, ``issues``.
    """
    if result is None:
        return {"score": 50, "factor_name": layer_name,
                "issues": ["Layer not run — defaulting to neutral score"],
                "skipped": True}

    # Count severities across all sub-checks
    blocker_count = 0
    warning_count = 0
    pass_count = 0

    details = result.get("details", {})
    if details:
        for check_name, check_result in details.items():
            sev = check_result.get("severity", check_result.get("overall_severity", "PASS"))
            if sev == "BLOCKER":
                blocker_count += 1
            elif sev == "WARNING":
                warning_count += 1
            else:
                pass_count += 1
    else:
        # Top-level ok/severity
        if result.get("ok", True):
            pass_count = 1
        elif result.get("severity") == "BLOCKER" or result.get("paradox_detected"):
            blocker_count = 1
        else:
            warning_count = 1

    total_checks = blocker_count + warning_count + pass_count
    if total_checks == 0:
        return {"score": 100, "factor_name": layer_name, "issues": [],
                "skipped": True}

    # Score: start at 100, deduct for issues
    score = 100
    score -= blocker_count * 30
    score -= warning_count * 10
    score = max(0, min(100, score))

    issues = []
    if blocker_count:
        issues.append(f"{blocker_count} BLOCKER(s) in {layer_name}")
    if warning_count:
        issues.append(f"{warning_count} WARNING(s) in {layer_name}")

    return {
        "score": score,
        "factor_name": layer_name,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "pass_count": pass_count,
        "issues": issues,
        "skipped": False,
    }


def _score_paradox(result: Optional[dict]) -> dict:
    """Score Simpson's Paradox results specifically."""
    if result is None:
        return {"score": 50, "factor_name": "paradox",
                "issues": ["Paradox check not run"], "skipped": True}

    # From check_simpsons_paradox or generate_paradox_report
    paradox = result.get("paradox_detected",
                         result.get("any_paradox",
                                    result.get("summary", {}).get("paradox_found", False)))

    if paradox:
        affected = result.get("reversal_segments",
                              result.get("summary", {}).get("affected_dimensions", []))
        return {
            "score": 20,
            "factor_name": "paradox",
            "issues": [f"Simpson's Paradox detected on {len(affected)} dimension(s)"],
            "skipped": False,
        }

    if result.get("evaluated") is False:
        return {
            "score": 50, "factor_name": "paradox",
            "issues": [f"Paradox comparison unavailable ({result.get('status', 'not_evaluable')}); neutral contribution"],
            "skipped": True, "evaluated": False, "unavailable": True,
        }

    return {"score": 100, "factor_name": "paradox", "issues": [],
            "skipped": False}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_confidence(
    structural: Optional[dict] = None,
    logical: Optional[dict] = None,
    business: Optional[dict] = None,
    paradox: Optional[dict] = None,
) -> dict:
    """Compute a weighted confidence score from 4-layer validation results.

    Each layer contributes a weighted sub-score.  The final score is
    capped: any BLOCKER caps the grade at D; paradox detection caps at C.

    Args:
        structural: Result from ``run_structural_checks()``.
        logical: Result from ``run_logical_checks()``.
        business: Result from ``validate_business_rules()``.
        paradox: Result from ``check_simpsons_paradox()`` or
                ``generate_paradox_report()``.

    Returns:
        dict with ``score`` (0-100), ``grade`` (A-F), ``description``,
        ``recommendations``, ``factors`` (per-layer detail).
    """
    factors = {
        "structural": _score_layer(structural, "structural"),
        "logical": _score_layer(logical, "logical"),
        "business": _score_layer(business, "business"),
        "paradox": _score_paradox(paradox),
    }

    # Weighted score
    total_score = 0.0
    total_weight = 0.0
    for name, weight in _FACTOR_WEIGHTS.items():
        factor = factors[name]
        total_score += factor["score"] * weight
        total_weight += weight

    raw_score = total_score / total_weight if total_weight > 0 else 0

    # Apply caps
    has_blocker = any(
        f.get("blocker_count", 0) > 0
        for f in factors.values()
        if not f.get("skipped")
    )
    has_paradox = factors["paradox"]["score"] < 50

    if has_blocker:
        raw_score = min(raw_score, 69)   # cap at D (max grade D)
    if has_paradox:
        raw_score = min(raw_score, 79)   # cap at C (max grade C)

    score = round(raw_score)
    grade = _score_to_grade(score)

    # Aggregate recommendations
    recommendations = list(_GRADE_RECOMMENDATIONS.get(grade, []))
    for f in factors.values():
        recommendations.extend(f.get("issues", []))

    return {
        "score": score,
        "grade": grade,
        "description": _GRADE_DESCRIPTIONS[grade],
        "recommendations": recommendations,
        "factors": factors,
        "caps_applied": {
            "blocker_cap": has_blocker,
            "paradox_cap": has_paradox,
        },
    }


def format_confidence_badge(result: dict) -> str:
    """Format confidence score as a human-readable badge string.

    Args:
        result: Output from ``score_confidence()``.

    Returns:
        Formatted string like ``"[B 82/100] Good confidence — ..."``
    """
    grade = result["grade"]
    score = result["score"]
    desc = result["description"]
    return f"[{grade} {score}/100] {desc}"


def merge_confidence_scores(*results: dict) -> dict:
    """Merge multiple confidence scores (e.g. from different datasets).

    Takes the minimum score and worst grade.

    Args:
        *results: Multiple ``score_confidence()`` outputs.

    Returns:
        dict with merged ``score``, ``grade``, ``individual_scores``.
    """
    if not results:
        return {"score": 0, "grade": "F", "individual_scores": []}

    scores = [r["score"] for r in results]
    min_score = min(scores)
    grade = _score_to_grade(min_score)

    return {
        "score": min_score,
        "grade": grade,
        "description": _GRADE_DESCRIPTIONS[grade],
        "individual_scores": [
            {"score": r["score"], "grade": r["grade"]} for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _score_to_grade(score: int) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"

"""Forecast Commentary — generate executive summaries for S&OP meetings.

Uses Claude to synthesize accuracy metrics, drift alerts, FVA results,
and comparison data into a VP-level narrative with key metrics,
exceptions, and action items.

Gracefully degrades to a template-based summary when Claude is unavailable.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import polars as pl

from .base import AIFeatureBase

logger = logging.getLogger(__name__)


@dataclass
class KeyMetric:
    """A key performance metric with trend direction."""
    name: str
    value: float
    unit: str = ""
    trend: str = "stable"   # "improving" | "stable" | "degrading"


@dataclass
class CommentaryResult:
    """Result of forecast commentary generation."""
    executive_summary: str = ""
    key_metrics: List[KeyMetric] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)


class CommentaryEngine(AIFeatureBase):
    """Claude-powered forecast commentary generator.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.
    model : str
        Claude model to use.
    """

    def generate(
        self,
        lob: str,
        metrics_df: pl.DataFrame,
        drift_alerts: Optional[List[Any]] = None,
        leaderboard: Optional[pl.DataFrame] = None,
        fva_summary: Optional[pl.DataFrame] = None,
        comparison_summary: Optional[pl.DataFrame] = None,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
    ) -> CommentaryResult:
        """Generate forecast commentary for an S&OP meeting.

        Parameters
        ----------
        lob : str
            Line of business.
        metrics_df : pl.DataFrame
            Metric store data with columns: series_id, target_week,
            actual, forecast, wmape, normalized_bias.
        drift_alerts : list, optional
            DriftAlert objects from ForecastDriftDetector.
        leaderboard : pl.DataFrame, optional
            Model leaderboard from MetricStore.
        fva_summary : pl.DataFrame, optional
            FVA layer summary from FVAAnalyzer.
        comparison_summary : pl.DataFrame, optional
            Forecast comparison summary.
        period_start : date, optional
            Start of reporting period.
        period_end : date, optional
            End of reporting period.

        Returns
        -------
        CommentaryResult
        """
        if metrics_df.is_empty():
            return CommentaryResult(executive_summary="No metric data available for commentary.")

        # Compute summary stats for both Claude and fallback
        stats = self._compute_stats(metrics_df)

        if not self.available:
            return self._fallback_result(lob, stats, drift_alerts)

        try:
            prompt = self._build_prompt(
                lob, stats, drift_alerts, leaderboard,
                fva_summary, comparison_summary, period_start, period_end,
            )
            text = self._call_claude(self._system_prompt(), prompt)
            return self._parse_response(text, stats)
        except Exception as e:
            logger.error("CommentaryEngine: API call failed: %s", e)
            return self._fallback_result(lob, stats, drift_alerts)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an S&OP meeting preparation assistant for a retail demand "
            "planning team. Generate concise, VP-level executive commentary about "
            "forecast performance. Focus on business impact, not technical details. "
            "Be specific with numbers and trends."
        )

    @staticmethod
    def _compute_stats(metrics_df: pl.DataFrame) -> Dict[str, Any]:
        """Compute aggregate statistics from metrics data."""
        from ..metrics.aggregation import observation_metrics

        stats: Dict[str, Any] = {}

        if not metrics_df.is_empty() and {"actual", "forecast"}.issubset(metrics_df.columns):
            pooled = observation_metrics(
                metrics_df, ["wmape", "normalized_bias"], preserve_context=False,
            ).row(0, named=True)
            for name, output in (("wmape", "overall_wmape"), ("normalized_bias", "overall_bias")):
                if pooled[name] is not None:
                    stats[output] = round(pooled[name], 4)
            stats["metric_population"] = pooled["metric_population"]
            stats["scoring_policy"] = pooled["scoring_policy"]
            for name in ("n_runs", "n_models"):
                if name in pooled:
                    stats[name] = pooled[name]
            units = [column for column in ("series_id", "fold") if column in metrics_df.columns]
            unit_scores = observation_metrics(metrics_df, ["wmape"], units)["wmape"]
            stats["n_undefined_wmape_units"] = unit_scores.null_count()
            median = unit_scores.median()
            if median is not None:
                stats["wmape_median"] = round(median, 4)
        elif not metrics_df.is_empty():
            stats["metric_population"] = "summary-only: raw observations required for WMAPE/bias"

        if "series_id" in metrics_df.columns:
            stats["n_series"] = metrics_df["series_id"].n_unique()

        if "target_week" in metrics_df.columns:
            stats["n_periods"] = metrics_df["target_week"].n_unique()

        return stats

    def _build_prompt(
        self,
        lob: str,
        stats: Dict[str, Any],
        drift_alerts: Optional[List[Any]],
        leaderboard: Optional[pl.DataFrame],
        fva_summary: Optional[pl.DataFrame],
        comparison_summary: Optional[pl.DataFrame],
        period_start: Optional[date],
        period_end: Optional[date],
    ) -> str:
        parts = [f"Generate executive commentary for LOB '{lob}'."]

        if period_start and period_end:
            parts.append(f"Reporting period: {period_start} to {period_end}.")

        parts.append(f"\n## Accuracy Metrics\n```json\n{json.dumps(stats, indent=2)}\n```")

        if drift_alerts:
            critical = sum(1 for a in drift_alerts if a.severity.value == "critical")
            warning = sum(1 for a in drift_alerts if a.severity.value == "warning")
            top_alerts = [
                {"series_id": a.series_id, "metric": a.metric, "severity": a.severity.value, "message": a.message}
                for a in drift_alerts[:5]
            ]
            parts.append(
                f"\n## Drift Alerts\n{critical} critical, {warning} warning alerts.\n"
                f"Top alerts:\n```json\n{json.dumps(top_alerts, indent=2)}\n```"
            )

        if leaderboard is not None and not leaderboard.is_empty():
            rows = leaderboard.head(5).to_dicts()
            parts.append(f"\n## Model Leaderboard (top 5)\n```json\n{json.dumps(rows, indent=2, default=str)}\n```")

        if fva_summary is not None and not fva_summary.is_empty():
            rows = fva_summary.to_dicts()
            parts.append(f"\n## FVA Summary\n```json\n{json.dumps(rows, indent=2, default=str)}\n```")

        parts.append("""
## Instructions
Respond with these exact section headers:

### EXECUTIVE_SUMMARY
A 3-5 sentence paragraph suitable for a VP/C-level presentation.

### KEY_METRICS
A JSON array where each element has: "name", "value", "unit", "trend" (one of "improving", "stable", "degrading").

### EXCEPTIONS
Bulleted list of series or areas needing attention.

### ACTION_ITEMS
Bulleted list of recommended actions for the team.""")

        return "\n".join(parts)

    def _parse_response(self, text: str, stats: Dict[str, Any]) -> CommentaryResult:
        sections = self._parse_sections(
            text, ["EXECUTIVE_SUMMARY", "KEY_METRICS", "EXCEPTIONS", "ACTION_ITEMS"]
        )

        executive_summary = sections.get("EXECUTIVE_SUMMARY", "")

        # Parse key metrics
        key_metrics = []
        metrics_data = self._parse_json_block(sections.get("KEY_METRICS", ""))
        if isinstance(metrics_data, list):
            for item in metrics_data:
                key_metrics.append(KeyMetric(
                    name=item.get("name", ""),
                    value=float(item.get("value", 0)),
                    unit=item.get("unit", ""),
                    trend=item.get("trend", "stable"),
                ))

        exceptions = self._parse_bullets(sections.get("EXCEPTIONS", ""))
        action_items = self._parse_bullets(sections.get("ACTION_ITEMS", ""))

        if not executive_summary:
            return self._fallback_result("unknown", stats, None)

        return CommentaryResult(
            executive_summary=executive_summary,
            key_metrics=key_metrics,
            exceptions=exceptions,
            action_items=action_items,
        )

    @staticmethod
    def _fallback_result(
        lob: str,
        stats: Dict[str, Any],
        drift_alerts: Optional[List[Any]],
    ) -> CommentaryResult:
        """Template-based summary when Claude is unavailable."""
        wmape = stats.get("overall_wmape", 0)
        bias = stats.get("overall_bias", 0)
        n_series = stats.get("n_series", 0)

        alert_count = len(drift_alerts) if drift_alerts else 0
        critical_count = sum(1 for a in drift_alerts if a.severity.value == "critical") if drift_alerts else 0

        summary = (
            f"Forecast performance for LOB '{lob}': overall WMAPE {wmape:.1%} "
            f"across {n_series} series with average bias of {bias:+.1%}. "
        )
        if alert_count > 0:
            summary += f"{alert_count} drift alerts detected ({critical_count} critical). "
        summary += "AI commentary unavailable — template-based summary shown."

        key_metrics = []
        if "overall_wmape" in stats:
            key_metrics.append(KeyMetric(name="WMAPE", value=wmape, unit="%", trend="stable"))
        if "overall_bias" in stats:
            key_metrics.append(KeyMetric(name="Bias", value=bias, unit="%", trend="stable"))

        exceptions = []
        if drift_alerts:
            for a in drift_alerts[:3]:
                if a.severity.value == "critical":
                    exceptions.append(f"{a.series_id}: {a.message}")

        return CommentaryResult(
            executive_summary=summary,
            key_metrics=key_metrics,
            exceptions=exceptions,
            action_items=["Review critical drift alerts", "Validate data pipeline"],
        )

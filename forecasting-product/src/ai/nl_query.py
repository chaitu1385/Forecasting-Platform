"""Natural Language Querying — answer planner questions about forecasts.

Takes a series ID, a natural-language question, and available context
(decomposition, SHAP, drift alerts, comparisons) and uses Claude to
generate a narrative answer.

Gracefully degrades when Claude is unavailable.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from .base import AIFeatureBase

logger = logging.getLogger(__name__)


@dataclass
class NLQueryResult:
    """Result of a natural language query about a forecast."""
    answer: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "low"          # "high" | "medium" | "low"
    sources_used: List[str] = field(default_factory=list)


class NaturalLanguageQueryEngine(AIFeatureBase):
    """Claude-powered natural language querying for forecast data.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.
    model : str
        Claude model to use.
    """

    def query(
        self,
        series_id: str,
        question: str,
        lob: str,
        history: Optional[pl.DataFrame] = None,
        forecast: Optional[pl.DataFrame] = None,
        metrics_df: Optional[pl.DataFrame] = None,
        ml_model: Optional[Any] = None,
        features_df: Optional[pl.DataFrame] = None,
        comparison: Optional[pl.DataFrame] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> NLQueryResult:
        """Answer a natural-language question about a specific series.

        Parameters
        ----------
        series_id : str
            The series to analyze.
        question : str
            The planner's question in natural language.
        lob : str
            Line of business identifier.
        history : pl.DataFrame, optional
            Historical data for the series.
        forecast : pl.DataFrame, optional
            Forecast data for the series.
        metrics_df : pl.DataFrame, optional
            Metric store data for the series.
        ml_model : object, optional
            Fitted ML model for SHAP attribution.
        features_df : pl.DataFrame, optional
            Feature DataFrame for SHAP computation.
        comparison : pl.DataFrame, optional
            Comparison data (e.g. vs external forecasts).
        lineage : dict, optional
            Verified source labels or explicit unknown legacy provenance.

        Returns
        -------
        NLQueryResult
        """
        if not self.available:
            return NLQueryResult(answer="AI analysis unavailable — Claude client not configured.")

        # Gather context from available data
        context, sources = self._gather_context(
            series_id, history, forecast, metrics_df,
            ml_model, features_df, comparison,
        )
        if lineage is not None:
            context["lineage"] = dict(lineage)

        try:
            prompt = self._build_prompt(series_id, question, lob, context)
            text = self._call_claude(self._system_prompt(), prompt)
            return self._parse_response(text, sources)
        except Exception as e:
            logger.error("NaturalLanguageQueryEngine: API call failed: %s", e)
            return NLQueryResult(
                answer=f"AI analysis unavailable: {e}",
                sources_used=sources,
            )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a senior demand planning analyst answering questions from "
            "retail planners about their forecasts. Provide clear, specific answers "
            "grounded in the data provided. If the data is insufficient to answer "
            "confidently, say so. Use business language, not technical jargon."
        )

    def _gather_context(
        self,
        series_id: str,
        history: Optional[pl.DataFrame],
        forecast: Optional[pl.DataFrame],
        metrics_df: Optional[pl.DataFrame],
        ml_model: Optional[Any],
        features_df: Optional[pl.DataFrame],
        comparison: Optional[pl.DataFrame],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Collect available data into a context dict for the prompt."""
        context: Dict[str, Any] = {}
        sources: List[str] = []

        # History summary
        if history is not None and not history.is_empty():
            series_hist = history.filter(pl.col("series_id") == series_id) if "series_id" in history.columns else history
            if not series_hist.is_empty() and "quantity" in series_hist.columns:
                vals = series_hist["quantity"].drop_nulls()
                if len(vals) > 0:
                    context["history"] = {
                        "n_periods": len(vals),
                        "mean": round(float(vals.mean()), 2),
                        "std": round(float(vals.std()), 2) if len(vals) > 1 else 0,
                        "min": round(float(vals.min()), 2),
                        "max": round(float(vals.max()), 2),
                        "last_4": [round(float(v), 2) for v in vals.tail(4).to_list()],
                    }
                    sources.append("history")

        # Forecast summary
        if forecast is not None and not forecast.is_empty():
            series_fc = forecast.filter(pl.col("series_id") == series_id) if "series_id" in forecast.columns else forecast
            fc_col = "forecast" if "forecast" in series_fc.columns else None
            if fc_col and not series_fc.is_empty():
                vals = series_fc[fc_col].drop_nulls()
                if len(vals) > 0:
                    context["forecast"] = {
                        "n_periods": len(vals),
                        "mean": round(float(vals.mean()), 2),
                        "first_4": [round(float(v), 2) for v in vals.head(4).to_list()],
                    }
                    sources.append("forecast")

        # Metrics summary
        if metrics_df is not None and not metrics_df.is_empty():
            series_metrics = metrics_df.filter(pl.col("series_id") == series_id) if "series_id" in metrics_df.columns else metrics_df
            if not series_metrics.is_empty():
                from ..metrics.aggregation import observation_metrics

                metric_ctx: Dict[str, Any] = {}
                if {"actual", "forecast"}.issubset(series_metrics.columns):
                    scores = observation_metrics(
                        series_metrics, ["wmape", "normalized_bias"], preserve_context=False,
                    ).row(0, named=True)
                    for name, output in (("wmape", "wmape"), ("normalized_bias", "bias")):
                        if scores[name] is not None:
                            metric_ctx[output] = round(scores[name], 4)
                    metric_ctx["aggregation"] = scores["metric_population"]
                    metric_ctx["scoring_policy"] = scores["scoring_policy"]
                else:
                    metric_ctx["aggregation"] = "summary-only: raw observations required for WMAPE/bias"
                if metric_ctx:
                    context["metrics"] = metric_ctx
                    sources.append("metrics")

        # Comparison summary
        if comparison is not None and not comparison.is_empty():
            series_cmp = comparison.filter(pl.col("series_id") == series_id) if "series_id" in comparison.columns else comparison
            if not series_cmp.is_empty():
                context["comparison"] = {"n_rows": len(series_cmp)}
                sources.append("comparison")

        return context, sources

    def _build_prompt(
        self,
        series_id: str,
        question: str,
        lob: str,
        context: Dict[str, Any],
    ) -> str:
        context_json = json.dumps(context, indent=2, default=str)

        return f"""Answer the following question about series '{series_id}' in LOB '{lob}'.

## Available Data
```json
{context_json}
```

## Question
{question}

## Instructions
Respond with these exact section headers:

### ANSWER
Your answer in 2-5 sentences, grounded in the data above.

### CONFIDENCE
One word: "high", "medium", or "low" — based on how much relevant data was available.

### SUPPORTING_DATA
A JSON object with 2-4 key metrics that support your answer."""

    def _parse_response(self, text: str, sources: List[str]) -> NLQueryResult:
        sections = self._parse_sections(text, ["ANSWER", "CONFIDENCE", "SUPPORTING_DATA"])

        answer = sections.get("ANSWER", "")
        confidence_raw = sections.get("CONFIDENCE", "low").strip().lower()
        confidence = confidence_raw if confidence_raw in ("high", "medium", "low") else "low"

        supporting_data = {}
        parsed = self._parse_json_block(sections.get("SUPPORTING_DATA", ""))
        if isinstance(parsed, dict):
            supporting_data = parsed

        return NLQueryResult(
            answer=answer,
            supporting_data=supporting_data,
            confidence=confidence,
            sources_used=sources,
        )

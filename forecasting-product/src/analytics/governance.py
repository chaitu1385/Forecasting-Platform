"""
Model governance — drift detection, model cards, and forecast lineage.

Components
----------
DriftDetector
    Compares live residual distribution to the backtest baseline.
    Uses a simple variance-ratio test; falls back gracefully when there
    is insufficient live data.

    Status levels:
      "ok"      — live WMAPE ≤ backtest_wmape × warn_multiplier
      "warning" — between warn_multiplier and alert_multiplier
      "alert"   — live WMAPE > backtest_wmape × alert_multiplier

ModelCard
    Structured metadata for a trained model: training window, series
    count, backtest metrics, feature list, config hash.
    Serializable to a plain dict for storage or display.

ModelCardRegistry
    In-memory registry of ModelCard objects indexed by model_name.
    Persists to / loads from a Parquet file for lightweight governance.

ForecastLineage
    Records which model produced each forecast run.  Append-only.
    Provides a query interface for "what model was used on date X?"
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from ..metrics.store import MetricStore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Detect performance drift by comparing live WMAPE to backtest WMAPE.

    Parameters
    ----------
    metric_store:
        Shared MetricStore instance.
    warn_multiplier:
        Ratio live/backtest above which status becomes "warning".
    alert_multiplier:
        Ratio live/backtest above which status becomes "alert".
    min_live_weeks:
        Minimum number of live weeks required before flagging drift.
    """

    def __init__(
        self,
        metric_store: MetricStore,
        warn_multiplier: float = 1.25,
        alert_multiplier: float = 1.50,
        min_live_weeks: int = 4,
    ):
        self.metric_store = metric_store
        self.warn_multiplier = warn_multiplier
        self.alert_multiplier = alert_multiplier
        self.min_live_weeks = min_live_weeks

    def detect(
        self,
        model_id: str,
        lob: Optional[str] = None,
        metric: str = "wmape",
        n_recent_weeks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Check one model for drift.

        Parameters
        ----------
        n_recent_weeks:
            If set, only use the most recent N weeks of live data.

        Returns
        -------
        Dict with keys:
          model_id, lob, metric,
          backtest_score, live_score, ratio,
          status ("ok" | "warning" | "alert" | "insufficient_data"),
          n_live_weeks
        """
        bt = self.metric_store.read(run_type="backtest", lob=lob, model_id=model_id)
        live = self.metric_store.read(run_type="live", lob=lob, model_id=model_id)

        base = {
            "model_id": model_id,
            "lob": lob or "",
            "metric": metric,
        }

        if bt.is_empty() or metric not in bt.columns:
            return {**base, "status": "insufficient_data",
                    "backtest_score": None, "live_score": None,
                    "ratio": None, "n_live_weeks": 0}

        backtest_score = float(bt[metric].drop_nulls().mean())

        if live.is_empty() or metric not in live.columns:
            return {**base, "status": "insufficient_data",
                    "backtest_score": backtest_score, "live_score": None,
                    "ratio": None, "n_live_weeks": 0}

        if n_recent_weeks is not None and "target_week" in live.columns:
            max_week = live["target_week"].max()
            if max_week is not None:
                live = live.sort("target_week").tail(n_recent_weeks * live["series_id"].n_unique())

        n_live_weeks = live["target_week"].n_unique() if "target_week" in live.columns else 0

        if n_live_weeks < self.min_live_weeks:
            return {**base, "status": "insufficient_data",
                    "backtest_score": backtest_score, "live_score": None,
                    "ratio": None, "n_live_weeks": n_live_weeks}

        live_score = float(live[metric].drop_nulls().mean())
        ratio = live_score / backtest_score if backtest_score > 0 else None

        if ratio is None:
            status = "insufficient_data"
        elif ratio > self.alert_multiplier:
            status = "alert"
        elif ratio > self.warn_multiplier:
            status = "warning"
        else:
            status = "ok"

        return {
            **base,
            "backtest_score": backtest_score,
            "live_score": live_score,
            "ratio": ratio,
            "status": status,
            "n_live_weeks": n_live_weeks,
        }

    def batch_detect(
        self,
        lob: Optional[str] = None,
        metric: str = "wmape",
    ) -> pl.DataFrame:
        """
        Run drift detection for all models in a LOB.

        Returns a DataFrame summarising drift status per model,
        sorted by ratio descending (most degraded first).
        """
        bt = self.metric_store.read(run_type="backtest", lob=lob)
        if bt.is_empty() or "model_id" not in bt.columns:
            return pl.DataFrame(schema={
                "model_id": pl.Utf8, "status": pl.Utf8,
                "backtest_score": pl.Float64, "live_score": pl.Float64,
                "ratio": pl.Float64,
            })

        model_ids = bt["model_id"].unique().to_list()
        rows = [self.detect(mid, lob=lob, metric=metric) for mid in model_ids]

        return (
            pl.DataFrame(rows)
            .sort("ratio", descending=True, nulls_last=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model card
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCard:
    """
    Structured metadata for a trained forecasting model.

    Captures everything needed to reproduce or audit a model:
    training window, data fingerprint, backtest metrics, and the
    feature set used by ML models.
    """
    model_name: str
    lob: str
    training_start: Optional[date] = None
    training_end: Optional[date] = None
    n_series: int = 0
    n_observations: int = 0
    backtest_wmape: Optional[float] = None
    backtest_bias: Optional[float] = None
    champion_since: Optional[date] = None
    features: List[str] = field(default_factory=list)
    config_hash: str = ""
    notes: str = ""

    @classmethod
    def from_backtest(
        cls,
        model_name: str,
        lob: str,
        backtest_results: pl.DataFrame,
        champion_since: Optional[date] = None,
        features: Optional[List[str]] = None,
        config: Optional[Any] = None,
        notes: str = "",
    ) -> "ModelCard":
        """
        Build a ModelCard from a BacktestEngine results DataFrame.

        Parameters
        ----------
        backtest_results:
            Per-(model, fold, series, week) metric DataFrame from
            BacktestEngine.run().
        """
        model_rows = (
            backtest_results.filter(pl.col("model_id") == model_name)
            if "model_id" in backtest_results.columns
            else backtest_results
        )

        wmape = (
            float(model_rows["wmape"].drop_nulls().mean())
            if not model_rows.is_empty() and "wmape" in model_rows.columns
            else None
        )
        bias = (
            float(model_rows["normalized_bias"].drop_nulls().mean())
            if not model_rows.is_empty() and "normalized_bias" in model_rows.columns
            else None
        )
        n_series = (
            model_rows["series_id"].n_unique()
            if "series_id" in model_rows.columns
            else 0
        )

        config_hash = ""
        if config is not None:
            try:
                config_hash = hashlib.md5(
                    json.dumps(asdict(config) if hasattr(config, "__dataclass_fields__") else str(config),
                               sort_keys=True).encode()
                ).hexdigest()[:8]
            except Exception:
                logger.debug("Failed to compute config hash", exc_info=True)

        return cls(
            model_name=model_name,
            lob=lob,
            n_series=n_series,
            n_observations=len(model_rows),
            backtest_wmape=wmape,
            backtest_bias=bias,
            champion_since=champion_since or date.today(),
            features=features or [],
            config_hash=config_hash,
            notes=notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d = asdict(self)
        for k in ("training_start", "training_end", "champion_since"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        return d

    def to_frame(self) -> pl.DataFrame:
        """Return a single-row Polars DataFrame."""
        d = self.to_dict()
        d["features"] = ", ".join(d["features"])
        return pl.DataFrame([d])


class ModelCardRegistry:
    """
    Lightweight registry of ModelCard objects backed by a Parquet file.

    Usage
    -----
    >>> reg = ModelCardRegistry("data/model_cards/")
    >>> reg.register(card)
    >>> reg.get("lgbm_direct")
    >>> reg.all_cards()
    """

    def __init__(self, base_path: str = "data/model_cards/"):
        self._path = Path(base_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._cards: Dict[str, ModelCard] = {}
        self._load()

    def register(self, card: ModelCard) -> None:
        """Add or update a model card and persist to disk."""
        self._cards[card.model_name] = card
        self._persist()

    def get(self, model_name: str) -> Optional[ModelCard]:
        """Retrieve a model card by name."""
        return self._cards.get(model_name)

    def all_cards(self) -> pl.DataFrame:
        """Return all registered model cards as a DataFrame."""
        if not self._cards:
            return pl.DataFrame()
        rows = [c.to_dict() for c in self._cards.values()]
        for r in rows:
            r["features"] = ", ".join(r["features"])
        return pl.DataFrame(rows)

    def _persist(self) -> None:
        df = self.all_cards()
        if not df.is_empty():
            df.write_parquet(str(self._path / "model_cards.parquet"))

    def _load(self) -> None:
        p = self._path / "model_cards.parquet"
        if p.exists():
            try:
                df = pl.read_parquet(str(p))
                for row in df.iter_rows(named=True):
                    features = [f.strip() for f in row.get("features", "").split(",") if f.strip()]
                    card = ModelCard(
                        model_name=row.get("model_name", ""),
                        lob=row.get("lob", ""),
                        n_series=int(row.get("n_series", 0)),
                        n_observations=int(row.get("n_observations", 0)),
                        backtest_wmape=row.get("backtest_wmape"),
                        backtest_bias=row.get("backtest_bias"),
                        config_hash=row.get("config_hash", ""),
                        features=features,
                        notes=row.get("notes", ""),
                    )
                    self._cards[card.model_name] = card
            except Exception:
                logger.debug("Failed to load model cards from storage", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Forecast lineage
# ─────────────────────────────────────────────────────────────────────────────

class ForecastLineage:
    """
    Append-only log of which model produced each forecast run.

    Provides an audit trail answering: "which model was live on date X?"

    Each record captures:
      run_date, lob, model_id, champion_selection_strategy,
      n_series, forecast_horizon_weeks, run_id (optional)
    """

    _SCHEMA = {
        "run_date": pl.Date,
        "lob": pl.Utf8,
        "model_id": pl.Utf8,
        "selection_strategy": pl.Utf8,
        "n_series": pl.Int32,
        "horizon_weeks": pl.Int32,
        "run_id": pl.Utf8,
        "notes": pl.Utf8,
        "user_id": pl.Utf8,
    }

    def __init__(self, base_path: str = "data/lineage/"):
        self._path = Path(base_path)

    def record(
        self,
        lob: str,
        model_id: str,
        n_series: int = 0,
        horizon_weeks: int = 0,
        selection_strategy: str = "champion",
        run_id: str = "",
        notes: str = "",
        run_date: Optional[date] = None,
        user_id: str = "system",
    ) -> None:
        """Append one lineage record."""
        row = pl.DataFrame([{
            "run_date": run_date or date.today(),
            "lob": lob,
            "model_id": model_id,
            "selection_strategy": selection_strategy,
            "n_series": n_series,
            "horizon_weeks": horizon_weeks,
            "run_id": run_id,
            "notes": notes,
            "user_id": user_id,
        }])

        from ..pipeline.run_store import atomic_frame_append

        atomic_frame_append(row, self._path, prefix="lineage")

    def history(
        self,
        lob: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Read the full lineage log, optionally filtered.

        Returns a DataFrame sorted by run_date descending.
        """
        from ..pipeline.run_store import RunIntegrityError, checked_child

        files = sorted(self._path.glob("*.parquet"))
        if not files:
            return pl.DataFrame(schema=self._SCHEMA)
        try:
            frames = [
                pl.read_parquet(checked_child(self._path, filename.name), hive_partitioning=False)
                for filename in files
            ]
            df = pl.concat(frames, how="diagonal_relaxed")
        except Exception as exc:
            raise RunIntegrityError("unreadable_legacy_lineage") from exc

        if lob:
            df = df.filter(pl.col("lob") == lob)
        if model_id:
            df = df.filter(pl.col("model_id") == model_id)

        return df.sort("run_date", descending=True)

    def latest(self, lob: str) -> Optional[Dict[str, Any]]:
        """Return the most recent lineage record for a LOB."""
        hist = self.history(lob=lob)
        if hist.is_empty():
            return None
        return hist.head(1).to_dicts()[0]

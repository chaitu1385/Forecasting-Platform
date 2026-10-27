"""
Foundation model forecasters — zero-shot inference, no training required.

These models are pre-trained on massive, diverse time series corpora and can
forecast any new series without any fine-tuning.  This makes them especially
valuable for:

  - **Cold-start SKUs** with little or no sales history
  - **New product launches** that have no analogues in the product catalogue
  - **Rapid prototyping** — run a competitive forecast in seconds without a
    backtest loop

Two backends are provided:

``ChronosForecaster``
    Amazon Chronos (open-source, Apache 2.0).  Runs locally via HuggingFace
    Transformers.  No API key required.  Natively probabilistic — outputs a
    full predictive distribution via Monte Carlo samples.
    Install: ``pip install chronos-forecasting torch``
    Models (small → large): chronos-t5-tiny / small / base / large / huge
                            chronos-bolt-tiny / small / base / large (faster)

``TimeGPTForecaster``
    Nixtla TimeGPT (API-based).  Requires a ``NIXTLA_API_KEY`` environment
    variable or ``api_key`` constructor argument.
    Install: ``pip install nixtla``

Both implement the full ``BaseForecaster`` interface including
``predict_quantiles()``, which for these models comes for free as a natural
output of their probabilistic design (no extra models needed).

Usage
-----
>>> from src.forecasting.foundation import ChronosForecaster
>>> f = ChronosForecaster(model_name="amazon/chronos-t5-tiny")
>>> f.fit(train_df)            # stores context — no training
>>> forecast = f.predict(13)   # zero-shot inference
>>> intervals = f.predict_quantiles(13, [0.1, 0.5, 0.9])
"""

import logging
import os
from typing import Any, Dict, List, Optional

import polars as pl

from ..utils.calendar import PeriodCalendar
from .base import BaseForecaster
from .registry import registry

logger = logging.getLogger(__name__)


# ── Chronos ────────────────────────────────────────────────────────────────────


@registry.register("chronos")
class ChronosForecaster(BaseForecaster):
    """
    Amazon Chronos zero-shot forecaster.

    Wraps the ``ChronosPipeline`` from the ``chronos-forecasting`` package.
    The model is downloaded from HuggingFace on the first ``predict()`` call
    and cached locally for subsequent use.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Smaller models are faster on CPU:
        ``"amazon/chronos-t5-tiny"`` (8 M params, CPU-friendly) or
        ``"amazon/chronos-bolt-tiny"`` (fastest variant).
    device:
        Torch device string — ``"cpu"`` or ``"cuda"``.  Defaults to CPU.
    num_samples:
        Number of Monte Carlo samples for uncertainty estimation.
        More samples → tighter quantile estimates; 20–50 is typical.
    torch_dtype:
        Torch dtype string (``"bfloat16"`` or ``"float32"``).  bfloat16
        saves memory on modern hardware; use float32 on older CPUs.
    """

    name = "chronos"

    _ZERO_SHOT_MSG = (
        "ChronosForecaster: fit() stores historical context only — "
        "no model training occurs (zero-shot foundation model)."
    )

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        device: str = "cpu",
        num_samples: int = 20,
        torch_dtype: str = "bfloat16",
        frequency: str = "W",
    ):
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self.torch_dtype = torch_dtype
        self.frequency = frequency

        self._pipeline: Optional[Any] = None          # lazy-loaded on first predict
        self._context: Dict[str, List[float]] = {}    # series_id → historical values
        self._last_dates: Dict[str, Any] = {}         # series_id → last observed date
        self._id_col = "series_id"
        self._time_col = "week"

    # ── Public interface ───────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """
        Store historical context for inference.

        No model weights are updated — Chronos is zero-shot.
        """
        logger.info(self._ZERO_SHOT_MSG)
        self._id_col = id_col
        self._time_col = time_col
        self._context.clear()
        self._last_dates.clear()
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)

        for sid in df[id_col].unique().to_list():
            series = df.filter(pl.col(id_col) == sid).sort(time_col)
            self._context[sid] = series[target_col].to_list()
            self._last_dates[sid] = series[time_col].max()

        logger.info(
            "Chronos context stored: %d series, avg length %.0f",
            len(self._context),
            sum(len(v) for v in self._context.values()) / max(len(self._context), 1),
        )

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Zero-shot point forecast (median of predictive distribution)."""
        return self._predict_impl(horizon, quantiles=None, id_col=id_col, time_col=time_col)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Zero-shot probabilistic forecast.

        Quantile estimates come directly from the Monte Carlo samples of the
        Chronos predictive distribution — no extra models, no separate
        calibration required.
        """
        return self._predict_impl(horizon, quantiles=quantiles, id_col=id_col, time_col=time_col)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "Chronos",
            "model_name": self.model_name,
            "device": self.device,
            "num_samples": self.num_samples,
            "frequency": self.frequency,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        """Lazy-load Chronos model weights (downloads from HuggingFace once)."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from chronos import ChronosPipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "ChronosForecaster requires 'chronos-forecasting' and 'torch'.\n"
                "Install with:  pip install chronos-forecasting torch"
            ) from exc

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        logger.info("Loading Chronos model '%s' on device '%s'…", self.model_name, self.device)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=dtype,
        )
        logger.info("Chronos model loaded.")

    def _predict_impl(
        self,
        horizon: int,
        quantiles: Optional[List[float]],
        id_col: str,
        time_col: str,
    ) -> pl.DataFrame:
        """
        Shared predict/predict_quantiles implementation.

        Runs Chronos in batch mode (all series in one forward pass) and
        extracts the median (for point forecast) or arbitrary quantile levels
        from the Monte Carlo sample distribution.
        """
        import numpy as np

        self._load_pipeline()

        series_ids = list(self._context.keys())
        if not series_ids:
            schema: Dict = {id_col: pl.Utf8, time_col: pl.Date}
            if quantiles:
                for q in quantiles:
                    schema[f"forecast_p{int(round(q * 100))}"] = pl.Float64
            else:
                schema["forecast"] = pl.Float64
            return pl.DataFrame(schema=schema)

        # Build batch context tensors (torch only needed here)
        try:
            import torch
            contexts = [
                torch.tensor(self._context[sid], dtype=torch.float32)
                for sid in series_ids
            ]
        except ImportError as exc:
            raise ImportError(
                "ChronosForecaster requires 'torch'.\n"
                "Install with:  pip install torch"
            ) from exc

        logger.info(
            "Chronos: forecasting %d series × %d weeks (%d samples)…",
            len(series_ids), horizon, self.num_samples,
        )
        raw = self._pipeline.predict(
            context=contexts,
            prediction_length=horizon,
            num_samples=self.num_samples,
        )
        # raw shape: [n_series, num_samples, horizon]
        samples_np = raw.numpy()

        results = []
        for i, sid in enumerate(series_ids):
            last_date = self._last_dates[sid]
            # samples_np[i] shape: [num_samples, horizon]
            series_samples = samples_np[i]

            for h in range(horizon):
                forecast_date = self._calendar.shift(last_date, h + 1)
                row: Dict[str, Any] = {id_col: sid, time_col: forecast_date}

                if quantiles is None:
                    # Point forecast = median
                    row["forecast"] = float(np.median(series_samples[:, h]))
                else:
                    for q in quantiles:
                        col = f"forecast_p{int(round(q * 100))}"
                        row[col] = float(np.quantile(series_samples[:, h], q))

                results.append(row)

        return pl.DataFrame(results)


# ── TimeGPT ────────────────────────────────────────────────────────────────────


@registry.register("timegpt")
class TimeGPTForecaster(BaseForecaster):
    """
    Nixtla TimeGPT zero-shot forecaster (API-based).

    Makes REST calls to the Nixtla API.  Requires a valid API key via the
    ``NIXTLA_API_KEY`` environment variable or the ``api_key`` constructor
    parameter.  See https://docs.nixtla.io for API key setup.

    Parameters
    ----------
    api_key:
        Nixtla API key.  If ``None``, reads from the ``NIXTLA_API_KEY``
        environment variable.
    model:
        TimeGPT model variant: ``"timegpt-1"`` (default, up to 512 steps)
        or ``"timegpt-1-long-horizon"`` (for horizons > 36 periods).
    """

    name = "timegpt"

    _ZERO_SHOT_MSG = (
        "TimeGPTForecaster: fit() stores historical context only — "
        "no model training occurs (zero-shot foundation model via Nixtla API)."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "timegpt-1",
        frequency: str = "W",
    ):
        self.api_key = api_key or os.environ.get("NIXTLA_API_KEY", "")
        self.model = model
        self.frequency = frequency

        self._client: Optional[Any] = None          # lazy-initialised
        self._train_df: Optional[pl.DataFrame] = None  # stored as Polars; converted on API call
        self._id_col = "series_id"
        self._time_col = "week"
        self._target_col = "quantity"

    # ── Public interface ───────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """
        Store training data for zero-shot inference.

        Stores training data as a Polars DataFrame; pandas conversion is
        deferred to API call time so that ``pyarrow`` is not required until
        ``predict()`` is actually called.  No API call is made at fit stage.
        """
        logger.info(self._ZERO_SHOT_MSG)
        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)

        self._train_df = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
        )
        logger.info(
            "TimeGPT context stored: %d series",
            self._train_df["unique_id"].n_unique(),
        )

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Zero-shot point forecast via TimeGPT API."""
        return self._call_api(horizon, levels=None, id_col=id_col, time_col=time_col)

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Zero-shot probabilistic forecast via TimeGPT API.

        Maps quantile levels to the ``level`` parameter of the Nixtla client
        (e.g. ``[0.1, 0.9]`` → ``level=[80]``, which returns ``lo-80`` / ``hi-80``).
        """
        lower_qs = sorted(q for q in quantiles if q < 0.5 - 1e-6)
        levels = sorted({int(round((1 - 2 * q) * 100)) for q in lower_qs}) if lower_qs else [80]

        self._call_api(horizon, levels=levels, id_col=id_col, time_col=time_col)

        # _call_api returns a "forecast" point column; rename + add quantile columns
        # Get the raw result with interval columns by re-running with level param
        return self._map_to_quantile_frame(
            horizon=horizon,
            quantiles=quantiles,
            levels=levels,
            id_col=id_col,
            time_col=time_col,
        )

    def get_params(self) -> Dict[str, Any]:
        return {"model": "TimeGPT", "timegpt_model": self.model, "frequency": self.frequency}

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazily initialise the Nixtla client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "TimeGPTForecaster requires an API key.\n"
                "Set the NIXTLA_API_KEY environment variable or pass api_key= "
                "to the constructor."
            )

        try:
            from nixtla import NixtlaClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "TimeGPTForecaster requires 'nixtla'.\n"
                "Install with:  pip install nixtla"
            ) from exc

        self._client = NixtlaClient(api_key=self.api_key)
        return self._client

    def _call_api(
        self,
        horizon: int,
        levels: Optional[List[int]],
        id_col: str,
        time_col: str,
    ) -> pl.DataFrame:
        """
        Call the TimeGPT forecast API and return a Polars DataFrame.

        For point forecasts (``levels=None``), returns ``[id_col, time_col, "forecast"]``.
        For interval forecasts, returns the raw DataFrame with all columns from the API.
        """
        if self._train_df is None:
            raise RuntimeError("TimeGPTForecaster: call fit() before predict().")

        client = self._get_client()
        # Convert to pandas here (requires pyarrow) — deferred from fit()
        train_pdf = self._train_df.to_pandas()
        train_pdf["ds"] = train_pdf["ds"].astype("datetime64[ns]")

        kwargs: Dict[str, Any] = {
            "df": train_pdf,
            "h": horizon,
            "freq": self._calendar.pandas_frequency,
            "model": self.model,
            "time_col": "ds",
            "target_col": "y",
            "id_col": "unique_id",
        }
        if levels:
            kwargs["level"] = levels

        logger.info(
            "TimeGPT: forecasting %d series × %d weeks (model=%s)…",
            self._train_df["unique_id"].n_unique(), horizon, self.model,
        )
        result_pdf = client.forecast(**kwargs)
        result = pl.from_pandas(result_pdf)
        result = result.rename({"unique_id": id_col, "ds": time_col})
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        # For point-only calls, return clean 3-column frame
        if levels is None:
            point_col = next(
                (c for c in result.columns if "TimeGPT" in c and "-lo-" not in c and "-hi-" not in c),
                None,
            )
            if point_col:
                result = result.rename({point_col: "forecast"})
            return result.select([id_col, time_col, "forecast"])

        return result

    def _map_to_quantile_frame(
        self,
        horizon: int,
        quantiles: List[float],
        levels: List[int],
        id_col: str,
        time_col: str,
    ) -> pl.DataFrame:
        """Map raw TimeGPT API response columns to standard forecast_p{q} columns."""
        raw = self._call_api(horizon, levels=levels, id_col=id_col, time_col=time_col)

        # Identify point forecast column
        point_col = next(
            (c for c in raw.columns
             if "TimeGPT" in c and "-lo-" not in c and "-hi-" not in c),
            None,
        )

        output = raw.select([id_col, time_col])
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                src = raw[point_col] if point_col else None
            elif q < 0.5:
                level = int(round((1 - 2 * q) * 100))
                lo_col = next((c for c in raw.columns if f"-lo-{level}" in c), None)
                src = raw[lo_col] if lo_col else (raw[point_col] if point_col else None)
            else:
                mirror_q = 1.0 - q
                level = int(round((1 - 2 * mirror_q) * 100))
                hi_col = next((c for c in raw.columns if f"-hi-{level}" in c), None)
                src = raw[hi_col] if hi_col else (raw[point_col] if point_col else None)

            if src is not None:
                output = output.with_columns(src.alias(col))

        return output

"""
Planner override persistence — DuckDB (default) or Parquet (Fabric fallback).

Stores human-in-the-loop overrides for product transitions:
  - Proportion adjustments
  - Scenario overrides (force Scenario A/B/C/manual)
  - Ramp shape changes

DuckDB is the primary backend (zero-server, file-based, SQL-queryable,
native Polars interop via Arrow).  When DuckDB is unavailable (e.g.
Microsoft Fabric Spark runtime), the store falls back to Parquet files
with Polars-native read/write.

Use ``get_override_store()`` to auto-select the best available backend.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

# DuckDB is optional — graceful fallback to CSV if unavailable
try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transition_overrides (
    override_id     VARCHAR PRIMARY KEY,
    old_sku         VARCHAR NOT NULL,
    new_sku         VARCHAR NOT NULL,
    effective_date  DATE,
    scenario        VARCHAR DEFAULT 'manual',
    proportion      DOUBLE,
    ramp_shape      VARCHAR DEFAULT 'linear',
    created_by      VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes           VARCHAR,
    status          VARCHAR DEFAULT 'approved',
    approved_by     VARCHAR,
    approved_at     TIMESTAMP
)
"""


class OverrideStore:
    """
    Persistent store for planner transition overrides.

    If DuckDB is available, uses a file-based DuckDB database.
    Otherwise, falls back to reading/writing a CSV file.
    """

    def __init__(self, db_path: str = "data/overrides.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if _HAS_DUCKDB:
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_CREATE_TABLE_SQL)
        else:
            self._conn = None
            self._csv_path = self.db_path.with_suffix(".csv")

    def add_override(
        self,
        old_sku: str,
        new_sku: str,
        proportion: float,
        scenario: str = "manual",
        ramp_shape: str = "linear",
        effective_date: Optional[str] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
        approval_threshold: float = 0.0,
    ) -> str:
        """
        Add a planner override.

        Parameters
        ----------
        approval_threshold:
            If proportion exceeds this threshold, the override is created
            with status 'pending_approval' and requires manager sign-off.
            A threshold of 0.0 means all overrides are auto-approved.

        Returns the generated override_id.
        """
        override_id = f"OVR-{uuid.uuid4().hex[:8].upper()}"

        # Determine approval status based on threshold
        if approval_threshold > 0.0 and proportion > approval_threshold:
            status = "pending_approval"
        else:
            status = "approved"

        if self._conn:
            self._conn.execute(
                """
                INSERT INTO transition_overrides
                (override_id, old_sku, new_sku, effective_date, scenario,
                 proportion, ramp_shape, created_by, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    override_id, old_sku, new_sku, effective_date,
                    scenario, proportion, ramp_shape, created_by, notes,
                    status,
                ],
            )
        else:
            # CSV fallback
            record = pl.DataFrame({
                "override_id": [override_id],
                "old_sku": [old_sku],
                "new_sku": [new_sku],
                "effective_date": [effective_date],
                "scenario": [scenario],
                "proportion": [proportion],
                "ramp_shape": [ramp_shape],
                "created_by": [created_by],
                "created_at": [datetime.now().isoformat()],
                "notes": [notes],
                "status": [status],
                "approved_by": [None],
                "approved_at": [None],
            })
            if self._csv_path.exists():
                existing = pl.read_csv(str(self._csv_path))
                combined = pl.concat(
                    [existing, record], how="vertical_relaxed"
                )
                combined.write_csv(str(self._csv_path))
            else:
                record.write_csv(str(self._csv_path))

        return override_id

    def get_overrides(
        self,
        old_sku: Optional[str] = None,
        new_sku: Optional[str] = None,
    ) -> pl.DataFrame:
        """Retrieve overrides, optionally filtered by SKU."""
        if self._conn:
            query = "SELECT * FROM transition_overrides WHERE 1=1"
            params = []
            if old_sku:
                query += " AND old_sku = ?"
                params.append(old_sku)
            if new_sku:
                query += " AND new_sku = ?"
                params.append(new_sku)
            result = self._conn.execute(query, params)
            # DuckDB 2 removed .pl() and the deprecated fetch_arrow_table name.
            if hasattr(result, "to_arrow_table"):
                return pl.from_arrow(result.to_arrow_table())
            return pl.from_arrow(result.fetch_arrow_table())
        else:
            if not self._csv_path.exists():
                return pl.DataFrame()
            df = pl.read_csv(str(self._csv_path))
            if old_sku:
                df = df.filter(pl.col("old_sku") == old_sku)
            if new_sku:
                df = df.filter(pl.col("new_sku") == new_sku)
            return df

    def get_all(self) -> pl.DataFrame:
        """Retrieve all overrides."""
        return self.get_overrides()

    def delete_override(self, override_id: str) -> bool:
        """Remove an override by ID."""
        if self._conn:
            self._conn.execute(
                "DELETE FROM transition_overrides WHERE override_id = ?",
                [override_id],
            )
            return True
        return False

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()


# ── Parquet-based fallback (Fabric-compatible) ────────────────────────────────

_OVERRIDE_SCHEMA = {
    "override_id": pl.Utf8,
    "old_sku": pl.Utf8,
    "new_sku": pl.Utf8,
    "effective_date": pl.Utf8,
    "scenario": pl.Utf8,
    "proportion": pl.Float64,
    "ramp_shape": pl.Utf8,
    "created_by": pl.Utf8,
    "created_at": pl.Utf8,
    "notes": pl.Utf8,
    "status": pl.Utf8,
    "approved_by": pl.Utf8,
    "approved_at": pl.Utf8,
}


class ParquetOverrideStore:
    """
    Parquet-backed override store for environments where DuckDB is unavailable.

    Drop-in replacement for ``OverrideStore`` using Polars read/write Parquet.
    Suitable for Microsoft Fabric notebooks and other constrained environments.
    """

    def __init__(self, parquet_path: str = "data/overrides.parquet"):
        self._path = Path(parquet_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> pl.DataFrame:
        """Read the Parquet file, returning an empty schema-correct DF if missing."""
        if self._path.exists():
            return pl.read_parquet(str(self._path))
        return pl.DataFrame(schema=_OVERRIDE_SCHEMA)

    def _write(self, df: pl.DataFrame) -> None:
        """Write the full DataFrame back to Parquet."""
        df.write_parquet(str(self._path))

    def add_override(
        self,
        old_sku: str,
        new_sku: str,
        proportion: float,
        scenario: str = "manual",
        ramp_shape: str = "linear",
        effective_date: Optional[str] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
        approval_threshold: float = 0.0,
    ) -> str:
        """Add an override. Returns the generated override_id."""
        override_id = f"OVR-{uuid.uuid4().hex[:8].upper()}"

        if approval_threshold > 0.0 and proportion > approval_threshold:
            status = "pending_approval"
        else:
            status = "approved"

        record = pl.DataFrame({
            "override_id": [override_id],
            "old_sku": [old_sku],
            "new_sku": [new_sku],
            "effective_date": [effective_date],
            "scenario": [scenario],
            "proportion": [proportion],
            "ramp_shape": [ramp_shape],
            "created_by": [created_by],
            "created_at": [datetime.now().isoformat()],
            "notes": [notes],
            "status": [status],
            "approved_by": [None],
            "approved_at": [None],
        })

        existing = self._read()
        combined = pl.concat([existing, record], how="vertical_relaxed")
        self._write(combined)
        return override_id

    def get_overrides(
        self,
        old_sku: Optional[str] = None,
        new_sku: Optional[str] = None,
    ) -> pl.DataFrame:
        """Retrieve overrides, optionally filtered by SKU."""
        df = self._read()
        if old_sku:
            df = df.filter(pl.col("old_sku") == old_sku)
        if new_sku:
            df = df.filter(pl.col("new_sku") == new_sku)
        return df

    def get_all(self) -> pl.DataFrame:
        """Retrieve all overrides."""
        return self.get_overrides()

    def delete_override(self, override_id: str) -> bool:
        """Remove an override by ID."""
        df = self._read()
        before = len(df)
        df = df.filter(pl.col("override_id") != override_id)
        if len(df) < before:
            self._write(df)
            return True
        return False

    def close(self) -> None:
        """No-op for Parquet backend (no persistent connection)."""
        pass


# ── Factory function ──────────────────────────────────────────────────────────

def get_override_store(
    path: str = "data/overrides",
    backend: str = "auto",
) -> "OverrideStore | ParquetOverrideStore":
    """
    Create an override store with the best available backend.

    Parameters
    ----------
    path:
        Base path for the store (extension is appended automatically).
    backend:
        ``"auto"`` — DuckDB if available, else Parquet.
        ``"duckdb"`` — Force DuckDB (raises ImportError if unavailable).
        ``"parquet"`` — Force Parquet backend.

    Returns
    -------
    ``OverrideStore`` (DuckDB) or ``ParquetOverrideStore``.
    """
    if backend == "parquet":
        return ParquetOverrideStore(f"{path}.parquet")
    elif backend == "duckdb":
        if not _HAS_DUCKDB:
            raise ImportError("DuckDB is required but not installed.")
        return OverrideStore(f"{path}.duckdb")
    else:  # auto
        if _HAS_DUCKDB:
            return OverrideStore(f"{path}.duckdb")
        return ParquetOverrideStore(f"{path}.parquet")

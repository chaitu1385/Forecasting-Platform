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

import errno
import math
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from io import BytesIO
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
_PATCH_FIELDS = frozenset({"proportion", "scenario", "ramp_shape", "effective_date", "notes"})
_FILE_MUTEXES: dict[str, threading.Lock] = {}
_FILE_MUTEXES_GUARD = threading.Lock()


class OverrideValidationError(ValueError):
    """An override mutation contains invalid or noneditable fields."""


class OverrideDataError(RuntimeError):
    """Stored overrides cannot be safely interpreted or uniquely identified."""


class OverrideStoreBusy(RuntimeError):
    """Another writer owns this local override file; retry explicitly."""


def _validate_iso_date(value: object) -> None:
    if not isinstance(value, str):
        raise OverrideValidationError("effective_date must be an ISO date (YYYY-MM-DD) or null.")
    try:
        valid = date.fromisoformat(value).isoformat() == value
    except ValueError:
        valid = False
    if not valid:
        raise OverrideValidationError("effective_date must be an ISO date (YYYY-MM-DD) or null.")


def _validated_changes(changes: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(changes, Mapping) or not changes:
        raise OverrideValidationError("Supply at least one editable override field.")
    if changes.keys() - _PATCH_FIELDS:
        raise OverrideValidationError(
            "Only proportion, scenario, ramp_shape, effective_date and notes are editable."
        )
    patch = dict(changes)
    for field, value in patch.items():
        if field == "proportion":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
                or not math.isfinite(value)
            ):
                raise OverrideValidationError("proportion must be a finite number in [0, 1].")
        elif field in {"scenario", "ramp_shape"}:
            if not isinstance(value, str):
                raise OverrideValidationError(f"{field} must be a string, not null.")
        elif field == "effective_date":
            if value is not None:
                _validate_iso_date(value)
        elif value is not None and not isinstance(value, str):
            raise OverrideValidationError("notes must be a string or null.")
    return patch


def _validate_override_frame(frame: pl.DataFrame) -> None:
    if set(frame.columns) != set(_OVERRIDE_SCHEMA):
        raise OverrideDataError("Stored overrides have an invalid set of columns.")
    ids = frame["override_id"]
    if ids.null_count() or ids.n_unique() != frame.height:
        raise OverrideDataError("Stored override IDs are null or duplicated.")
    for row in frame.iter_rows(named=True):
        for field in ("override_id", "old_sku", "new_sku", "status"):
            if not isinstance(row[field], str) or (field == "override_id" and not row[field]):
                raise OverrideDataError(f"Stored override {field} is invalid.")
        for field in ("created_by", "approved_by"):
            if row[field] is not None and not isinstance(row[field], str):
                raise OverrideDataError(f"Stored override {field} is invalid.")
        for field in ("created_at", "approved_at"):
            value = row[field]
            if value is None and field == "approved_at":
                continue
            if isinstance(value, datetime):
                continue
            try:
                if not isinstance(value, str):
                    raise ValueError
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise OverrideDataError(f"Stored override {field} is invalid.") from exc
        editable = {field: row[field] for field in _PATCH_FIELDS}
        # DuckDB DATE columns arrive as dates; file-backed stores retain ISO text.
        if type(editable["effective_date"]) is date:
            editable["effective_date"] = editable["effective_date"].isoformat()
        try:
            _validated_changes(editable)
        except OverrideValidationError as exc:
            raise OverrideDataError("Stored override editable fields are invalid.") from exc


def _duckdb_frame(result) -> pl.DataFrame:
    # DuckDB 2 removed .pl() and the deprecated fetch_arrow_table name.
    if hasattr(result, "to_arrow_table"):
        return pl.from_arrow(result.to_arrow_table())
    return pl.from_arrow(result.fetch_arrow_table())


@contextmanager
def _duckdb_transaction(connection) -> Iterator[None]:
    connection.execute("BEGIN TRANSACTION")
    try:
        yield
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


@contextmanager
def _override_file_lock(path: Path) -> Iterator[None]:
    """Lock a store's canonical path without unlinking the sidecar inode."""
    key = os.path.normcase(str(path))
    with _FILE_MUTEXES_GUARD:
        mutex = _FILE_MUTEXES.setdefault(key, threading.Lock())
    if not mutex.acquire(blocking=False):
        raise OverrideStoreBusy("Override store is busy; retry the mutation explicitly.")
    try:
        descriptor = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+b") as handle:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise OverrideStoreBusy(
                        "Override store is busy; retry the mutation explicitly."
                    ) from exc
                raise
            try:
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        mutex.release()


def _read_override_file(path: Path) -> pl.DataFrame:
    """Read under the caller's lock without retaining a mapped file handle."""
    if not path.exists():
        return pl.DataFrame(schema=_OVERRIDE_SCHEMA)
    source = BytesIO(path.read_bytes())
    if path.suffix == ".parquet":
        frame = pl.read_parquet(source)
    else:
        frame = pl.read_csv(source, schema_overrides=_OVERRIDE_SCHEMA)
    _validate_override_frame(frame)
    return frame


def _write_override_frame_atomically(frame: pl.DataFrame, path: Path) -> None:
    """Replace a mutable local store only after serialization and file flush."""
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            if path.suffix == ".parquet":
                frame.write_parquet(handle)
            else:
                frame.write_csv(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _patched_frame(
    frame: pl.DataFrame, override_id: str, changes: Mapping[str, object],
) -> Optional[pl.DataFrame]:
    if not frame["override_id"].eq(override_id).any():
        return None
    return frame.with_columns([
        pl.when(pl.col("override_id") == override_id)
        .then(pl.lit(value, dtype=_OVERRIDE_SCHEMA[field]))
        .otherwise(pl.col(field))
        .alias(field)
        for field, value in changes.items()
    ])


def _new_record(
    override_id: str,
    old_sku: str,
    new_sku: str,
    proportion: float,
    scenario: str,
    ramp_shape: str,
    effective_date: Optional[str],
    created_by: Optional[str],
    notes: Optional[str],
    approval_threshold: float,
) -> dict[str, object]:
    editable = _validated_changes({
        "proportion": proportion, "scenario": scenario, "ramp_shape": ramp_shape,
        "effective_date": effective_date, "notes": notes,
    })
    record = {
        "override_id": override_id,
        "old_sku": old_sku, "new_sku": new_sku, **editable,
        "created_by": created_by, "created_at": datetime.now().isoformat(),
        "status": (
            "pending_approval" if approval_threshold > 0.0 and proportion > approval_threshold
            else "approved"
        ),
        "approved_by": None, "approved_at": None,
    }
    _validate_override_frame(pl.DataFrame([record], schema=_OVERRIDE_SCHEMA))
    return record


class OverrideStore:
    """
    Persistent store for planner transition overrides.

    If DuckDB is available, uses a file-based DuckDB database.
    Otherwise, falls back to reading/writing a CSV file.
    """

    def __init__(self, db_path: str = "data/overrides.duckdb"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._use_duckdb = _HAS_DUCKDB
        self._conn = None
        self._csv_path = self.db_path.with_suffix(".csv")

    def _connection(self, *, create: bool = False):
        # Missing-ID reads/mutations must not create an empty database.
        if self._conn is None and (create or self.db_path.exists()):
            connection = duckdb.connect(str(self.db_path))
            try:
                connection.execute(_CREATE_TABLE_SQL)
            except BaseException:
                connection.close()
                raise
            self._conn = connection
        return self._conn

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
        record = _new_record(
            override_id, old_sku, new_sku, proportion, scenario, ramp_shape, effective_date,
            created_by, notes, approval_threshold,
        )
        if self._use_duckdb:
            connection = self._connection(create=True)
            connection.execute(
                """
                INSERT INTO transition_overrides
                (override_id, old_sku, new_sku, effective_date, scenario,
                 proportion, ramp_shape, created_by, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    override_id, old_sku, new_sku, effective_date,
                    scenario, proportion, ramp_shape, created_by, notes,
                    record["status"],
                ],
            )
        else:
            with _override_file_lock(self._csv_path):
                existing = _read_override_file(self._csv_path)
                combined = pl.concat(
                    [existing, pl.DataFrame([record], schema=_OVERRIDE_SCHEMA)],
                    how="vertical_relaxed",
                )
                _validate_override_frame(combined)
                _write_override_frame_atomically(combined, self._csv_path)

        return override_id

    def get_overrides(
        self,
        old_sku: Optional[str] = None,
        new_sku: Optional[str] = None,
    ) -> pl.DataFrame:
        """Retrieve overrides, optionally filtered by SKU."""
        if self._use_duckdb:
            connection = self._connection()
            if connection is None:
                return pl.DataFrame(schema=_OVERRIDE_SCHEMA)
            query = "SELECT * FROM transition_overrides WHERE 1=1"
            params = []
            if old_sku:
                query += " AND old_sku = ?"
                params.append(old_sku)
            if new_sku:
                query += " AND new_sku = ?"
                params.append(new_sku)
            return _duckdb_frame(connection.execute(query, params))
        else:
            with _override_file_lock(self._csv_path):
                df = _read_override_file(self._csv_path)
            if old_sku:
                df = df.filter(pl.col("old_sku") == old_sku)
            if new_sku:
                df = df.filter(pl.col("new_sku") == new_sku)
            return df

    def get_all(self) -> pl.DataFrame:
        """Retrieve all overrides."""
        return self.get_overrides()

    def update_override(self, override_id: str, changes: Mapping[str, object]) -> bool:
        """Patch one existing record without changing identity or approval metadata.

        Omitted fields are retained. Only ``effective_date`` and ``notes`` may
        be cleared with ``None``. A missing ID returns ``False``.
        """
        patch = _validated_changes(changes)
        if self._use_duckdb:
            connection = self._connection()
            if connection is None:
                return False
            with _duckdb_transaction(connection):
                target = _duckdb_frame(connection.execute(
                    "SELECT * FROM transition_overrides WHERE override_id = ?", [override_id],
                ))
                _validate_override_frame(target)
                if target.is_empty():
                    return False
                assignments = ", ".join(f"{field} = ?" for field in patch)
                returned = connection.execute(
                    f"UPDATE transition_overrides SET {assignments} "
                    "WHERE override_id = ? RETURNING override_id",
                    [*patch.values(), override_id],
                ).fetchall()
                if returned != [(override_id,)]:
                    raise OverrideDataError("Override update did not identify exactly one record.")
            return True
        with _override_file_lock(self._csv_path):
            updated = _patched_frame(_read_override_file(self._csv_path), override_id, patch)
            if updated is None:
                return False
            _write_override_frame_atomically(updated, self._csv_path)
        return True

    def delete_override(self, override_id: str) -> bool:
        """Remove an override by ID."""
        if self._use_duckdb:
            connection = self._connection()
            if connection is None:
                return False
            with _duckdb_transaction(connection):
                target = _duckdb_frame(connection.execute(
                    "SELECT * FROM transition_overrides WHERE override_id = ?", [override_id],
                ))
                _validate_override_frame(target)
                if target.is_empty():
                    return False
                returned = connection.execute(
                    "DELETE FROM transition_overrides WHERE override_id = ? RETURNING override_id",
                    [override_id],
                ).fetchall()
                if returned != [(override_id,)]:
                    raise OverrideDataError("Override deletion did not identify exactly one record.")
            return True
        with _override_file_lock(self._csv_path):
            frame = _read_override_file(self._csv_path)
            retained = frame.filter(pl.col("override_id") != override_id)
            if retained.height == frame.height:
                return False
            _write_override_frame_atomically(retained, self._csv_path)
        return True

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()


# ── Parquet-based fallback (Fabric-compatible) ────────────────────────────────

class ParquetOverrideStore:
    """
    Parquet-backed override store for environments where DuckDB is unavailable.

    Drop-in replacement for ``OverrideStore`` using Polars read/write Parquet.
    Mutations require a same-host local filesystem, not OneLake/network locking.
    """

    def __init__(self, parquet_path: str = "data/overrides.parquet"):
        self._path = Path(parquet_path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> pl.DataFrame:
        """Read the Parquet file, returning an empty schema-correct DF if missing."""
        return _read_override_file(self._path)

    def _write(self, df: pl.DataFrame) -> None:
        """Replace the full Parquet file; callers hold the mutation lock."""
        _write_override_frame_atomically(df, self._path)

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
        record = _new_record(
            override_id, old_sku, new_sku, proportion, scenario, ramp_shape, effective_date,
            created_by, notes, approval_threshold,
        )
        with _override_file_lock(self._path):
            existing = self._read()
            combined = pl.concat(
                [existing, pl.DataFrame([record], schema=_OVERRIDE_SCHEMA)],
                how="vertical_relaxed",
            )
            _validate_override_frame(combined)
            self._write(combined)
        return override_id

    def get_overrides(
        self,
        old_sku: Optional[str] = None,
        new_sku: Optional[str] = None,
    ) -> pl.DataFrame:
        """Retrieve overrides, optionally filtered by SKU."""
        with _override_file_lock(self._path):
            df = self._read()
        if old_sku:
            df = df.filter(pl.col("old_sku") == old_sku)
        if new_sku:
            df = df.filter(pl.col("new_sku") == new_sku)
        return df

    def get_all(self) -> pl.DataFrame:
        """Retrieve all overrides."""
        return self.get_overrides()

    def update_override(self, override_id: str, changes: Mapping[str, object]) -> bool:
        """Patch one record under the shared add/update/delete writer lock."""
        patch = _validated_changes(changes)
        with _override_file_lock(self._path):
            updated = _patched_frame(self._read(), override_id, patch)
            if updated is None:
                return False
            self._write(updated)
        return True

    def delete_override(self, override_id: str) -> bool:
        """Remove an override by ID."""
        with _override_file_lock(self._path):
            df = self._read()
            retained = df.filter(pl.col("override_id") != override_id)
            if retained.height == df.height:
                return False
            self._write(retained)
        return True

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

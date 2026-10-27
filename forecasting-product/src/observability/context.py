"""
PipelineContext — correlation ID threading for pipeline observability.

Every pipeline run creates a ``PipelineContext`` that carries a ``run_id``
through all modules.  Sub-pipelines (e.g. per-model backtest) get child
contexts whose ``parent_run_id`` links back to the root run.

Usage
-----
>>> ctx = PipelineContext(lob="rossmann")
>>> print(ctx.run_id)       # e.g. "a1b2c3d4e5f67890"
>>> child = ctx.child("backtest")
>>> print(child.run_id)     # e.g. "a1b2c3d4e5f67890-backtest"
>>> print(child.parent_run_id)  # "a1b2c3d4e5f67890"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4


def new_run_id() -> str:
    """Generate an independent server attempt identity."""
    return uuid4().hex


@dataclass
class PipelineContext:
    """
    Immutable context for a single pipeline run.

    Attributes
    ----------
    run_id:
        Unique identifier for this run (auto-generated if not provided).
    lob:
        Line-of-business identifier.
    started_at:
        UTC timestamp when the run started.
    parent_run_id:
        If this is a child context, the parent's run_id.
    tags:
        Arbitrary key-value metadata (model name, config hash, etc.).
    """

    run_id: str = field(default_factory=new_run_id)
    lob: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    parent_run_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def child(self, suffix: str) -> "PipelineContext":
        """
        Create a child context for a sub-pipeline.

        The child inherits ``lob`` and ``tags``, gets a derived ``run_id``,
        and records this context as ``parent_run_id``.

        Parameters
        ----------
        suffix:
            Short label appended to the run_id (e.g. ``"backtest"``,
            ``"fold-0"``, ``"lgbm_direct"``).
        """
        return PipelineContext(
            run_id=f"{self.run_id}-{suffix}",
            lob=self.lob,
            parent_run_id=self.run_id,
            tags=dict(self.tags),
        )

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since ``started_at``."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    def as_dict(self) -> Dict[str, str]:
        """Flat dict suitable for structured log records."""
        return {
            "run_id": self.run_id,
            "lob": self.lob,
            "parent_run_id": self.parent_run_id or "",
            "started_at": self.started_at.isoformat(),
            **self.tags,
        }

    def __str__(self) -> str:
        parent = f" parent={self.parent_run_id}" if self.parent_run_id else ""
        return f"PipelineContext(run_id={self.run_id}, lob={self.lob}{parent})"

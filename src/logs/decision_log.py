"""
Decision log spine. Every layer writes its state here and the regression model
reads from here at feature-build time.

Storage is SQLite for simplicity and zero-dependency operation. Schema:

  setups(
    setup_id PRIMARY KEY,
    symbol, timestamp, status,
    -- one column per layer, holding the layer's payload as JSON text
    scanner, portfolio, filter, bias, entry_state, entry_trigger,
    features, regression, claude, sizing, order, outcome
  )

Updates are upserts keyed by setup_id. status tracks the terminal outcome
(or 'in_progress' until the trade closes).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


LAYER_COLUMNS = [
    "scanner", "portfolio", "filter", "bias",
    "entry_state", "entry_trigger",
    "features", "regression", "claude",
    "sizing", "order", "outcome",
]


# quote column names so SQL reserved words (e.g. "order") don't break the schema.
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS setups (
    setup_id   TEXT PRIMARY KEY,
    symbol     TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'in_progress',
    {",".join(f'"{c}" TEXT' for c in LAYER_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS idx_setups_symbol  ON setups(symbol);
CREATE INDEX IF NOT EXISTS idx_setups_status  ON setups(status);
CREATE INDEX IF NOT EXISTS idx_setups_ts      ON setups(timestamp);
"""


class DecisionLog:
    """The append/upsert log every layer writes to."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # one persistent connection. SQLite is happy with this for a single
        # process; multi-process consumers should open their own.
        self._c = sqlite3.connect(db_path, isolation_level=None)  # autocommit
        self._c.execute("PRAGMA journal_mode=WAL")
        self._c.execute("PRAGMA synchronous=NORMAL")
        self._c.executescript(SCHEMA)

    def close(self) -> None:
        try:
            self._c.close()
        except sqlite3.Error:
            pass

    def __del__(self):
        self.close()

    # -- writes --------------------------------------------------------------

    def new_setup(self, symbol: str, timestamp: str) -> str:
        setup_id = str(uuid.uuid4())[:12]
        self._c.execute(
            "INSERT INTO setups(setup_id, symbol, timestamp) VALUES (?, ?, ?)",
            (setup_id, symbol, timestamp),
        )
        return setup_id

    def write(self, setup_id: str, layer: str, payload: dict[str, Any]) -> None:
        if layer not in LAYER_COLUMNS:
            raise ValueError(f"unknown log layer: {layer}")
        self._c.execute(
            f'UPDATE setups SET "{layer}" = ? WHERE setup_id = ?',
            (json.dumps(payload, default=str), setup_id),
        )

    def set_status(self, setup_id: str, status: str) -> None:
        self._c.execute(
            "UPDATE setups SET status = ? WHERE setup_id = ?",
            (status, setup_id),
        )

    def write_outcome(self, setup_id: str, realized_r: float, exit_reason: str,
                  holding_bars: int, mfe_r: float = 0.0, mae_r: float = 0.0) -> None:
        payload = {
            "realized_r": realized_r,
            "exit_reason": exit_reason,
            "holding_bars": holding_bars,
            "mfe_r": float(mfe_r),
            "mae_r": float(mae_r),
        }
        self.write(setup_id, "outcome", payload)
        self.set_status(setup_id, "closed")

    # -- reads ---------------------------------------------------------------

    def get(self, setup_id: str) -> dict[str, Any] | None:
        row = self._c.execute("SELECT * FROM setups WHERE setup_id = ?", (setup_id,)).fetchone()
        if row is None:
            return None
        cols = [d[1] for d in self._c.execute("PRAGMA table_info(setups)").fetchall()]
        return self._row_to_dict(cols, row)

    def all_setups(self) -> list[dict[str, Any]]:
        cols = [d[1] for d in self._c.execute("PRAGMA table_info(setups)").fetchall()]
        rows = self._c.execute("SELECT * FROM setups").fetchall()
        return [self._row_to_dict(cols, r) for r in rows]

    def closed_setups_with_features(self) -> list[dict[str, Any]]:
        """Return only setups that have features and a final outcome -- the regression's training set."""
        out = []
        for s in self.all_setups():
            if s.get("features") and s.get("outcome"):
                out.append(s)
        return out

    @staticmethod
    def _row_to_dict(cols: list[str], row: tuple) -> dict[str, Any]:
        d = dict(zip(cols, row))
        for k in LAYER_COLUMNS:
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

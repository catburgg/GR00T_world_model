from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DummyReplayPolicyAdapter:
    """WebSocket server 需要的接口：`infer(obs_list) -> Dict[str, np.ndarray]`。"""

    ACTION_COLUMN = "action"

    def __init__(
        self,
        parquet_path: str,
        modality_json_path: str,
        *,
        action_horizon: int = 1,
        start_index: int = 0,
    ) -> None:
        self.parquet_path = Path(parquet_path)

        self.df = pd.read_parquet(self.parquet_path)

        with open(modality_json_path, "r") as f:
            modality = json.load(f)
        self.action_slices: dict[str, tuple[int, int]] = {
            f"action.{k}": (int(v["start"]), int(v["end"])) for k, v in modality["action"].items()
        }

        self.action_horizon = int(action_horizon)
        self.current_idx = int(start_index)
        self._action_dim = int(self._row_action_vec(0).shape[0])

    def _row_action_vec(self, row_idx: int) -> np.ndarray:
        raw: Any = self.df[self.ACTION_COLUMN].iloc[row_idx]
        vec = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
        return vec.astype(np.float32, copy=False)

    def _stack_horizon(self) -> np.ndarray:
        n = len(self.df)
        H = self.action_horizon

        rows: list[np.ndarray] = []
        idx = self.current_idx
        for _ in range(H):
            if idx >= n:
                rows.append(rows[-1].copy() if rows else self._row_action_vec(n - 1))
                continue
            rows.append(self._row_action_vec(idx))
            idx += 1

        mat = np.stack(rows, axis=0)
        return mat

    def infer(self, obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        action_mat = self._stack_horizon()  # (1, D)

        out: dict[str, np.ndarray] = {}
        for k, (s, e) in self.action_slices.items():
            out[k] = action_mat[:, s:e]

        self.current_idx += 1
        return out

    def reset(self) -> None:
        self.current_idx = 0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Structured episode logger for Sentinel Garden v2.

Logs complete episode trajectories to JSON files for debugging, auditing,
human review, and replay. Each episode gets its own log file.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EpisodeLogger:
    """Log episode trajectories to structured JSON files.

    Args:
        log_dir: Directory to store episode logs.
        enabled: If False, all logging operations are no-ops.
        export_format: "json" or "prometheus" metrics format.
    """

    def __init__(
        self,
        log_dir: str = "./logs/episodes",
        enabled: bool = True,
        export_format: str = "json",
    ) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.export_format = export_format

        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Episode logging enabled: %s", self.log_dir)

        self._current_episode: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(
        self,
        episode_id: Optional[str] = None,
        task_type: str = "guardrail_enforcer",
        session_id: Optional[Any] = None,
        scoring_mode: str = "prelabeled",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start logging a new episode.

        Returns:
            The episode ID.
        """
        if not self.enabled:
            return episode_id or str(uuid.uuid4())

        episode_id = episode_id or str(uuid.uuid4())
        self._current_episode = {
            "episode_id": episode_id,
            "task_type": task_type,
            "session_id": session_id,
            "scoring_mode": scoring_mode,
            "start_time": time.time(),
            "end_time": None,
            "steps": [],
            "total_reward": 0.0,
            "final_score": None,
            "metadata": metadata or {},
        }
        return episode_id

    def log_step(
        self,
        step: int,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        rewards: Dict[str, float],
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log a single step in the current episode."""
        if not self.enabled or self._current_episode is None:
            return

        step_data = {
            "step": step,
            "timestamp": time.time(),
            "observation": observation,
            "action": action,
            "rewards": rewards,
            "latency_ms": latency_ms,
        }
        self._current_episode["steps"].append(step_data)
        self._current_episode["total_reward"] += rewards.get("total", 0.0)

    def end_episode(self, final_score: Optional[float] = None) -> Optional[Path]:
        """Finalize the current episode and write to disk.

        Returns:
            Path to the written log file, or None if logging is disabled.
        """
        if not self.enabled or self._current_episode is None:
            return None

        self._current_episode["end_time"] = time.time()
        self._current_episode["final_score"] = final_score

        # Calculate duration
        start = self._current_episode["start_time"]
        end = self._current_episode["end_time"]
        self._current_episode["duration_seconds"] = end - start

        # Write to file
        episode_id = self._current_episode["episode_id"]
        filename = f"{episode_id}.json"
        filepath = self.log_dir / filename

        filepath.write_text(
            json.dumps(self._current_episode, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.debug("Episode log written: %s", filepath)
        self._current_episode = None
        return filepath

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_episodes(
        self,
        task_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List recent episode logs with optional filtering.

        Returns:
            List of episode metadata dicts (not full trajectories).
        """
        if not self.log_dir.exists():
            return []

        episodes = []
        for filepath in sorted(self.log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if len(episodes) >= limit:
                break
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                meta = {
                    "episode_id": data.get("episode_id"),
                    "task_type": data.get("task_type"),
                    "session_id": data.get("session_id"),
                    "scoring_mode": data.get("scoring_mode"),
                    "start_time": data.get("start_time"),
                    "duration_seconds": data.get("duration_seconds"),
                    "total_reward": data.get("total_reward"),
                    "final_score": data.get("final_score"),
                    "num_steps": len(data.get("steps", [])),
                }
                if task_type is None or meta["task_type"] == task_type:
                    episodes.append(meta)
            except Exception as exc:
                logger.warning("Failed to read episode log %s: %s", filepath, exc)

        return episodes

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a full episode log by ID."""
        filepath = self.log_dir / f"{episode_id}.json"
        if not filepath.exists():
            return None
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read episode %s: %s", episode_id, exc)
            return None

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """Return True if the log directory is writable."""
        if not self.enabled:
            return True
        try:
            test_file = self.log_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all logged episodes."""
        episodes = self.list_episodes(limit=10000)
        if not episodes:
            return {}

        rewards = [e["total_reward"] for e in episodes if e["total_reward"] is not None]
        scores = [e["final_score"] for e in episodes if e["final_score"] is not None]

        return {
            "total_episodes": len(episodes),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "by_task": {},
        }

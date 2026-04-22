# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JSON file-backed session store.

Loads sessions from a JSON file on disk. Supports hot-reloading for
development and simple deployments.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from models import Session
from session_store.base import SessionFilter, SessionStore

logger = logging.getLogger(__name__)


class JSONSessionStore(SessionStore):
    """Load sessions from a JSON file.

    Args:
        path: Path to the JSON file containing sessions.
        validate: If True, validate sessions against the Session schema on load.
        hot_reload: If True, check for file changes on each access.
    """

    def __init__(
        self,
        path: str = "data/sessions.json",
        validate: bool = True,
        hot_reload: bool = False,
    ) -> None:
        self.path = Path(path)
        self._validate = validate
        self._hot_reload = hot_reload
        self._sessions: List[Session] = []
        self._last_mtime: float = 0.0
        self._load()

    def _load(self) -> None:
        """Load sessions from the JSON file."""
        if not self.path.exists():
            # Try resolving relative to project root
            env_path = os.getenv("SESSIONS_PATH")
            if env_path:
                self.path = Path(env_path)
            else:
                logger.error("Session file not found: %s", self.path)
                self._sessions = []
                return

        try:
            mtime = self.path.stat().st_mtime
            if mtime == self._last_mtime and self._sessions:
                return  # No changes

            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.error("Expected JSON list in %s, got %s", self.path, type(data).__name__)
                self._sessions = []
                return

            sessions = []
            for i, item in enumerate(data):
                try:
                    if self._validate:
                        sessions.append(Session.model_validate(item))
                    else:
                        sessions.append(Session(**item))
                except Exception as exc:
                    logger.warning("Skipping invalid session at index %d: %s", i, exc)

            self._sessions = sessions
            self._last_mtime = mtime
            logger.info(
                "Loaded %d sessions from %s (%d skipped)",
                len(sessions),
                self.path,
                len(data) - len(sessions),
            )

        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", self.path, exc)
            self._sessions = []
        except Exception as exc:
            logger.error("Failed to load sessions from %s: %s", self.path, exc)
            self._sessions = []

    def reload(self) -> None:
        """Force reload from disk."""
        self._last_mtime = 0.0
        self._load()

    def _maybe_reload(self) -> None:
        if self._hot_reload:
            self._load()

    # ------------------------------------------------------------------
    # SessionStore interface
    # ------------------------------------------------------------------

    def load_sessions(self, filters: Optional[SessionFilter] = None) -> List[Session]:
        self._maybe_reload()
        sessions = self._sessions

        if filters is None:
            return list(sessions)

        result = []
        for s in sessions:
            if filters.task_type and s.task_type != filters.task_type:
                continue
            if filters.min_turns is not None and len(s.labels) < filters.min_turns:
                continue
            if filters.max_turns is not None and len(s.labels) > filters.max_turns:
                continue
            if filters.session_ids is not None and s.id not in filters.session_ids:
                continue
            result.append(s)

        return result

    def add_session(self, session: Session) -> None:
        self._maybe_reload()
        self._sessions.append(session)
        self._save()
        logger.info("Added session %s (total: %d)", session.id, len(self._sessions))

    def count(self, filters: Optional[SessionFilter] = None) -> int:
        return len(self.load_sessions(filters))

    def _save(self) -> None:
        """Persist sessions back to the JSON file."""
        data = [s.model_dump(mode="json") for s in self._sessions]
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_mtime = self.path.stat().st_mtime

    @property
    def healthy(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract base class for session store backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

from models import Session


@dataclass
class SessionFilter:
    """Filter criteria for loading sessions."""

    task_type: Optional[str] = None
    min_turns: Optional[int] = None
    max_turns: Optional[int] = None
    session_ids: Optional[List[Union[int, str]]] = None
    tags: Optional[List[str]] = None
    difficulty: Optional[str] = None  # easy | medium | hard


class SessionStore(ABC):
    """Abstract session data source.

    Implementations provide different backends for storing and retrieving
    conversation sessions with per-turn labels.
    """

    @abstractmethod
    def load_sessions(
        self,
        filters: Optional[SessionFilter] = None,
    ) -> List[Session]:
        """Load sessions with optional filtering.

        Args:
            filters: Optional filter criteria.

        Returns:
            List of matching Session objects.
        """

    @abstractmethod
    def add_session(self, session: Session) -> None:
        """Add a new session (for online learning / data augmentation).

        Args:
            session: The session to add.
        """

    @abstractmethod
    def count(self, filters: Optional[SessionFilter] = None) -> int:
        """Return the number of sessions matching the filters."""

    def reload(self) -> None:
        """Reload data from the backing store (no-op for most backends)."""

    @property
    @abstractmethod
    def healthy(self) -> bool:
        """Return True if the store is accessible and operational."""

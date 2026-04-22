# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sentinel Garden Environment v2 — OpenEnv-compatible RL environment.

Key improvements over v1:
    - Pluggable scoring engine (prelabeled / judge / hybrid modes)
    - Smooth continuous reward functions
    - Semantic nudge matching
    - Episode logging and replay
    - Dynamic task registration
    - Session store abstraction
    - Concurrent environment support
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    # Fallback for when openenv is not installed
    Environment = object  # type: ignore[misc,assignment]

from models import (
    SentinelGardenAction,
    SentinelGardenObservation,
    SentinelGardenState,
    TrajectoryStep,
    VALID_TASKS,
)
from scoring.engine import ScoringEngine
from scoring.judge_client import JudgeClient
from scoring.rewards import SmoothRewardFunction
from session_store import JSONSessionStore, SessionFilter, SessionStore
from task_registry import TaskRegistry
from episode_logging.episode_logger import EpisodeLogger

logger = logging.getLogger(__name__)


class SentinelGardenEnvironment(Environment):
    """OpenEnv environment for the Sentinel Garden Overseer task (v2).

    Supports three evaluation tasks and three scoring modes:
        Tasks:  guardrail_enforcer, prompt_compiler, jailbreak_detector
        Modes:  prelabeled (default), judge, hybrid

    Reward is decomposed into three components per step:
        action_reward  ~ [-2.0, +3.0]
        risk_reward    ~ [-0.75, +2.0]
        note_reward    ~ [-1.0, +1.0]

    Args:
        data: Optional pre-loaded session data.
        scoring_mode: "prelabeled", "judge", or "hybrid".
        session_store: Custom session store backend.
        task_registry: Custom task registry.
        scoring_engine: Custom scoring engine.
        episode_logger: Custom episode logger.
        config: Additional configuration dictionary.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        scoring_mode: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        task_registry: Optional[TaskRegistry] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        episode_logger: Optional[EpisodeLogger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}

        # Scoring mode
        self.scoring_mode = scoring_mode or os.getenv("SENTINEL_SCORING_MODE", "prelabeled")
        if self.scoring_mode not in ("prelabeled", "judge", "hybrid"):
            logger.warning("Invalid scoring mode %r, using prelabeled", self.scoring_mode)
            self.scoring_mode = "prelabeled"

        # Task registry
        self.task_registry = task_registry or TaskRegistry()

        # Session store
        if data is not None:
            # Backward compat: wrap raw data
            self.session_store = _InMemorySessionStore(data)
        elif session_store is not None:
            self.session_store = session_store
        else:
            sessions_path = self.config.get("data", {}).get("path", "data/sessions.json")
            self.session_store = JSONSessionStore(
                path=sessions_path,
                validate=True,
                hot_reload=False,
            )

        # Scoring engine
        if scoring_engine is not None:
            self.scoring = scoring_engine
        else:
            judge_config = self.config.get("scoring", {}).get("judge", {})
            judge_client = None
            if self.scoring_mode in ("judge", "hybrid"):
                judge_client = JudgeClient(
                    model=judge_config.get("model", "gpt-4o"),
                    api_base=judge_config.get("api_base"),
                    api_key=judge_config.get("api_key") or os.getenv("JUDGE_API_KEY"),
                    temperature=judge_config.get("temperature", 0.0),
                    max_tokens=judge_config.get("max_tokens", 512),
                    rate_limit_rps=judge_config.get("rate_limit_rps"),
                )

            prelabeled_cfg = self.config.get("scoring", {}).get("prelabeled", {})
            reward_fn = SmoothRewardFunction(
                semantic_model=prelabeled_cfg.get("semantic_model", "all-MiniLM-L6-v2"),
                similarity_threshold=prelabeled_cfg.get("similarity_threshold", 0.7),
            )

            self.scoring = ScoringEngine(
                mode=self.scoring_mode,
                reward_function=reward_fn,
                judge_client=judge_client,
                config=prelabeled_cfg,
            )

        # Episode logger
        log_cfg = self.config.get("logging", {})
        self.episode_logger = episode_logger or EpisodeLogger(
            log_dir=log_cfg.get("episode_log_dir", "./logs/episodes"),
            enabled=log_cfg.get("enabled", True),
            export_format=log_cfg.get("export_format", "json"),
        )

        # Episode state
        self._state = SentinelGardenState()
        self._pair_index: int = 1
        self._session: Optional[Dict[str, Any]] = None
        self._current_task: str = "guardrail_enforcer"

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        task_type: Optional[str] = None,
    ) -> SentinelGardenObservation:
        """Reset the environment and begin a new episode.

        Args:
            data: Optional override data for this episode.
            task_type: Optional task type override.

        Returns:
            Initial observation.
        """
        start_time = time.time()

        # Determine task type
        raw_task = task_type or os.getenv("SENTINEL_TASK", "guardrail_enforcer").strip()
        if raw_task not in self.task_registry.list_tasks():
            logger.warning("Unknown task %r, using guardrail_enforcer", raw_task)
            raw_task = "guardrail_enforcer"
        self._current_task = raw_task

        # Select session
        if data is not None:
            import random

            self._session = random.choice(data)
        else:
            sessions = self.session_store.load_sessions(
                filters=SessionFilter(task_type=self._current_task)
                if self.session_store.count() > 10
                else None
            )
            if not sessions:
                # Fallback: load all sessions
                sessions = self.session_store.load_sessions()
            if not sessions:
                raise RuntimeError("No sessions available in the session store")
            import random

            self._session = random.choice(sessions).model_dump()

        guardrail = self._session["chat_history"][0]["content"]
        chat_history = self._session["chat_history"]
        labels = self._session["labels"]

        self._pair_index = 1
        episode_id = str(self._session.get("id", uuid.uuid4()))

        self._state = SentinelGardenState(
            episode_id=episode_id,
            step_count=0,
            guardrail=guardrail,
            chat_history=chat_history,
            max_turn=len(labels),
            task_type=self._current_task,
        )

        # Start episode logging
        self.episode_logger.start_episode(
            episode_id=episode_id,
            task_type=self._current_task,
            session_id=self._session.get("id"),
            scoring_mode=self.scoring_mode,
        )

        obs = SentinelGardenObservation(
            done=False,
            reward=None,
            guardrail=guardrail,
            current_prompt=chat_history[self._pair_index]["content"],
            proposed_response=chat_history[self._pair_index + 1]["content"],
            risk_score=0.0,
            internal_note="",
            task_type=self._current_task,
        )

        logger.debug(
            "Reset episode=%s task=%s turns=%d (%.1fms)",
            episode_id,
            self._current_task,
            len(labels),
            (time.time() - start_time) * 1000,
        )
        return obs

    def step(self, action_obj: SentinelGardenAction) -> SentinelGardenObservation:
        """Execute one step in the environment.

        Args:
            action_obj: The overseer's action.

        Returns:
            Observation with reward and done flag.
        """
        step_start = time.time()
        target = self._state.chat_history["labels"][self._state.step_count]
        task_type = self._current_task

        # Advance counters
        self._state.step_count += 1
        self._pair_index += 2
        if target["action"] == "RE-PROMPT":
            self._pair_index += 2  # skip nudge+correction pair

        # Unpack action
        action = action_obj.action.strip().upper()
        internal_note = action_obj.internal_note.strip()
        risk_score = float(action_obj.risk_score)
        nudge = (action_obj.nudge or "").strip()
        target_action = target["action"].strip().upper()

        # Get current conversation context
        user_message = self._state.chat_history[self._pair_index - 1]["content"]
        model_response = self._state.chat_history[self._pair_index]["content"]
        guardrail = self._state.guardrail

        # Score the action
        if self.scoring_mode == "prelabeled":
            total_reward, components = self.scoring.score_prelabeled(
                predicted_action=action,
                target_action=target_action,
                predicted_risk=risk_score,
                target_risk=float(target.get("risk_score", 0.0)),
                predicted_note=internal_note,
                target_note=target.get("internal_note", ""),
                predicted_nudge=nudge,
                target_nudge=target.get("nudge", ""),
            )
            judge_details = None
        else:
            # Use sync version for OpenEnv compatibility
            total_reward, components, judge_details = self.scoring.score_judge_sync(
                task_type=task_type,
                guardrail=guardrail,
                user_message=user_message,
                model_response=model_response,
                overseer_action=action,
                overseer_risk_score=risk_score,
                overseer_note=internal_note,
                overseer_nudge=nudge,
            )

        action_reward = components["action_reward"]
        risk_reward = components["risk_reward"]
        note_reward = components["note_reward"]

        # Check termination
        done = self._state.step_count >= len(self._state.chat_history["labels"])

        if done:
            next_prompt = None
            next_resp = None
        else:
            next_prompt = self._state.chat_history[self._pair_index]["content"]
            next_resp = self._state.chat_history[self._pair_index + 1]["content"]

        # Build observation
        obs = SentinelGardenObservation(
            done=done,
            reward=total_reward,
            guardrail=guardrail,
            current_prompt=next_prompt,
            proposed_response=next_resp,
            risk_score=risk_score,
            internal_note=internal_note,
            task_type=task_type,
            action_reward=action_reward,
            risk_reward=risk_reward,
            note_reward=note_reward,
        )

        # Attach judge evaluation if available
        if judge_details:
            obs.judge_evaluation = judge_details

        latency_ms = (time.time() - step_start) * 1000

        # Log step
        self.episode_logger.log_step(
            step=self._state.step_count,
            observation={
                "current_prompt": next_prompt,
                "proposed_response": next_resp,
                "task_type": task_type,
            },
            action={
                "action": action,
                "risk_score": risk_score,
                "internal_note": internal_note,
                "nudge": nudge,
            },
            rewards={
                "action": action_reward,
                "risk": risk_reward,
                "note": note_reward,
                "total": total_reward,
            },
            latency_ms=latency_ms,
        )

        # End episode if done
        if done:
            self.episode_logger.end_episode(final_score=None)

        return obs

    @property
    def state(self) -> SentinelGardenState:
        return self._state

    # ------------------------------------------------------------------
    # Health and diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Deep health check: model loaded, data accessible, judge reachable."""
        result = {
            "status": "healthy",
            "scoring_mode": self.scoring_mode,
            "session_store_healthy": self.session_store.healthy,
            "episode_logger_healthy": self.episode_logger.healthy,
            "session_count": self.session_store.count(),
        }

        if self.scoring_mode in ("judge", "hybrid"):
            result["judge_available"] = (
                self.scoring.judge is not None and self.scoring.judge.available
            )

        if not result["session_store_healthy"]:
            result["status"] = "unhealthy"

        return result


# ---------------------------------------------------------------------------
# In-memory session store for backward compatibility
# ---------------------------------------------------------------------------


class _InMemorySessionStore(SessionStore):
    """Wraps raw session data for backward compatibility."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self._data = data

    def load_sessions(self, filters: Optional[SessionFilter] = None):
        from models import Session

        result = []
        for item in self._data:
            try:
                result.append(Session.model_validate(item))
            except Exception:
                result.append(Session(**item))
        return result

    def add_session(self, session):
        self._data.append(session.model_dump())

    def count(self, filters=None):
        return len(self._data)

    @property
    def healthy(self):
        return len(self._data) > 0

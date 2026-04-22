# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Sentinel Garden v2.

Pydantic v2 models with strict validation, schema evolution support,
and full backward compatibility with OpenEnv interfaces.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_GUARD = "guardrail_enforcer"
TASK_PROMPT = "prompt_compiler"
TASK_JAIL = "jailbreak_detector"

VALID_TASKS = {TASK_GUARD, TASK_PROMPT, TASK_JAIL}

ActionType = Literal["PASS", "RE-PROMPT", "BLOCK"]
ScoringMode = Literal["prelabeled", "judge", "hybrid"]
NudgeMatchMode = Literal["exact", "semantic"]
RiskFunctionMode = Literal["discrete", "smooth"]


# ---------------------------------------------------------------------------
# Session data models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., min_length=0, description="Message content")

    @field_validator("role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got {v!r}")
        return v


class TurnLabel(BaseModel):
    """Pre-labeled ground truth for a single conversation turn."""

    model_config = ConfigDict(frozen=True)

    action: ActionType = Field(..., description="Correct overseer action")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ground-truth risk score in [0.0, 1.0]",
    )
    internal_note: str = Field(default="", description="Ground-truth reasoning note")
    nudge: Optional[str] = Field(
        default=None,
        description="Re-prompt nudge text (only when action == RE-PROMPT)",
    )

    @model_validator(mode="after")
    def _check_nudge(self) -> "TurnLabel":
        if self.action == "RE-PROMPT" and not (self.nudge and self.nudge.strip()):
            raise ValueError("nudge is required when action == 'RE-PROMPT'")
        return self


class Session(BaseModel):
    """A complete conversation session with chat history and per-turn labels."""

    model_config = ConfigDict(frozen=False)

    id: Union[int, str] = Field(..., description="Unique session identifier")
    task_type: Optional[str] = Field(
        default=None,
        description="Optional task type override",
    )
    chat_history: List[ChatMessage] = Field(
        ...,
        min_length=2,
        description="Conversation messages",
    )
    labels: List[TurnLabel] = Field(
        ...,
        min_length=1,
        description="Per-turn ground-truth labels",
    )

    @model_validator(mode="after")
    def _validate_lengths(self) -> "Session":
        # Labels should match the number of (user, assistant) pairs
        expected = (len(self.chat_history) - 1) // 2
        actual = len(self.labels)
        if actual != expected:
            raise ValueError(
                f"Expected {expected} labels for {len(self.chat_history)} messages, "
                f"got {actual}"
            )
        return self

    @field_validator("task_type")
    @classmethod
    def _valid_task(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_TASKS:
            raise ValueError(f"task_type must be one of {VALID_TASKS}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Trajectory step (for episode logging / replay)
# ---------------------------------------------------------------------------

class TrajectoryStep(BaseModel):
    """A single step in the episode trajectory."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=0)
    timestamp: float = Field(..., description="Unix timestamp of the step")
    observation: Dict[str, Any] = Field(default_factory=dict)
    action: Dict[str, Any] = Field(default_factory=dict)
    rewards: Dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = Field(default=None, description="Step latency in milliseconds")


# ---------------------------------------------------------------------------
# Action / Observation / State — OpenEnv interface
# ---------------------------------------------------------------------------

class SentinelGardenAction(BaseModel):
    """Action produced by the Overseer agent at each step."""

    model_config = ConfigDict(extra="forbid")

    action: ActionType = Field(default="PASS", description="PASS | RE-PROMPT | BLOCK")
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Continuous risk estimate in [0.0, 1.0]",
    )
    internal_note: str = Field(
        default="",
        max_length=500,
        description="Concise reasoning, carried to next step",
    )
    nudge: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Re-prompt hint; only when action == RE-PROMPT",
    )

    @model_validator(mode="after")
    def _check_nudge(self) -> "SentinelGardenAction":
        if self.action == "RE-PROMPT" and not (self.nudge and self.nudge.strip()):
            # Auto-set a default nudge instead of failing
            self.nudge = "Please revise your response."
        if self.action != "RE-PROMPT":
            self.nudge = None
        return self


class SentinelGardenObservation(BaseModel):
    """Observation returned to the Overseer agent after each step."""

    model_config = ConfigDict(extra="forbid")

    done: bool = Field(default=False, description="Episode terminal flag")
    reward: Optional[float] = Field(default=None, description="Total step reward")

    guardrail: str = Field(default="", description="Active system guardrail text")
    current_prompt: Optional[str] = Field(
        default=None, description="Latest user message (None on terminal step)"
    )
    proposed_response: Optional[str] = Field(
        default=None, description="Model's draft response (None on terminal step)"
    )

    risk_score: float = Field(default=0.0, description="Risk score from previous step")
    internal_note: str = Field(default="", description="Note from previous step")
    task_type: str = Field(default=TASK_GUARD, description="Current evaluation task")

    # Decomposed reward components
    action_reward: Optional[float] = Field(default=None, description="Action correctness reward")
    risk_reward: Optional[float] = Field(default=None, description="Risk calibration reward")
    note_reward: Optional[float] = Field(default=None, description="Note quality reward")

    # Judge evaluation details (populated in judge/hybrid mode)
    judge_evaluation: Optional[JudgeEvaluation] = Field(
        default=None, description="Judge model evaluation (judge/hybrid mode only)"
    )

    @field_validator("task_type")
    @classmethod
    def _valid_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task_type must be one of {VALID_TASKS}, got {v!r}")
        return v


class SentinelGardenState(BaseModel):
    """Hidden server-side state for the current episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: Optional[str] = Field(default=None, description="Unique episode identifier")
    step_count: int = Field(default=0, ge=0, description="Current step within episode")
    guardrail: str = Field(default="", description="Active guardrail text")
    chat_history: List[Dict[str, str]] = Field(
        default_factory=list, description="Full conversation history"
    )
    current_risk_score: float = Field(default=0.0, description="Last submitted risk score")
    current_note: str = Field(default="", description="Last submitted internal note")
    max_turn: int = Field(default=0, description="Total turns in this episode")
    task_type: str = Field(default=TASK_GUARD, description="Active task type")

    # Episode trajectory (for logging / replay)
    trajectory: List[TrajectoryStep] = Field(
        default_factory=list, description="Full episode trajectory"
    )

    @field_validator("task_type")
    @classmethod
    def _valid_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task_type must be one of {VALID_TASKS}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Judge evaluation model
# ---------------------------------------------------------------------------

class JudgeEvaluation(BaseModel):
    """Structured evaluation returned by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    correct_action: ActionType = Field(..., description="Judge's assessment of correct action")
    correct_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's assessment of appropriate risk score",
    )
    action_correct: bool = Field(..., description="Whether overseer chose correct action")
    risk_score_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How accurate the risk score was (1.0 = perfect)",
    )
    note_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score for the internal_note",
    )
    reasoning: str = Field(default="", description="Judge's brief explanation")


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

class TaskWeights(BaseModel):
    """Per-task scoring weights."""

    model_config = ConfigDict(frozen=True)

    action: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.3, ge=0.0, le=1.0)
    note: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "TaskWeights":
        total = self.action + self.risk + self.note
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return self


class TaskConfig(BaseModel):
    """Configuration for a registered evaluation task."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    weights: TaskWeights = Field(default_factory=lambda: TaskWeights())
    judge_prompt_path: Optional[str] = Field(default=None)
    scoring_config: Optional[Dict[str, Any]] = Field(default=None)
    extends: Optional[str] = Field(
        default=None,
        description="Inherit from another task's configuration",
    )


# ---------------------------------------------------------------------------
# Episode log (for structured logging and replay)
# ---------------------------------------------------------------------------

class EpisodeLog(BaseModel):
    """Structured log of a complete episode for debugging, auditing, and replay."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(...)
    task_type: str = Field(...)
    session_id: Union[int, str] = Field(...)
    scoring_mode: ScoringMode = Field(default="prelabeled")
    start_time: float = Field(..., description="Unix timestamp of episode start")
    end_time: Optional[float] = Field(default=None)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = Field(default=0.0)
    final_score: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

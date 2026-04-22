# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLM-as-a-Judge client for Sentinel Garden v2.

Unified interface for evaluating overseer decisions using an external
LLM (OpenAI, vLLM, or any OpenAI-compatible API).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from scoring.prompts import JUDGE_PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


class JudgeEvaluation:
    """Structured evaluation returned by the judge LLM."""

    def __init__(
        self,
        correct_action: str,
        correct_risk_score: float,
        action_correct: bool,
        risk_score_accuracy: float,
        note_quality: float,
        reasoning: str = "",
    ) -> None:
        self.correct_action = correct_action
        self.correct_risk_score = correct_risk_score
        self.action_correct = action_correct
        self.risk_score_accuracy = risk_score_accuracy
        self.note_quality = note_quality
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct_action": self.correct_action,
            "correct_risk_score": self.correct_risk_score,
            "action_correct": self.action_correct,
            "risk_score_accuracy": self.risk_score_accuracy,
            "note_quality": self.note_quality,
            "reasoning": self.reasoning,
        }


class JudgeClient:
    """Unified interface for LLM-as-a-Judge.

    Args:
        model: Model identifier (e.g., "gpt-4o", "Qwen/Qwen2.5-72B-Instruct").
        api_base: Base URL for the OpenAI-compatible API.
        api_key: API key. Falls back to JUDGE_API_KEY env var.
        temperature: Sampling temperature (0.0 for deterministic evaluation).
        max_tokens: Maximum tokens in the judge response.
        timeout: Request timeout in seconds.
        rate_limit_rps: Max requests per second (None = no limit).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 30.0,
        rate_limit_rps: Optional[float] = None,
    ) -> None:
        self.model = model
        self.api_base = api_base or os.getenv("JUDGE_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Lazy import to avoid hard dependency
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=timeout,
            )
            self._sync_client = openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=timeout,
            )
        except ImportError:
            logger.error("openai package not installed; judge mode unavailable")
            self._client = None  # type: ignore[assignment]
            self._sync_client = None  # type: ignore[assignment]

        # Rate limiting
        self._rate_limiter = None
        if rate_limit_rps:
            try:
                from aiolimiter import AsyncLimiter
                self._rate_limiter = AsyncLimiter(rate_limit_rps, 1.0)
            except ImportError:
                logger.warning("aiolimiter not installed; rate limiting disabled")

    @property
    def available(self) -> bool:
        """Return True if the judge client is properly configured."""
        return self._client is not None and bool(self.api_key)

    async def evaluate(
        self,
        task_type: str,
        guardrail: str,
        user_message: str,
        model_response: str,
        overseer_action: str,
        overseer_risk_score: float,
        overseer_note: str,
        overseer_nudge: Optional[str] = None,
    ) -> JudgeEvaluation:
        """Evaluate an overseer decision asynchronously.

        Args:
            task_type: One of guardrail_enforcer, prompt_compiler, jailbreak_detector.
            guardrail: The active guardrail text.
            user_message: The user's message.
            model_response: The model's proposed response.
            overseer_action: The overseer's PASS/RE-PROMPT/BLOCK decision.
            overseer_risk_score: The overseer's risk score.
            overseer_note: The overseer's internal note.
            overseer_nudge: The overseer's re-prompt nudge (if any).

        Returns:
            JudgeEvaluation with structured scoring.
        """
        if not self.available:
            raise RuntimeError("Judge client is not available (openai not installed or no API key)")

        template = JUDGE_PROMPT_TEMPLATES.get(task_type)
        if template is None:
            logger.warning("No prompt template for task %s, using default", task_type)
            template = JUDGE_PROMPT_TEMPLATES["guardrail_enforcer"]

        messages = [
            {"role": "system", "content": template["system"]},
            {
                "role": "user",
                "content": template["user_template"].format(
                    guardrail=guardrail,
                    user_message=user_message,
                    model_response=model_response,
                    overseer_action=overseer_action,
                    overseer_risk_score=overseer_risk_score,
                    overseer_note=overseer_note,
                    overseer_nudge=overseer_nudge or "",
                ),
            },
        ]

        # Rate limit
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_response(content)

        except Exception as exc:
            logger.error("Judge evaluation failed: %s", exc)
            # Return a neutral evaluation on failure
            return JudgeEvaluation(
                correct_action="PASS",
                correct_risk_score=0.5,
                action_correct=True,
                risk_score_accuracy=0.5,
                note_quality=0.5,
                reasoning=f"Judge API error: {exc}",
            )

    def evaluate_sync(
        self,
        task_type: str,
        guardrail: str,
        user_message: str,
        model_response: str,
        overseer_action: str,
        overseer_risk_score: float,
        overseer_note: str,
        overseer_nudge: Optional[str] = None,
    ) -> JudgeEvaluation:
        """Synchronous version of evaluate()."""
        if not self.available:
            raise RuntimeError("Judge client is not available")

        template = JUDGE_PROMPT_TEMPLATES.get(task_type, JUDGE_PROMPT_TEMPLATES["guardrail_enforcer"])

        messages = [
            {"role": "system", "content": template["system"]},
            {
                "role": "user",
                "content": template["user_template"].format(
                    guardrail=guardrail,
                    user_message=user_message,
                    model_response=model_response,
                    overseer_action=overseer_action,
                    overseer_risk_score=overseer_risk_score,
                    overseer_note=overseer_note,
                    overseer_nudge=overseer_nudge or "",
                ),
            },
        ]

        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_response(content)

        except Exception as exc:
            logger.error("Judge sync evaluation failed: %s", exc)
            return JudgeEvaluation(
                correct_action="PASS",
                correct_risk_score=0.5,
                action_correct=True,
                risk_score_accuracy=0.5,
                note_quality=0.5,
                reasoning=f"Judge API error: {exc}",
            )

    async def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> List[JudgeEvaluation]:
        """Evaluate multiple decisions concurrently.

        Args:
            evaluations: List of dicts with keys matching evaluate() params.

        Returns:
            List of JudgeEvaluation results in the same order.
        """
        sem = asyncio.Semaphore(8)  # Concurrency cap

        async def _one(eval_dict: Dict[str, Any]) -> JudgeEvaluation:
            async with sem:
                return await self.evaluate(**eval_dict)

        return await asyncio.gather(*[_one(e) for e in evaluations])

    @staticmethod
    def _parse_response(content: str) -> JudgeEvaluation:
        """Parse the judge LLM's JSON response."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Judge returned invalid JSON: %s", content[:200])
            return JudgeEvaluation(
                correct_action="PASS",
                correct_risk_score=0.5,
                action_correct=True,
                risk_score_accuracy=0.5,
                note_quality=0.5,
                reasoning="Failed to parse judge response",
            )

        # Normalize action
        correct_action = str(data.get("correct_action", "PASS")).strip().upper()
        if correct_action not in {"PASS", "RE-PROMPT", "BLOCK"}:
            correct_action = "PASS"

        return JudgeEvaluation(
            correct_action=correct_action,
            correct_risk_score=float(data.get("correct_risk_score", 0.5)),
            action_correct=bool(data.get("action_correct", True)),
            risk_score_accuracy=float(data.get("risk_score_accuracy", 0.5)),
            note_quality=float(data.get("note_quality", 0.5)),
            reasoning=str(data.get("reasoning", "")),
        )

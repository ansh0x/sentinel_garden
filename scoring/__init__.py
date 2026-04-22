# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scoring engine for Sentinel Garden v2.

Pluggable scoring with three modes:
    prelabeled — exact label matching with configurable tolerance
    judge      — LLM-as-a-Judge for open-ended evaluation
    hybrid     — combination of both (future)
"""

from scoring.engine import ScoringEngine, ScoringMode
from scoring.rewards import SmoothRewardFunction, DiscreteRewardFunction
from scoring.judge_client import JudgeClient
from scoring.prompts import JUDGE_PROMPT_TEMPLATES

__all__ = [
    "ScoringEngine",
    "ScoringMode",
    "SmoothRewardFunction",
    "DiscreteRewardFunction",
    "JudgeClient",
    "JUDGE_PROMPT_TEMPLATES",
]

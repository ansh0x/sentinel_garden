# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sentinel Garden v2 — Market-Ready RL Environment for LLM Safety Overseers

Key features:
    - Pluggable scoring engine (prelabeled / judge / hybrid modes)
    - Smooth continuous reward functions
    - Semantic nudge matching
    - Dynamic task registration
    - Episode logging and replay
    - Session store abstraction

Usage:
    from models import SentinelGardenAction, SentinelGardenObservation
    from client import SentinelGardenEnv

    env = SentinelGardenEnv(base_url="http://localhost:8000")
    result = await env.reset(task_type="guardrail_enforcer")
    observation = result.observation
"""

__version__ = "2.0.0"

from models import (
    SentinelGardenAction,
    SentinelGardenObservation,
    SentinelGardenState,
    VALID_TASKS,
)
from client import SentinelGardenEnv

__all__ = [
    "SentinelGardenAction",
    "SentinelGardenObservation",
    "SentinelGardenState",
    "SentinelGardenEnv",
    "VALID_TASKS",
]

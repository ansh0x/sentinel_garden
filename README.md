---
title: Sentinel Garden Environment Server
emoji: 🌿
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - guardrails
  - llm-safety
  - rl-environment
---
# Sentinel Garden v2

**Market-Ready RL Environment for Training LLM Safety Overseers**

Sentinel Garden is an OpenEnv-compatible reinforcement learning environment for training **Overseer agents** — small models that monitor LLM outputs and decide `PASS`, `RE-PROMPT`, or `BLOCK` per turn. Version 2 represents a ground-up rebuild focused on production readiness, extensibility, and dual-mode scoring.

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| **Scoring Modes** | Pre-labeled only | Pre-labeled + LLM-as-a-Judge + Hybrid |
| **Reward Function** | Discrete 3-bucket | Smooth continuous (better for RL) |
| **Nudge Matching** | Exact string | Semantic similarity |
| **Task System** | Hardcoded enum | Dynamic YAML/JSON registration |
| **Session Store** | Single JSON file | Pluggable (JSON/DB/API) |
| **Episode Logging** | None | Full trajectory + replay |
| **Data Validation** | Basic | Pydantic v2 strict validation |
| **Concurrent Envs** | 1 | Configurable (default 4) |
| **Health Checks** | Basic ping | Deep (model, data, judge) |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ansh0x/sentinel_garden.git
cd sentinel_garden

# Install dependencies
pip install -e ".[all]"
```

### Running the Server

```bash
# Start the environment server
python -m server.app --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Running Inference

```bash
# Set your API key for the judge/inference model
export API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"

# Run all 3 tasks
python inference.py
```

### Training with GRPO

```bash
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct

# Start environment server
python -m server.app --port 8001

# Run training
CUDA_VISIBLE_DEVICES=1 python train.py
```

## Architecture

```
                    Client / Trainer
                           |
                    HTTP/WebSocket
                           |
              +------------------------+
              |  Sentinel Garden v2    |
              |                        |
              |  +------------------+  |
              |  | Scoring Engine   |  |
              |  | - Prelabeled     |  |
              |  | - Judge (LLM)    |  |
              |  | - Hybrid         |  |
              |  +------------------+  |
              |  | Task Registry    |  |
              |  | (Dynamic)        |  |
              |  +------------------+  |
              |  | Session Store    |  |
              |  | (JSON/DB/API)    |  |
              |  +------------------+  |
              |  | Episode Logger   |  |
              |  +------------------+  |
              +------------------------+
```

## Configuration

The environment is configured via `config.yaml`:

```yaml
environment:
  scoring_mode: prelabeled  # prelabeled | judge | hybrid
  max_concurrent_envs: 4

scoring:
  prelabeled:
    nudge_match: semantic      # exact | semantic
    risk_function: smooth      # discrete | smooth

  judge:
    model: gpt-4o
    api_base: https://api.openai.com/v1
    temperature: 0.0
```

Override via environment variables:

- `SENTINEL_SCORING_MODE` — scoring mode
- `SENTINEL_TASK` — default task type
- `JUDGE_API_KEY` — API key for judge mode
- `SESSIONS_PATH` — path to session data

## Task System

### Built-in Tasks

| Task | Focus | Weights (Action/Risk/Note) |
|---|---|---|
| `guardrail_enforcer` | Policy violation detection | 60% / 30% / 10% |
| `prompt_compiler` | Context drift detection | 30% / 20% / 50% |
| `jailbreak_detector` | Injection/jailbreak detection | 40% / 50% / 10% |

### Adding Custom Tasks

```python
from task_registry import TaskRegistry
from models import TaskWeights

registry = TaskRegistry()
registry.register_task(
    task_id="custom_pii_detector",
    name="PII Detector",
    description="Detects PII leakage in model outputs",
    weights=TaskWeights(action=0.50, risk=0.40, note=0.10),
)
```

Or via YAML:

```yaml
tasks:
  - id: custom_pii_detector
    name: "PII Detector"
    weights:
      action: 0.50
      risk: 0.40
      note: 0.10
    judge_prompt: prompts/pii_detector.txt
    extends: guardrail_enforcer
```

## Scoring Engine

### Mode 1: Pre-labeled (Default)

Uses ground-truth labels from the dataset. Supports:

- **Semantic nudge matching** — sentence similarity instead of exact string
- **Smooth risk rewards** — exponential decay instead of discrete buckets
- **Partial credit** — close actions get intermediate rewards

### Mode 2: LLM-as-a-Judge

Uses an external LLM to evaluate overseer decisions dynamically.

```bash
export SENTINEL_SCORING_MODE=judge
export JUDGE_API_KEY="your-openai-key"
export JUDGE_MODEL="gpt-4o"
python -m server.app
```

The judge receives task-specific prompts that adapt the evaluation criteria
to each task type (guardrail vs drift vs jailbreak).

### Mode 3: Hybrid (Future)

Combines pre-labeled data for "obvious" cases with judge evaluation for edge cases.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Submit an action |
| GET | `/state` | Get current state |
| GET | `/schema` | Get action/observation schemas |
| GET | `/health` | Deep health check |
| WS | `/ws` | WebSocket session |

## Docker

```bash
# Build
docker build -t sentinel-garden:v2 -f server/Dockerfile .

# Run
docker run -p 8000:8000 \
  -e SENTINEL_SCORING_MODE=prelabeled \
  -v $(pwd)/data:/app/data \
  sentinel-garden:v2
```

## File Structure

```
sentinel_garden/
|-- config.yaml              # Main configuration
|-- models.py                # Pydantic v2 data models
|-- task_registry.py         # Dynamic task registration
|-- client.py                # HTTP/WebSocket client
|-- grader.py                # Task-specific graders
|-- train.py                 # GRPO training script
|-- inference.py             # Multi-task inference
|-- server/
|   |-- app.py               # FastAPI application
|   |-- sentinel_garden_environment.py  # Environment core
|   |-- Dockerfile
|-- scoring/
|   |-- engine.py            # Pluggable scoring engine
|   |-- rewards.py           # Smooth/discrete reward functions
|   |-- judge_client.py      # LLM-as-a-Judge client
|   |-- prompts.py           # Task-specific judge prompts
|-- session_store/
|   |-- base.py              # Abstract session store
|   |-- json_store.py        # JSON file backend
|-- logging/
|   |-- episode_logger.py    # Episode trajectory logging
|-- prompts/
|   |-- guardrail_enforcer.txt
|   |-- prompt_compiler.txt
|   |-- jailbreak_detector.txt
|-- data/
|   |-- sessions.json        # 15 expanded sessions
```

## License

BSD-3-Clause License. See LICENSE file for details.

## Acknowledgments

Originally developed for the **Meta X Scalar Hackathon India**.

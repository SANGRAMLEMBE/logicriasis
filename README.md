---
title: LogiCrisis Multi-Agent Logistics Recovery
emoji: 🚛
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - logistics
  - supply-chain
  - grpo
  - trl
license: mit
---

# LogiCrisis: Multi-Agent Logistics Recovery

**Meta PyTorch OpenEnv Hackathon — Theme #1: Multi-Agent Interactions**

A real-world supply chain crisis simulation where LLM agents cooperate, negotiate, and form coalitions to restore India's logistics network after cascading disruptions. Built on the OpenEnv spec with 5 agent roles, 6 reward signals, and 3 progressively harder graded tasks.

---

## Environment Overview

India's supply network — 10 cities, 26 bidirectional routes — is hit by floods, port strikes, and road closures. Agents operate under **partial observability**: each sees only its own region, cargo queue, neighbor bids, and coalition proposals. To succeed, agents must reason about other agents' hidden state (theory-of-mind), negotiate fair SLAs, and form coalitions that share reward proportionally.

### Network

```
Mumbai ─── Pune ─── Hyderabad ─── Bangalore ─── Chennai
  │                     │
Surat ── Ahmedabad ── Delhi ─── Jaipur
              Kolkata ──────────────────────────┘
```

**10 cities**: Mumbai, Delhi, Kolkata, Chennai, Bangalore, Hyderabad, Pune, Ahmedabad, Jaipur, Surat  
**Disruption types**: Flood, Port Strike, Road Closure (each blocks a set of routes for the full episode)

---

## Agent Roles

| Role | ID | Specialisation |
|---|---|---|
| Carrier | `carrier_0` | Freight transport, rerouting |
| Warehouse | `warehouse_0` | Cold storage, cargo staging |
| Customs Broker | `customs_broker_0` | Cross-border clearance |
| Insurer | `insurer_0` | Risk assessment, cold-chain insurance |
| Shipper | `shipper_0` | SLA negotiation, cargo priority |

---

## Observation Space

Each agent receives a **partial** `ObservationSchema` JSON object per turn:

```json
{
  "agent_id": "carrier_0",
  "role": "carrier",
  "turn": 3,
  "max_turns": 20,
  "own_region": "West",
  "own_capacity_tons": 142.5,
  "own_budget": 9800.0,
  "own_cargo_queue": ["C001", "C004"],
  "pending_deadlines": [["C001", 5], ["C004", 8]],
  "disrupted_routes": ["Chennai-Bangalore", "Bangalore-Chennai"],
  "disrupted_nodes": ["Chennai"],
  "neighbor_bids": [{"bid_id": "a3f1", "from": "warehouse_0", "cargo": "C001", "price": 120.0}],
  "coalition_proposals": [{"coalition_id": "coal_x", "lead": "warehouse_0", "members": [...]}],
  "action_history": [...],
  "active_coalition_id": null,
  "active_contracts": [],
  "prompt_text": "..."
}
```

`prompt_text` is a natural language rendering of all the above, ready to pass directly to an LLM.

---

## Action Space

Agents submit structured JSON actions. All 13 action types are valid at any turn:

```json
{
  "agent_id": "carrier_0",
  "action_type": "reroute",
  "cargo_id": "C001",
  "route_id": "Mumbai-Pune",
  "reasoning": "Direct route to destination, unblocked"
}
```

| Category | Action Types |
|---|---|
| Logistics | `reroute`, `request_transfer`, `prioritize_cargo`, `deploy_cold_storage` |
| Negotiation | `make_bid`, `accept_bid`, `reject_bid`, `counter_propose` |
| Coalition | `propose_coalition`, `join_coalition`, `leave_coalition`, `assign_coalition_role` |
| No-op | `wait` |

**Delivery rule**: A `reroute` action delivers cargo if `route.to_node == cargo.destination` and the route is not blocked.

---

## Reward Signals

6 independent, additive reward components per agent per turn:

| Signal | Description |
|---|---|
| R1 — Delivery | +1.0 per on-time delivery, proportional partial credit |
| R2 — Coalition | Bonus for maintaining active, fair coalitions |
| R3 — Negotiation | Reward for accepted bids at fair prices |
| R4 — Cold Chain | Penalty if temp-sensitive cargo spoils |
| R5 — Efficiency | Budget and capacity utilisation score |
| R6 — Anti-Cheat | Penalty for suspicious action patterns (overseer detection) |

All reward components are in `[0.0, 1.0]` before summing. Episode-level grader scores are normalised to `[0.0, 1.0]`.

---

## Tasks

### Task 1 — Single Route Recovery (Easy)

- **Agents**: 2 (`carrier_0`, `warehouse_0`)
- **Cargo**: 5 items, 1 disruption
- **Max turns**: 10
- **Grader**: `score = (on_time × 1.0 + late × 0.5) / total`
- **Pass threshold**: ≥ 0.60
- **Baseline (heuristic)**: **1.0000 — PASS** (OTIF 100%)

### Task 2 — Coalition Logistics (Medium)

- **Agents**: 3 (`carrier_0`, `warehouse_0`, `customs_broker_0`)
- **Cargo**: 15 items including 6 cold-chain, 2 disruptions
- **Max turns**: 15
- **Grader**: `0.5 × OTIF + 0.3 × cold_chain_integrity + 0.2 × coalition_formed`
- **Pass threshold**: ≥ 0.55
- **Baseline (heuristic)**: **0.7667 — PASS** (OTIF 73.3%, cold 0.667, coalition 1.0)

### Task 3 — Cascade Failure Recovery (Hard)

- **Agents**: 5 (all roles)
- **Cargo**: 20 items including 6 cold-chain, 3 disruptions, 60% routes blocked
- **Max turns**: 20
- **Grader**: `0.4 × OTIF + 0.3 × cold_chain + 0.2 × turn_efficiency + 0.1 × budget_efficiency`
- **Cascade penalty**: if >60% cargo spoils, score ×= 0.5
- **Pass threshold**: ≥ 0.45
- **Baseline (heuristic)**: **0.6254 — PASS** (OTIF 85.0%)

### Task 4 — Cold Chain Emergency (Medium-Hard)

- **Agents**: 3 (`carrier_0`, `warehouse_0`, `customs_broker_0`)
- **Cargo**: 12 items, ALL temperature-sensitive (cold_chain_ratio=1.0), 2 disruptions
- **Max turns**: 12
- **Grader**: `0.7 × cold_chain_integrity + 0.3 × OTIF`; cascade penalty ×0.5 if >50% spoiled
- **Pass threshold**: ≥ 0.60
- **Baseline (heuristic)**: **0.8333 — PASS** (cold 0.833, OTIF 83.3%)

### Task 5 — Negotiation Sprint (Medium)

- **Agents**: 4 (`carrier_0`, `warehouse_0`, `customs_broker_0`, `insurer_0`)
- **Cargo**: 10 items, 1 disruption
- **Max turns**: 10
- **Grader**: `0.35 × OTIF + 0.40 × negotiation_activity + 0.25 × coalition_quality`
- **Pass threshold**: ≥ 0.50
- **Baseline (heuristic)**: **0.6000 — PASS** (negotiation_score=0.0 — LLMs expected to score higher)

### Task 6 — Full National Recovery (Expert)

- **Agents**: 5 (all roles)
- **Cargo**: 25 items (40% cold-chain), 4 disruptions
- **Max turns**: 25
- **Grader**: `0.30×OTIF + 0.25×cold_chain + 0.20×coalition + 0.15×negotiation + 0.10×budget`
- **Cascade penalty**: if >50% spoiled, score ×= 0.4
- **Pass threshold**: ≥ 0.35 (very hard)
- **Baseline (heuristic)**: **0.6261 — PASS** (OTIF 68.0%)

### Task 7 — Earthquake Relief Operations (Hard) ★ Research Task

- **Agents**: 4, **Cargo**: 18, **Disruptions**: 3, **Max turns**: 15
- **Priority tiers**: CRITICAL medical (4×), HIGH rescue (2×), MEDIUM food/water (1×), LOW shelter (0.5×)
- **Grader**: Priority-weighted OTIF; −0.15 per undelivered CRITICAL item
- **Pass threshold**: ≥ 0.55
- **Baseline (heuristic)**: **0.1176 — FAIL** (needs priority reasoning — heuristic cannot triage by urgency)

### Task 8 — Capacity Crunch (Hard) ★ Research Task

- **Agents**: 5, **Cargo**: 20, **Disruptions**: 2, **Max turns**: 15, **capacity_multiplier**: 0.25
- **Scenario**: Fleet at 25% capacity (COVID-surge driver shortage). Must trade via bid market.
- **Grader**: `0.40×OTIF + 0.35×utilisation + 0.25×market_activity`
- **Pass threshold**: ≥ 0.45
- **Baseline (heuristic)**: **0.3770 — FAIL** (market_score=0.0 — heuristic never bids)

### Task 9 — Just-In-Time Breakdown (Medium-Hard)

- **Agents**: 3, **Cargo**: 14, **Disruptions**: 2, **Max turns**: 10, **deadline_max**: 6
- **Grader**: `0.6 × value_score + 0.4 × triage_score` (strict — zero credit for late delivery)
- **Pass threshold**: ≥ 0.50
- **Baseline (heuristic)**: **0.9515 — PASS** (12/14 on-time, triage_score=1.0)

**Aggregate baseline score (9 tasks, seed=42): 0.6553**

---

## API (OpenEnv Spec)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start episode: `{"task_id": "...", "seed": 42}` |
| `POST` | `/step` | Execute turn: `{"actions": [...ActionSchema]}` |
| `GET` | `/state` | Full world snapshot (ground truth) |
| `GET` | `/tasks` | List all 9 tasks with metadata |
| `POST` | `/grade` | Run grader on current episode → score 0.0–1.0 |
| `GET` | `/validate` | OpenEnv self-validation (returns pass/fail per check) |
| `GET` | `/action_types` | All valid `action_type` values |
| `GET` | `/agent_roles` | All valid agent roles |

---

## Setup

### Local (API + Demo)

```bash
git clone <repo>
cd logicriasis
pip install fastapi uvicorn pydantic openai gradio numpy httpx

# Start API server
uvicorn api.app:app --reload --port 8000

# In another terminal: run Gradio demo
python demo/app.py

# Run inference baseline (heuristic, no API key needed)
python inference.py

# Run inference with LLM
API_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct \
HF_TOKEN=hf_xxx \
python inference.py
```

### Docker

```bash
docker build -t logicriasis .
docker run -p 7860:7860 logicriasis

# With LLM inference
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct \
  -e HF_TOKEN=hf_xxx \
  logicriasis python inference.py
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use for agent policy |
| `HF_TOKEN` | *(none)* | HuggingFace token (or `OPENAI_API_KEY`) |

---

## Inference Script

`inference.py` in the root directory runs all 9 tasks and emits structured stdout logs:

```
[START] {"task_id": "single_route_recovery", "agent_ids": [...], "max_turns": 10, ...}
[STEP]  {"turn": 1, "actions": {...}, "rewards": {...}, "otif_percent": 40.0, ...}
[STEP]  {"turn": 2, ...}
[END]   {"task_id": "single_route_recovery", "score": 1.0, "passed": true, "verdict": "PASS", ...}
```

If no API key is set, runs the deterministic heuristic baseline (7/9 tasks PASS, average 0.6553). With an LLM key, uses the model for action generation with automatic heuristic fallback on parse errors. Runtime < 20 minutes on vCPU=2/8GB RAM.

---

## Project Structure

```
logicriasis/
├── inference.py              # OpenEnv baseline script (root, required)
├── openenv.yaml              # OpenEnv manifest (9 tasks)
├── Dockerfile                # Container for HF Spaces (port 7860)
├── requirements.txt
├── environment/
│   ├── models.py             # AgentRole, ActionType, Cargo, Route, Disruption, etc.
│   ├── world.py              # WorldState, India network topology (10 cities, 13 edges)
│   ├── env.py                # LogiCrisisEnv (reset/step/state)
│   ├── rewards.py            # 6 reward functions (R1–R6) + anti-cheat overseer
│   ├── schemas.py            # Pydantic API schemas
│   └── tasks/
│       ├── task1_single_route.py        # Easy: 2 agents, 5 cargo
│       ├── task2_coalition_logistics.py # Medium: coalition + cold-chain
│       ├── task3_cascade_failure.py     # Hard: 60% routes blocked
│       ├── task4_cold_chain_emergency.py # Medium-Hard: 100% temp-sensitive
│       ├── task5_negotiation_sprint.py  # Medium: bid/counter-propose focus
│       ├── task6_national_recovery.py   # Expert: all mechanics, 25 turns
│       ├── task7_earthquake_relief.py   # Hard ★: humanitarian priority triage
│       ├── task8_capacity_crunch.py     # Hard ★: market-based capacity trading
│       └── task9_jit_breakdown.py       # Medium-Hard: JIT strict OTIF + triage
├── api/
│   └── app.py                # FastAPI OpenEnv server
├── demo/
│   └── app.py                # Gradio interactive demo (live OTIF chart, grader panel)
├── agents/
│   └── prompts.py            # LLM system/user prompt builders
└── training/
    └── train.py              # GRPO training with TRL + Unsloth 4-bit QLoRA (optional)
```

---

## Baseline Scores

Heuristic policy (no LLM, deterministic, seed=42):

| Task | Difficulty | Score | OTIF | Status | Note |
|---|---|---|---|---|---|
| single_route_recovery | Easy | **1.0000** | 100.0% | ✓ PASS | |
| coalition_logistics | Medium | **0.7667** | 73.3% | ✓ PASS | |
| cascade_failure_recovery | Hard | **0.6254** | 85.0% | ✓ PASS | |
| cold_chain_emergency | Medium-Hard | **0.8333** | 83.3% | ✓ PASS | |
| negotiation_sprint | Medium | **0.6000** | 100.0% | ✓ PASS | negotiation_score=0.0 |
| national_recovery | Expert | **0.6261** | 68.0% | ✓ PASS | |
| earthquake_relief | Hard | **0.1176** | 56.8% | ✗ FAIL | needs priority reasoning |
| capacity_crunch | Hard | **0.3770** | 55.0% | ✗ FAIL | needs market bidding |
| jit_breakdown | Medium-Hard | **0.9515** | 85.7% | ✓ PASS | |
| **Average** | | **0.6553** | | **7/9 PASS** | |

Tasks 7 (earthquake_relief) and 8 (capacity_crunch) are **intentional heuristic failures** — they require capabilities that rule-based agents cannot demonstrate: humanitarian priority reasoning and market-based capacity trading. These are the key research targets for LLM fine-tuning via GRPO.

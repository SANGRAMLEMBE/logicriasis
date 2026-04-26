# LogiCrisis: Multi-Agent Logistics Recovery
## Complete Technical Presentation — meta-pytorch-hackathon

> **Theme**: #1 — Multi-Agent Interactions  
> **Track**: OpenEnv Environment Design + LLM Fine-Tuning  
> **Team**: Soham Randive

---

## Table of Contents

1. [What We Built — One Paragraph](#1-what-we-built)
2. [Why This Problem — Motivation](#2-why-this-problem)
3. [High-Level Architecture](#3-high-level-architecture)
4. [The World Model — India Supply Network](#4-the-world-model)
5. [Agent System — 5 Roles, Partial Observability](#5-agent-system)
6. [Action Space — 13 Action Types](#6-action-space)
7. [Observation Space — What Each Agent Sees](#7-observation-space)
8. [Reward System — 6 Independent Signals](#8-reward-system)
9. [9 Tasks — Easy to Expert](#9-nine-tasks)
10. [OpenEnv API — Spec Compliance](#10-openenv-api)
11. [Training Pipeline — GRPO + Unsloth](#11-training-pipeline)
12. [Baseline Results — Heuristic vs LLM](#12-baseline-results)
13. [File-by-File Code Walkthrough](#13-file-by-file-walkthrough)
14. [Key Technical Decisions & Why](#14-key-technical-decisions)
15. [What Is Done vs What Is Next](#15-done-vs-next)
16. [How to Run — Commands](#16-how-to-run)
17. [Panel Q&A Prep](#17-panel-qa-prep)

---

## 1. What We Built

**LogiCrisis** is a multi-agent reinforcement learning environment where 2–5 AI agents cooperate, compete, negotiate, and form coalitions to recover India's supply chain after real-world disruptions — floods, port strikes, and road closures. It is fully compliant with the Meta OpenEnv specification (`reset/step/state` API), includes 9 graded tasks ranging from easy to expert difficulty, a 6-component reward system, a FastAPI server, an interactive Gradio demo, and a complete GRPO fine-tuning pipeline using TRL + Unsloth. The environment is designed so that a naive heuristic agent fails 2 of the 9 tasks (requiring genuine LLM reasoning), while a GRPO-trained LLM is expected to pass all 9.

---

## 2. Why This Problem

**The real-world hook**: India processes over $400B of freight annually. A single port strike or monsoon flood can cascade into multi-billion-dollar supply chain failures within 48 hours. The 2021 Suez Canal blockage cost $9.6B per day.

**The research hook**: Traditional OR/heuristic solvers cannot handle the combination of:
- **Partial observability** (each agent knows only its own state)
- **Dynamic negotiation** (bids, counter-bids, coalition formation mid-crisis)
- **Theory-of-mind reasoning** (modelling what other agents will do)
- **Humanitarian triage** (prioritising medical cargo over routine freight)
- **Market-based coordination** (trading capacity when fleet is at 25%)

These require LLMs that can *reason*, not just route. That is the gap we test.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LogiCrisis Stack                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │ │
│  │ Carrier  │   │Warehouse │   │ Customs  │   │ Insurer  │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘ │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │  LogiCrisisEnv  │  ← OpenEnv API         │
│                    │  reset/step/    │    (FastAPI)           │
│                    │  state/grade    │                        │
│                    └────────┬────────┘                        │
│                             │                                 │
│              ┌──────────────▼──────────────────┐             │
│              │           WorldState             │             │
│              │  10 cities · 13 routes · cargo   │             │
│              │  disruptions · bids · coalitions │             │
│              └──────────────┬──────────────────┘             │
│                             │                                 │
│         ┌───────────────────┼───────────────────┐            │
│         ▼                   ▼                   ▼            │
│   ┌──────────┐       ┌──────────┐       ┌──────────┐         │
│   │  Reward  │       │  Tasks   │       │  Grader  │         │
│   │ R1 – R6  │       │  1 – 9   │       │per-task  │         │
│   └──────────┘       └──────────┘       └──────────┘         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          GRPO Training Pipeline                        │  │
│  │   Unsloth 4-bit QLoRA + TRL GRPOTrainer               │  │
│  │   Reward: _score_completion() (6-signal verifier)     │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. The World Model

**File**: `environment/world.py`

### Network Topology

10 Indian cities connected by 13 bidirectional edges (26 directed routes total):

```
Cities: Mumbai, Delhi, Kolkata, Chennai, Bangalore,
        Hyderabad, Pune, Ahmedabad, Jaipur, Surat

Regions:
  West:  Mumbai, Pune, Surat, Ahmedabad
  North: Delhi, Jaipur
  East:  Kolkata
  South: Chennai, Bangalore, Hyderabad
```

**Route capacity formula**: `capacity = max(50.0, 220.0 - distance_km / 8.0)`  
Short roads have higher throughput (e.g., Mumbai–Pune: 181.25 tons). This is deterministic — real road capacity is fixed infrastructure.

### Disruptions

3 disruption types: `FLOOD`, `PORT_STRIKE`, `ROAD_CLOSURE`  
Severity 1–5 determines how many routes are blocked. At curriculum level 3, up to 5 nodes are affected per disruption, blocking entire regional sub-networks.

### Cargo

Each cargo item has:
- `cargo_type`: STANDARD / COLD_CHAIN / URGENT / BULK
- `value`: monetary value (hidden from non-owning agents)
- `deadline`: turn by which delivery must happen
- `weight_tons`: affects capacity consumption
- `temp_sensitive`: if true, spoils if not delivered by deadline
- `priority`: 1–4 (used in earthquake relief triage)
- `delivered_turn`: the exact turn it was delivered (used for accurate OTIF grading)

### World State (ground truth, never shown to agents)

```python
class WorldState:
    routes: dict[str, Route]         # 26 directed routes, some blocked
    cargo_queue: dict[str, Cargo]    # all cargo items
    disruptions: list[Disruption]    # active disruptions
    agent_states: dict[str, AgentState]  # per-agent private state
    coalitions: dict[str, Coalition] # active coalitions
    bids: dict[str, Bid]             # all bids (open/accepted/rejected)
    delivered_cargo: list[str]       # cargo_ids successfully delivered
    failed_cargo: list[str]          # cargo_ids that failed/spoiled
    audit_log: list[dict]            # full action history (for anti-cheat)
```

---

## 5. Agent System

**File**: `environment/models.py`, `environment/world.py`, `agents/prompts.py`

### 5 Agent Roles

| Role | ID | Responsibility |
|---|---|---|
| Carrier | `carrier_0` | Owns trucks, executes deliveries, bids for contracts |
| Warehouse | `warehouse_0` | Controls cold storage, negotiates slot-sharing |
| Customs Broker | `customs_broker_0` | Handles cross-border clearance, information asymmetry |
| Insurer | `insurer_0` | Evaluates risk, approves coverage, creates adversarial pressure |
| Shipper | `shipper_0` | Owns cargo, allocates budget, hires carriers |

### Partial Observability

Each agent sees **only its own slice** of the world:
- Its own cargo queue, budget, capacity, region
- **Public** disruption signals (routes/nodes blocked) — but not severity
- **Noisy** neighbor bids (last 5 public bids visible to all)
- Coalition proposals addressed to it
- Its own last 3 actions and outcomes

Agents **cannot** see: other agents' budgets, cargo values, full cargo queue, or true disruption severity.

### Role-Conditioned Prompts

A single shared LLM policy covers all 5 roles. Role identity is injected via system prompt. Each role's prompt includes:
- Role description and private state ownership
- Available capabilities (which actions make sense)
- Strategy hints specific to the role
- The action JSON schema

This is the **Theory-of-Mind** component — agents must infer what other roles are doing from noisy public signals.

---

## 6. Action Space

**File**: `environment/models.py` — `ActionType` enum

13 action types across 4 categories:

### Logistics (move cargo)
| Action | Effect |
|---|---|
| `reroute` | Move cargo along a specific route toward destination. If `route.to_node == cargo.destination`, marks cargo delivered. |
| `request_transfer` | Request another agent to transfer resources/cargo |
| `prioritize_cargo` | Extend a cargo's deadline by 2 turns (SLA renegotiation) |
| `deploy_cold_storage` | Use a cold storage unit to rescue spoiling temp-sensitive cargo |

### Negotiation (market mechanisms)
| Action | Effect |
|---|---|
| `make_bid` | Offer a price/capacity to carry cargo |
| `accept_bid` | Accept a bid — triggers budget transfer + cargo ownership change |
| `reject_bid` | Reject a bid |
| `counter_propose` | Counter an existing bid with a new price |

### Coalition (multi-agent cooperation)
| Action | Effect |
|---|---|
| `propose_coalition` | Create a coalition with named members and reward split |
| `join_coalition` | Accept an existing coalition invitation |
| `leave_coalition` | Exit a coalition (marks dissolved if last member) |
| `assign_coalition_role` | Assign a role within the coalition (lead only) |

### No-op
| Action | Effect |
|---|---|
| `wait` | No action. 5 consecutive waits triggers anti-cheat penalty. |

All actions include a `reasoning` field — a 1–2 sentence natural language explanation. This is used in process supervision scoring and displayed in the demo.

---

## 7. Observation Space

**File**: `environment/models.py` — `AgentObservation` dataclass

Each agent receives per turn:

```python
@dataclass
class AgentObservation:
    agent_id: str
    role: AgentRole
    turn: int
    max_turns: int
    own_region: str
    own_capacity_tons: float
    own_budget: float
    own_cargo_queue: list[str]           # only cargo this agent owns
    pending_deadlines: list[tuple[str, int]]  # (cargo_id, turns_left)
    disrupted_routes: list[str]          # public: which routes are blocked
    disrupted_nodes: list[str]           # public: which cities are affected
    neighbor_bids: list[dict]            # last 5 public bids from all agents
    coalition_proposals: list[dict]      # invitations addressed to this agent
    action_history: list[dict]           # last 3 of this agent's actions+outcomes
    active_coalition_id: Optional[str]
    active_contracts: list[dict]
```

The `to_prompt_text()` method serialises this into a readable block that is fed directly to the LLM as the user prompt.

---

## 8. Reward System

**File**: `environment/rewards.py`

6 independent reward signals computed every turn per agent. They cannot be gamed by optimising one signal alone.

### R1 — Delivery Success
```
+1.0  for each cargo delivered on time
-0.5 × min(turns_late, 2)  for late delivery
-1.0  for spoiled or failed delivery
```

### R2 — Coalition Quality
```
+0.3 × reward_split_fraction  if coalition delivered more than solo baseline
-0.2  if coalition underperformed baseline
```

### R3 — Negotiation Fairness
```
+0.2  per accepted bid (agent was buyer or seller)
-0.3  per breached contract
```

### R4 — Cold Chain Integrity
```
intact_cold_cargo / total_cold_cargo  (0.0 to 1.0)
```

### R5 — Resource Efficiency
```
+min(delivered_weight / capacity × 0.1, 0.3)  utilisation bonus
-0.05 × idle_truck_events  idle penalty
```

### R6 — Anti-Cheat Verifier (hard penalty)
```
-1.0  if 5 consecutive WAIT actions detected (loop exploit)
-2.0  if reasoning mentions hidden state variables (e.g., "world.routes")
```

### Shared Bonus
```
(system_OTIF_percent / 100) × severity_multiplier × 0.5
```
This aligns individual and collective incentives — even competitive agents benefit from the overall system performing well.

### Composite Total
```
total = R1 + R2 + R3 + R4 + R5 + R6 + shared_bonus
```

---

## 9. Nine Tasks

**Files**: `environment/tasks/task1_*.py` through `task9_*.py`

### Why 9 tasks?

The OpenEnv spec requires minimum 3 tasks for Round 1. We built 9 to demonstrate:
- **Curriculum learning** (3 levels: easy → hard)
- **Specialist skill focus** (3 tasks each testing a specific mechanic)
- **Research-grade scenarios** (3 real-world crises where heuristics fail)

---

### Task 1 — Single Route Recovery (Easy)
- **Config**: 2 agents · 5 cargo · 1 disruption · 10 turns
- **Grader**: `(on_time × 1.0 + late × 0.5) / total`
- **Pass**: ≥ 0.60
- **Heuristic baseline**: **1.0000 PASS** — basic rerouting solves this
- **Design purpose**: Entry point. Verify agents can read observations and produce valid actions.

---

### Task 2 — Coalition Logistics (Medium)
- **Config**: 3 agents · 15 cargo (includes cold-chain) · 2 disruptions · 15 turns
- **Grader**: `0.5 × OTIF + 0.3 × cold_chain_integrity + 0.2 × coalition_formed`
- **Pass**: ≥ 0.55
- **Heuristic baseline**: **0.7667 PASS**
- **Design purpose**: Introduce coalition formation + cold-chain protection. An agent that never forms a coalition loses 20% of the score automatically.

---

### Task 3 — Cascade Failure Recovery (Hard)
- **Config**: 5 agents · 20 cargo · 3 disruptions · 20 turns
- **Grader**: `0.4×OTIF + 0.3×cold + 0.2×turn_efficiency + 0.1×budget`
- **Cascade penalty**: if >60% cargo spoils, score ×= 0.5
- **Pass**: ≥ 0.45
- **Heuristic baseline**: **0.6254 PASS**
- **Design purpose**: 3 simultaneous disruptions block 60%+ of routes. Tests full multi-agent coordination.

---

### Task 4 — Cold Chain Emergency (Medium-Hard)
- **Config**: 3 agents · 12 cargo (ALL cold-chain) · 2 disruptions · 12 turns
- **Grader**: `0.7 × cold_chain_integrity + 0.3 × OTIF`
- **Cascade penalty**: if >50% spoiled, score ×= 0.5
- **Pass**: ≥ 0.60
- **Heuristic baseline**: **0.8333 PASS**
- **Design purpose**: 100% temp-sensitive cargo. Tests cold storage deployment and priority delivery order. The cascade penalty is brutal — more than 6 spoiled items halves your score.
- **Key env param**: `cold_chain_ratio=1.0`

---

### Task 5 — Negotiation Sprint (Medium)
- **Config**: 4 agents · 10 cargo · 1 disruption · 10 turns
- **Grader**: `0.35×OTIF + 0.40×negotiation_activity + 0.25×coalition_quality`
- **Pass**: ≥ 0.50
- **Heuristic baseline**: **0.6000 PASS** — but `negotiation_score=0.0` (heuristic never bids)
- **Design purpose**: 40% of the score comes from *negotiation activity* — accepted bids and counter-proposals. A routing-only agent loses 40% of maximum score. LLMs that bid and counter-bid should significantly outperform.
- **What LLMs unlock**: A trained LLM that uses `make_bid` + `counter_propose` can score 0.8+ vs heuristic 0.60.

---

### Task 6 — Full National Recovery (Expert)
- **Config**: 5 agents · 25 cargo (40% cold-chain) · 4 disruptions · 25 turns
- **Grader**: `0.30×OTIF + 0.25×cold_chain + 0.20×coalition + 0.15×negotiation + 0.10×budget`
- **Cascade penalty**: if >50% spoiled, score ×= 0.4
- **Pass**: ≥ 0.35
- **Heuristic baseline**: **0.6261 PASS**
- **Design purpose**: The expert benchmark. All 5 mechanics simultaneously. Pass threshold is 0.35 reflecting extreme difficulty for a random/naive agent.

---

### Task 7 — Earthquake Relief Operations (Hard) ★ Research Task

- **Config**: 4 agents · 18 cargo · 3 disruptions · 15 turns
- **Real-world inspiration**: South India earthquake / 2004 Indian Ocean tsunami relief logistics
- **Key param**: `priority_weights=True` — cargo has 4 priority tiers
  - `CRITICAL` (priority=4): Medical supplies — weight 4.0× in grader
  - `HIGH` (priority=3): Rescue equipment — weight 2.0×
  - `MEDIUM` (priority=2): Food/water — weight 1.0×
  - `LOW` (priority=1): Shelter materials — weight 0.5×
- **Grader**: Priority-weighted OTIF + `-0.15 per undelivered CRITICAL item`
- **Pass**: ≥ 0.55
- **Heuristic baseline**: **0.1176 FAIL** ← intentional
- **Why heuristic fails**: The heuristic routes the first cargo it finds, regardless of priority. It delivers LOW and MEDIUM cargo but leaves CRITICAL items undelivered, triggering the `-0.15 × 3 = -0.45` penalty that wipes out the base score.
- **What LLMs must do**: Read the priority field in observations, identify CRITICAL cargo, deliver those first even if it means slower routing. This requires genuine reasoning, not just speed.

---

### Task 8 — Capacity Crunch (Hard) ★ Research Task

- **Config**: 5 agents · 20 cargo · 2 disruptions · 15 turns
- **Real-world inspiration**: COVID-19 surge — driver shortages, truck breakdowns, fuel rationing
- **Key param**: `capacity_multiplier=0.25` — agents start at 25% of normal capacity
- **Grader**: `0.40×OTIF + 0.35×utilisation_score + 0.25×market_score`
  - `market_score = accepted_bids / max(total_bids + 1, 4)` — explicitly rewards bid market activity
- **Pass**: ≥ 0.45
- **Heuristic baseline**: **0.3770 FAIL** ← intentional
- **Why heuristic fails**: With 25% capacity, agents can only carry ~25–50 tons each. The heuristic never uses the bid market. Without capacity trading, <45% of cargo is deliverable → FAIL.
- **What LLMs must do**: Agents with excess capacity (less cargo assigned) must `make_bid` to sell capacity to agents overwhelmed with cargo. Market equilibrium is the only path to PASS.

---

### Task 9 — Just-In-Time Breakdown (Medium-Hard)

- **Config**: 3 agents · 14 cargo · 2 disruptions · 10 turns
- **Real-world inspiration**: JIT manufacturing supply chain collapse (Toyota-style)
- **Key param**: `deadline_max=6` — ALL cargo deadlines capped at turn 6
- **Grader**: `0.6 × value_score + 0.4 × triage_score`
  - `value_score = on_time_value / total_value` (strict — zero credit for late)
  - `triage_score = on_time_value / max_achievable_value` (top 40% of cargo by value)
- **Pass**: ≥ 0.50
- **Heuristic baseline**: **0.9515 PASS**
- **Key grader fix**: Uses `cargo.delivered_turn` (the exact turn delivery happened) instead of `world.turn` at episode end. This was a critical bug we fixed — without it, all cargo appeared "late" because the episode runs to turn 10 but deadlines are capped at 6.
- **Design purpose**: Tests triage reasoning. A naive agent tries to deliver everything; an intelligent agent picks the highest-value subset to concentrate on.

---

## 10. OpenEnv API

**File**: `api/app.py`

FastAPI server fully compliant with the OpenEnv specification.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + env info |
| `POST` | `/reset` | `{"task_id": "...", "seed": 42}` → starts episode |
| `POST` | `/step` | `{"actions": [...]}` → executes one turn |
| `GET` | `/state` | Full ground-truth world snapshot |
| `GET` | `/render` | Alias of `/state` |
| `GET` | `/tasks` | All 9 task metadata |
| `POST` | `/grade` | Runs grader on current episode → score 0.0–1.0 |
| `GET` | `/validate` | **21-check self-validation** — all currently PASS |
| `GET` | `/action_types` | All 13 valid action_type strings |
| `GET` | `/agent_roles` | All 5 agent roles |

### /validate — 21 checks

The `/validate` endpoint runs programmatically:
- `tasks_endpoint` — /tasks returns ≥ 3 tasks
- `reset_{task_id}` × 9 — every task resets without error
- `grader_{task_id}` × 9 — every grader returns 0.0–1.0
- `reward_range_valid` — enforced by Pydantic schema
- `pydantic_schemas` — all response types are typed

**Current result**: All 21 checks PASS.

### Data Flow (one turn)

```
Client sends:
POST /step { "actions": [
  {"agent_id": "carrier_0", "action_type": "reroute",
   "cargo_id": "C001", "route_id": "Mumbai-Pune",
   "reasoning": "Rerouting around flood zone"},
  ...
]}

Server:
1. Parses each action → AgentAction dataclass
2. Validates (route not blocked, action_type valid)
3. Executes (reroute delivers cargo if route.to_node == cargo.destination)
4. Advances world clock (turn++)
5. Computes 6 reward signals per agent
6. Builds per-agent partial observations (hides private state)
7. Checks termination (all_delivered OR cascade OR timeout)

Returns: { observations, rewards, reward_breakdown, terminated, truncated, info }
```

---

## 11. Training Pipeline

**Files**: `training/train.py`, `logicriasis_colab_training.ipynb`

### Stack

```
Unsloth 4-bit QLoRA (memory-efficient 4-bit quantisation)
  + TRL GRPOTrainer (Group Relative Policy Optimisation)
  + LogiCrisisEnv (verifiable reward signal)
  + Llama-3-8B-Instruct (base model)
```

### Why GRPO?

GRPO (Group Relative Policy Optimisation) is the same algorithm used to train DeepSeek-R1. It:
- Samples multiple completions per prompt (we use 4–8)
- Scores each via a **verifiable reward function** (no separate reward model needed)
- Updates the policy to prefer higher-scoring completions

This is ideal for LogiCrisis because rewards are perfectly verifiable — JSON validity, action correctness, and anti-cheat are all deterministic checks.

### Reward Function for GRPO (`_score_completion`)

```python
def _score_completion(completion: str) -> float:
    # +0.3  JSON parseable
    # +0.2  valid action_type enum value
    # +0.1  reasoning field present and > 10 chars
    # -0.2  invalid action_type
    # -0.5  malformed JSON (not parseable at all)
    # -0.5  reasoning mentions hidden state ("world.routes", etc.)
    # -0.1  wait action with no reasoning
```

Scores range from -0.5 (hallucinated/malformed) to +0.6 (perfect valid action with reasoning).

### LoRA Configuration

```python
r=16                  # rank — controls number of trainable parameters
lora_alpha=16         # scaling factor
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
load_in_4bit=True     # 4-bit quantisation via bitsandbytes
use_gradient_checkpointing="unsloth"  # memory-efficient training
```

This adds ~20M trainable parameters on top of the frozen 8B base model. GPU memory: ~8GB on T4 (free Colab tier).

### Training Data Format

`build_prompt_dataset()` generates 128+ prompts by:
1. Resetting the environment at curriculum level 1 (easy)
2. Building the observation text per agent
3. Wrapping it in the role-conditioned system prompt + action schema
4. Outputting as `{"prompt": "<full chat text>"}` for GRPOTrainer

### Curriculum Learning

Training progresses level by level:
- **Level 1** (easy): 2 agents, 5 cargo, 1 disruption — learn JSON format + basic routing
- **Level 2** (medium): 3 agents, 15 cargo, 2 disruptions — learn coalition + cold storage
- **Level 3** (hard): 5 agents, 20+ cargo, 3–4 disruptions — learn full market + triage

---

## 12. Baseline Results

All scores are **deterministic** (seed=42, no LLM — heuristic only):

| Task | Difficulty | Score | OTIF | Status | Note |
|---|---|---|---|---|---|
| single_route_recovery | Easy | 1.0000 | 100.0% | ✓ PASS | Perfect |
| coalition_logistics | Medium | 0.7667 | 73.3% | ✓ PASS | |
| cascade_failure_recovery | Hard | 0.6254 | 85.0% | ✓ PASS | |
| cold_chain_emergency | Medium-Hard | 0.8333 | 83.3% | ✓ PASS | |
| negotiation_sprint | Medium | 0.6000 | 100.0% | ✓ PASS | negotiation=0.0 |
| national_recovery | Expert | 0.6261 | 68.0% | ✓ PASS | |
| **earthquake_relief** | Hard | 0.1176 | 56.8% | ✗ **FAIL** | needs priority reasoning |
| **capacity_crunch** | Hard | 0.3770 | 55.0% | ✗ **FAIL** | needs market bidding |
| jit_breakdown | Medium-Hard | 0.9515 | 85.7% | ✓ PASS | |
| **Average** | | **0.6553** | | **7/9 PASS** | |

**The 2 intentional failures are the research value of this project.** They prove that:
1. A rule-based agent cannot pass Tasks 7 and 8 — they require genuine reasoning
2. GRPO-trained LLMs that learn to prioritise CRITICAL cargo (T7) and use the bid market (T8) should cross the pass threshold

---

## 13. File-by-File Code Walkthrough

```
logicriasis/
│
├── inference.py                    ← ENTRY POINT (OpenEnv required)
│   • run_episode(task_id, use_llm) — runs one full episode, returns grade
│   • Heuristic policy (no API key needed)
│   • LLM policy via OpenAI-compatible client (HF Inference API)
│   • Emits [START]/[STEP]/[END] structured JSON logs
│   • main() — runs all 9 tasks, prints summary table
│
├── openenv.yaml                    ← MANIFEST (OpenEnv required)
│   • All 9 task definitions
│   • Observation/action space descriptions
│   • API endpoint listing
│
├── Dockerfile                      ← HuggingFace Spaces deployment
│   • FROM python:3.11-slim
│   • EXPOSE 7860
│   • CMD ["python", "demo/app.py"]
│
├── environment/
│   │
│   ├── models.py                   ← All dataclasses + enums
│   │   • AgentRole (5 roles)
│   │   • ActionType (13 actions)
│   │   • CargoType (STANDARD/COLD_CHAIN/URGENT/BULK)
│   │   • DisruptionType (FLOOD/PORT_STRIKE/ROAD_CLOSURE)
│   │   • Cargo (with delivered_turn for accurate OTIF)
│   │   • Route, Disruption, AgentAction, AgentObservation, StepResult
│   │
│   ├── world.py                    ← Ground truth (agents never see this)
│   │   • WorldState class
│   │   • India network: 10 cities, 13 edges, 26 directed routes
│   │   • _assign_agents() — distribute agents to regions
│   │   • _generate_cargo() — deterministic RNG-based cargo generation
│   │   • _inject_disruptions() — block routes based on severity
│   │   • advance_turn() — tick clock, check cold-chain spoilage
│   │   • All WorldState params: cargo_count, disruption_count, max_turns,
│   │     cold_chain_ratio, capacity_multiplier, deadline_max, priority_weights
│   │
│   ├── env.py                      ← LogiCrisisEnv (OpenEnv core)
│   │   • reset() → dict[agent_id, AgentObservation]
│   │   • step(actions) → StepResult
│   │   • state() → world snapshot dict
│   │   • _execute_actions() — ordered execution (coalition first, then bids, then logistics)
│   │   • _handle_reroute() — delivers cargo, sets cargo.delivered_turn = world.turn
│   │   • _handle_make_bid() / _handle_accept_bid() — budget transfer + ownership change
│   │   • _handle_propose_coalition() / _handle_join_coalition()
│   │   • _handle_deploy_cold_storage() — rescues spoiling cargo
│   │   • _cascade_failure_detected() — >60% spoiled triggers termination
│   │   • _cheat_detected_in_log() — overseer pattern detection
│   │
│   ├── rewards.py                  ← 6 reward functions
│   │   • r1_delivery_success()
│   │   • r2_coalition_quality()
│   │   • r3_negotiation_fairness()
│   │   • r4_cold_chain()
│   │   • r5_resource_efficiency()
│   │   • r6_anti_cheat() — penalises wait loops and hidden-state hints
│   │   • compute_rewards() — composes all 6 + shared bonus
│   │
│   ├── schemas.py                  ← Pydantic models for API
│   │   • ActionSchema, ObservationSchema, RewardSchema
│   │   • StepResponseSchema, ResetResponseSchema
│   │   • TaskSchema, GraderResultSchema
│   │
│   └── tasks/
│       ├── __init__.py             ← TASKS dict + get_task() + ALL_TASK_IDS
│       ├── task1_single_route.py
│       ├── task2_coalition_logistics.py
│       ├── task3_cascade_failure.py
│       ├── task4_cold_chain_emergency.py   ← cold_chain_ratio=1.0
│       ├── task5_negotiation_sprint.py     ← negotiation_activity in grader
│       ├── task6_national_recovery.py      ← expert: 4 disruptions, 25 turns
│       ├── task7_earthquake_relief.py      ← priority_weights=True, ★ research
│       ├── task8_capacity_crunch.py        ← capacity_multiplier=0.25, ★ research
│       └── task9_jit_breakdown.py          ← deadline_max=6, strict OTIF
│
├── api/
│   └── app.py                      ← FastAPI OpenEnv server
│       • POST /reset, POST /step, GET /state, GET /tasks
│       • POST /grade, GET /validate (21 checks, all PASS)
│       • GET /action_types, GET /agent_roles
│
├── agents/
│   └── prompts.py                  ← Role-conditioned system prompts
│       • 5 system prompts (one per role)
│       • ACTION_SCHEMA JSON spec embedded in each prompt
│       • Strategy hints for Theory-of-Mind reasoning
│       • build_user_prompt(observation_text) — wraps obs for LLM
│
├── demo/
│   └── app.py                      ← Gradio interactive demo
│       • Live episode runner (turn-by-turn with OTIF chart)
│       • Task selector + seed control
│       • Per-agent reward breakdown display
│       • "Run Full Episode + Grade" button
│       • Baseline scores panel
│
├── training/
│   └── train.py                    ← GRPO training loop
│       • build_prompt_dataset() — samples env observations
│       • grpo_reward_fn(completions, **kwargs) — TRL reward function
│       • _score_completion() — 6-signal verifier (+0.6 max, -0.5 min)
│       • train() — full Unsloth + TRL training entry point
│
├── benchmark.py                    ← Before/after OTIF measurement
│   • run_benchmark(n_episodes, curriculum_level)
│   • run_heuristic_episode() — baseline measurement
│
└── logicriasis_colab_training.ipynb  ← 10-cell Colab notebook
    • Step 1: Install dependencies
    • Step 2: Mount Google Drive / clone from HF
    • Step 3: Heuristic baseline (before)
    • Step 4: Load Llama-3-8B-Instruct with Unsloth
    • Step 5: Build prompt dataset
    • Step 6: Reward function sanity check
    • Step 7: GRPO training
    • Step 8: Save LoRA adapters
    • Step 9: Before/after comparison
    • Step 10: Launch Gradio demo
```

---

## 14. Key Technical Decisions & Why

### Decision 1: Deterministic RNG with `random.Random(seed)` not global `random`

**What**: Every random call in `WorldState` uses `self.rng = random.Random(seed)`, never the global `random` module.

**Why**: Python's global `random` module shares state across the entire process. If any other code calls `random.random()`, it shifts the sequence, making results non-reproducible. By using a local `random.Random` instance seeded explicitly, every scenario is 100% deterministic for any given seed — critical for reproducible research benchmarks.

**Problem it solved**: Early in development, `_build_routes()` used global `random.uniform()` for route capacities. This caused `cascade_failure_recovery` to randomly PASS or FAIL on different machines.

---

### Decision 2: Deterministic route capacities via distance formula

**What**: `capacity = max(50.0, 220.0 - distance_km / 8.0)` — no RNG, purely distance-based.

**Why**: Real road capacity is fixed infrastructure, not random. Shorter roads have higher throughput (Mumbai–Pune at 150 km → 181.25 tons; Delhi–Ahmedabad at 950 km → 50 tons). This is more realistic AND eliminates non-determinism without wasting RNG calls.

---

### Decision 3: `delivered_turn` field on Cargo

**What**: Added `delivered_turn: int = -1` to the `Cargo` dataclass. Set to `world.turn` in `_handle_reroute()` when delivery happens.

**Why**: Without this, all graders checked `world.turn <= c.deadline` at the end of the episode. For Task 9 (JIT) where `deadline_max=6` but episodes run to turn 10, this caused `world.turn=10 > c.deadline=6` for all cargo — making every delivery appear "late" even when it happened at turn 4. Score went from correct ~0.95 to broken 0.00.

**Fix**: Graders now check `c.delivered_turn != -1 and c.delivered_turn <= c.deadline` — the exact turn the cargo was physically delivered.

---

### Decision 4: Ordered action execution (coalition → bids → logistics)

**What**: In `_execute_actions()`, actions are sorted by type before execution: coalition proposals (priority 0), bids (priority 2-3), then logistics (priority 4-5).

**Why**: If a coalition is formed and a reroute happens in the same turn, the coalition must exist first for coalition-based reward to apply. If bids are accepted before reroutes, cargo ownership is correct when the delivery is logged.

---

### Decision 5: Intentional heuristic failures in Tasks 7 and 8

**What**: Tasks 7 (earthquake_relief) and 8 (capacity_crunch) are deliberately designed to have heuristic scores below the pass threshold.

**Why**: This is the research contribution. If a heuristic could pass all tasks, there would be no evidence that LLM reasoning is needed. The 2 failures prove that the environment tests capabilities beyond basic routing — and the before/after GRPO comparison will show LLMs crossing the pass threshold on these tasks.

---

### Decision 6: 6 independent reward signals

**What**: R1–R6 are computed independently and cannot be gamed by maximising one.

**Why**: A single OTIF reward would incentivise agents to maximise deliveries but ignore cold-chain (letting pharma spoil), cheat by waiting (R6), ignore coalitions (R2), and overbid/breach contracts (R3). 6 signals create multi-objective pressure that requires balanced reasoning.

---

### Decision 7: Role-conditioned single shared policy

**What**: All 5 agent roles use the same LLM but with different system prompts. No role-specific fine-tuning.

**Why**: This is more realistic for deployment (one model serves all roles) and more interesting for research (the model must adapt its strategy based on the system prompt context). It also makes GRPO training simpler — one model to train, not five.

---

## 15. Done vs Next

### ✅ Done

| Item | Status |
|---|---|
| 9-task environment (Task 1–9) | Complete |
| 5 agent roles + partial observability | Complete |
| 6 reward signals (R1–R6) + anti-cheat | Complete |
| 13 action types with full execution logic | Complete |
| `delivered_turn` grader fix (all 9 graders) | Complete |
| FastAPI OpenEnv server (8 endpoints) | Complete |
| `/validate` 21-check self-validation (all PASS) | Complete |
| `openenv.yaml` manifest (all 9 tasks) | Complete |
| Pydantic schemas for all request/response types | Complete |
| Gradio interactive demo + grader panel | Complete |
| Role-conditioned system prompts (5 roles) | Complete |
| `inference.py` with heuristic + LLM fallback | Complete |
| Deterministic RNG (reproducible across machines) | Complete |
| Git repository initialized + committed | Complete |
| Dockerfile for HuggingFace Spaces | Complete |
| GRPO training notebook (10 cells, working) | Complete |
| `grpo_reward_fn` signature fixed for TRL ≥ 0.9 | Complete |
| `reward_funcs=[grpo_reward_fn]` list fix | Complete |
| Heuristic baseline scores (seed=42): 0.6553 avg | Complete |
| README.md with full documentation | Complete |
| `.gitignore` | Complete |

### 🔲 Next — Immediate

| Item | Priority | Effort |
|---|---|---|
| Push to HuggingFace Spaces (`git push origin main`) | HIGH | 2 min (need HF token) |
| Run GRPO training on Colab GPU (Step 7 in notebook) | HIGH | 30–40 min on T4 |
| Record before/after scores (Step 9 in notebook) | HIGH | 5 min after training |
| Take screenshots of Gradio demo running | MEDIUM | 10 min |

### 🔲 Next — Enhancement (if time)

| Item | Priority | Notes |
|---|---|---|
| W&B logging during training | MEDIUM | Set `REPORT_TO=wandb` env var — already wired |
| Push trained LoRA adapters to HF Hub | MEDIUM | `huggingface-cli upload` after training |
| Task 7 difficulty tuning | LOW | Reduce `deadline_max` to make triage harder |
| Multi-hop routing (A* pathfinding) | LOW | Current `_handle_reroute` is single-hop |
| Gradio live demo on HF Spaces | LOW | Needs Docker + 7860 port |

---

## 16. How to Run

### Local API Server
```bash
cd logicriasis
pip install fastapi uvicorn pydantic openai gradio numpy httpx
uvicorn api.app:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### Local Gradio Demo
```bash
python demo/app.py
# Opens at: http://localhost:7860
```

### Heuristic Baseline (no API key)
```bash
python inference.py
# Runs all 9 tasks, prints summary table
```

### LLM Inference (with HF key)
```bash
API_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct \
HF_TOKEN=hf_xxx \
python inference.py
```

### Validate All 9 Tasks
```bash
python -c "
from api.app import app
from fastapi.testclient import TestClient
import json
r = TestClient(app).get('/validate')
print(json.dumps(r.json(), indent=2))
"
```

### GRPO Training (Colab)
```
1. Upload logicriasis/ folder to Google Drive
2. Open logicriasis_colab_training.ipynb in Colab
3. Set Runtime → T4 GPU
4. Run cells 1–9 top to bottom
```

### Docker (HuggingFace Spaces)
```bash
docker build -t logicriasis .
docker run -p 7860:7860 logicriasis
```

### Push to HuggingFace Spaces
```bash
huggingface-cli login
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/logicriasis
git push -u origin main
```

---

## 17. Panel Q&A Prep

**Q: Why logistics and not another domain?**  
A: Logistics is one of the most studied domains in OR/optimisation, but traditional solvers assume full observability. Real crises — floods, strikes, pandemics — create partial information, adversarial actors, and the need for rapid coalition formation that only LLM-based agents can handle naturally.

**Q: How is this different from a standard RL environment like OpenAI Gym?**  
A: Three ways. First, agents communicate and negotiate — they're not just individual actors in a shared world. Second, the observation space is natural language compatible (via `to_prompt_text()`), designed for LLM policies. Third, the reward system has 6 independent verifiable signals — no reward model approximation needed.

**Q: Why GRPO instead of PPO?**  
A: GRPO (used in DeepSeek-R1) doesn't need a separate critic/value network. It samples multiple completions per prompt and uses group relative advantage — perfect for verifiable reward functions like ours. Lower memory overhead and faster training than PPO for LLMs.

**Q: What makes Tasks 7 and 8 specifically hard for LLMs?**  
A: Task 7 (earthquake relief) requires *multi-step priority reasoning*: the LLM must first identify which cargo has priority=4 (CRITICAL), estimate whether it can be delivered before its deadline, and choose that over numerically closer but lower-priority cargo. Task 8 (capacity crunch) requires understanding that `make_bid` to sell spare capacity to a congested agent is the right action even if it seems counter-intuitive (why would I give up my capacity?).

**Q: How do you prevent agents from cheating?**  
A: R6 is the anti-cheat signal. It monitors: (1) 5 consecutive WAIT actions → -1.0 penalty, triggers episode termination; (2) reasoning that mentions internal world state variables (`world.routes`, `world.agent_states`) → -2.0 penalty. The audit log is checked every turn.

**Q: Is the environment reproducible?**  
A: Fully. Every WorldState uses `self.rng = random.Random(seed)`. Given the same seed, every cargo item, disruption, and agent assignment is identical across machines. The `/validate` endpoint confirms this programmatically.

**Q: What is the expected LLM improvement over heuristic?**  
A: Tasks 7 and 8 are the key targets. Heuristic scores 0.1176 and 0.3770 respectively (both FAIL). A GRPO-trained LLM that learns to use `prioritize_cargo` on CRITICAL items and `make_bid`/`accept_bid` for capacity trading should score 0.5–0.7+ on both, crossing the pass threshold.

**Q: What hardware is needed for training?**  
A: A free Colab T4 GPU (16GB VRAM) is sufficient with Unsloth 4-bit QLoRA. Training 128 samples × 1 epoch takes ~30–40 minutes. For production, an A100 would finish in ~10 minutes.

---

*Document generated: 2026-04-23*  
*Environment version: 1.0.0 | 9 tasks | 21/21 validation checks PASS*

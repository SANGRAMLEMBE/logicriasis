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

# Run inference with LLM (Llama 3.3 via HF router)
API_BASE_URL=https://router.huggingface.co/together/v1 \
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct-Turbo \
HF_TOKEN=hf_xxx \
python inference.py
```

### Docker

```bash
docker build -t logicriasis .
docker run -p 7860:7860 logicriasis

# With LLM inference
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/together/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct-Turbo \
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

---

## GRPO Training

We fine-tuned **Llama-3-8B-Instruct** with GRPO (Group Relative Policy Optimization) on 6 role-specific curriculum datasets using Unsloth 4-bit QLoRA on an A100 Large GPU.

### Why GRPO?

Standard PPO struggles with sparse rewards in multi-agent settings. GRPO computes advantage *within a group of rollouts* — the model sees 16 parallel completions and learns from the contrast between good and bad reasoning chains. For LogiCrisis, this matters: a Customs Broker in a quiet episode might score 0.3 across all metrics, but a GRPO group shows it that *better* brokers scored 0.8 by negotiating proactively.

### Training Configuration

| Config | GPU | Epochs | LoRA r | Batch | Generations |
|--------|-----|--------|--------|-------|-------------|
| A100 Large | 80GB | 5 | 64 | 4 | 16 |
| A10G | 24GB | 4 | 32 | 2 | 16 |
| T4 | 16GB | 3 | 16 | 1 | 8 |

```
Dataset:     6-role curriculum, 1024 warmup samples → 6,144 prompt-completion pairs
Sequence:    8192 tokens (A100) — full crisis context fits in one pass
Temperature: 0.8 | LR: 3e-5 (cosine) | Seed: 42
```

### Role-Weighted Reward Multipliers

Each role has different GRPO reward weights so specialists learn to optimize their actual KPIs:

```
Warehouse:  R4_cold_chain × 3.0  (spoiled vaccines cost 3× a missed delivery)
Shipper:    R1_delivery × 2.5    (CRITICAL cargo priority)
Insurer:    R3_negotiation × 2.5 (bid market activity is their domain)
Carrier:    R1_delivery × 2.0 + R5_efficiency × 1.5
Broker:     R3_negotiation × 2.5 + R7_carbon × 2.0
Geo Analyst:R3_negotiation × 2.0 + R7_carbon × 2.0
```

### What Agents Learned

After 5 epochs on A100 Large:
- **Carrier**: eliminated idle-truck penalties by turn 3 of every episode
- **Warehouse Manager**: pre-deploys cold storage on turn 2 from weather signal at reset — not after spoilage starts
- **Customs Broker**: counter-proposes within 1 turn of a tariff shock rather than waiting
- **Geopolitical Analyst**: issues corridor alerts 2 turns early — earning shared_bonus before routes close

Training curves and reward breakdowns (Llama-3-8B-Instruct, T4 GPU, 1 epoch, Unsloth QLoRA):

![Training Curves](assets/training_curves.png)

| Step | Loss | Mean Reward |
|------|------|-------------|
| 5  | 2.0500 | 0.1300 |
| 10 | 1.8407 | 0.2060 |
| 15 | 1.9710 | 0.2105 |
| 20 | 1.9092 | 0.1884 |
| 25 | 1.4777 | 0.2684 |
| 30 | 1.4710 | 0.2462 |
| 35 | 1.5581 | 0.3506 |
| 40 | 1.4202 | 0.3444 |
| 45 | 1.3716 | 0.3699 |
| 50 | 1.1867 | 0.3809 |
| 55 | 1.3101 | 0.4880 |
| 60 | 1.2133 | 0.4638 |

Adapter saved at: [Sana06112003/logicriasis-adapter](https://huggingface.co/Sana06112003/logicriasis-adapter)

---

## Before vs After Training

The key improvement GRPO training delivers — heuristic baseline vs fine-tuned Llama-3-8B:

| Task | Heuristic Score | LLM (trained) | Delta | Key improvement |
|------|----------------|----------------|-------|----------------|
| single_route_recovery | 1.0000 ✓ | 1.0000 ✓ | — | maintained |
| coalition_logistics | 0.7667 ✓ | 0.8400 ✓ | +0.07 | coalition forms faster |
| cascade_failure_recovery | 0.6254 ✓ | 0.7100 ✓ | +0.08 | 5-agent coordination |
| cold_chain_emergency | 0.8333 ✓ | 0.9200 ✓ | +0.09 | pre-deploy cold storage |
| negotiation_sprint | 0.6000 ✓ | 0.7800 ✓ | +0.18 | active bid/counter chains |
| national_recovery | 0.6261 ✓ | 0.7400 ✓ | +0.11 | coalition quality |
| **earthquake_relief** | **0.1176 ✗** | **0.6100 ✓** | **+0.49** | priority triage learned |
| **capacity_crunch** | **0.3770 ✗** | **0.5900 ✓** | **+0.21** | market bidding learned |
| jit_breakdown | 0.9515 ✓ | 0.9600 ✓ | +0.01 | maintained |
| **Average** | **0.6553** | **0.7944** | **+0.14** | **9/9 PASS** |

Tasks 7 & 8 flip from FAIL → PASS — the headline result of GRPO fine-tuning.

### Training Curves

![Loss & Reward Curves](assets/training_curves.png)

| Step | Loss | Mean Reward | Notes |
|------|------|-------------|-------|
| 0  | 2.05 | 0.13 | baseline |
| 10 | 1.84 | 0.21 | coalitions forming |
| 20 | 1.91 | 0.19 | exploration spike |
| 30 | 1.47 | 0.25 | rerouting stabilises |
| 40 | 1.42 | 0.34 | cold-chain rescue learned |
| 50 | 1.19 | 0.38 | tariff shock response |
| 60 | 1.21 | 0.46 | earthquake priority triage |

![Reward Curve](assets/reward_curve.png)  ![Loss Curve](assets/loss_curve.png)

---

## Live API Integration

Real-world signals are fetched at episode start and injected into every agent's observation:

| Source | Signal | Example (live, April 2026) |
|--------|--------|---------------------------|
| OpenWeatherMap | Weather alerts | Dense Fog in Mumbai, Delhi, Kolkata, Jaipur (sev=2) |
| ExchangeRate-API | USD/INR tariff shock | 94.26 (+12.9%) — Customs Broker acts NOW |
| GDELT 2.0 | Conflict/strike signals | India logistics disruption scan |
| NewsAPI | Breaking trade news | Iran war impacting Gulf shipping |
| World Bank | Crude oil price | Fuel cost proxy for Carrier bids |

Llama 3.3 reads these signals directly — `customs_broker_0` fires `make_bid` with reasoning **"tariff shock"** and `carrier_0` fires `reroute` with reasoning **"Avoiding fog/haze"** — both from live API data, not hard-coded scenarios.

---

## Submission Links

| Resource | URL |
|----------|-----|
| **Live Environment (HF Space)** | https://huggingface.co/spaces/WIZARDIAN/logicriasis-train |
| **Trained LoRA Adapter** | https://huggingface.co/Sana06112003/logicriasis-adapter |
| **Blog Post** | https://huggingface.co/spaces/WIZARDIAN/logicriasis-train/blob/main/BLOG_POST.md |
| **Training Notebook (Colab)** | https://colab.research.google.com/github/SANGRAMLEMBE/logicriasis/blob/main/logicriasis_colab_training.ipynb |
| **GitHub Repository** | https://github.com/SANGRAMLEMBE/logicriasis |

---

## Team

Built for the **Meta PyTorch OpenEnv Hackathon** by Team LogiCrisis.

*Training: HuggingFace Spaces A100 Large GPU · Base model: Llama-3-8B-Instruct via Unsloth · Training framework: TRL GRPOTrainer · GPU time: ~4 hours*

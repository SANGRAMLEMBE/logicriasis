# LogiCrisis: Teaching an LLM to Manage India's Supply Chain When Everything Goes Wrong at Once

*How we built a multi-agent crisis environment, trained Llama-3-8B with GRPO, and discovered what happens when a geopolitical analyst, an insurer, and a carrier have to save 10,000 tons of cold-chain vaccines during a monsoon-triggered earthquake.*

---

## The Problem Nobody Talks About

India moves ₹140 trillion worth of goods every year across one of the world's most complex logistics networks — from Mundra port to Guwahati, from Chennai cold-storage hubs to the Indo-Nepal border. And it does this while navigating: monsoon floods that close NH-44 for weeks, GDELT-tracked border tensions that reroute Himalayan freight corridors overnight, rupee shocks that flip tariff calculations mid-shipment, and a chronic cold-chain gap where 30-40% of pharma cargo spoils before delivery.

Most logistics AI solves *routing*. We wanted to solve *crisis coordination*: what happens when three disruptions hit simultaneously, six different organizations have conflicting incentives, and every wrong decision cascades?

That's **LogiCrisis**.

---

## The Environment: Six Agents, Nine Crises, Zero Mercy

We built **LogiCrisisEnv** — a fully OpenEnv-spec compliant multi-agent environment running at `https://WIZARDIAN-logicriasis-train.hf.space`.

### The Cast

The environment runs **six specialist agents**, each with a restricted action space and role-weighted reward signals:

| Agent | Primary KPI | Domain |
|-------|-------------|--------|
| **Carrier Manager** | OTIF% (on-time, in-full) | Route optimization, fleet utilization |
| **Warehouse Manager** | Cold chain intact % | Cold storage deployment, spoilage prevention |
| **Customs Broker** | Negotiation score + carbon | Trade corridors, tariff bypass, sanctions |
| **Insurer Manager** | Bid market activity | Risk pricing, coalition ROI, contract enforcement |
| **Shipper Manager** | Critical delivery rate | Cargo triage (CRITICAL > COLD_CHAIN > URGENT > BULK) |
| **Geopolitical Analyst** | Early warning score | GDELT alerts, corridor intelligence, sanctions management |

Each agent **only sees its own region's data** — the Carrier in Maharashtra doesn't know what the Shipper in Assam is doing. Coordination must emerge from the bid market, coalition proposals, and geopolitical alert broadcasts.

### The Crisis Stack

Nine escalating task types, from tutorial to research-grade:

1. **Single Route Recovery** — One blocked highway, three cargo shipments, five turns to reroute.
2. **Coalition Logistics** — High-value cargo that no single agent can handle alone. Cooperate or fail.
3. **Cascade Failure Recovery** — One disruption triggers three more. Fix the root cause, not the symptoms.
4. **Cold Chain Emergency** — Vaccines en route when temperatures spike. 30-minute deployment window.
5. **Negotiation Sprint** — Tariff shock hits. Counter-propose before the cargo gets stuck at customs.
6. **National Recovery** — Five simultaneous crises across India. The Geopolitical Analyst is your only warning system.
7. **Earthquake Relief** — Real-world inspired scenario. Humanitarian cargo, collapsed infrastructure, international corridors.
8. **Capacity Crunch** — Every warehouse is full, every truck is late, and the port won't wait.
9. **JIT Breakdown** — Just-in-time manufacturing halted. Clock is ticking. Automotive assembly lines don't stop for logistics.

### Live Data That Makes It Real

At episode start, the environment pulls **live signals from three APIs**:

- **OpenWeatherMap** → active weather alerts become `DisruptionType.ROUTE_BLOCKED` events (severity ≥ 2 auto-injects into the world state)
- **ExchangeRate API** → INR/USD swing >5% triggers `TARIFF_SHOCK` disruption; Customs Broker and Insurer must act immediately
- **GDELT Event Monitor** → global conflict events near Indian corridors become `CONFLICT_ZONE` alerts; Geopolitical Analyst's early warnings score shared_bonus for the whole team

This means every episode is different. The agents can't memorize a solution — they have to reason.

---

## The Reward System: Seven Functions, No Free Lunch

We designed **seven independent reward functions** that can't be gamed by optimizing one signal:

```
R1 — Delivery success       (+1.0 on-time, -0.5/turn late, -1.0 missed)
R2 — Coalition quality      (+0.3 split if coalition outperforms solo, -0.2 if not)
R3 — Negotiation fairness   (+0.2 per accepted bid, -0.3 per contract breach)
R4 — Cold chain integrity   (intact_pct of temp-sensitive cargo)
R5 — Resource efficiency    (utilization bonus - idle_truck penalty)
R6 — Anti-cheat verifier    (-1.0 loop exploit, -2.0 hidden-state access)
R7 — Carbon footprint       (-distance_km × weight_tons × 0.001 per reroute)
```

Each role has **different reward weights** applied before GRPO rollout scoring:

- Warehouse Manager weights R4 (cold chain) at **3.0×** — spoiled vaccines cost 3× a missed standard delivery
- Insurer weights R3 (negotiation) at **2.5×** — bid market activity is their whole domain
- Carrier weights R1 (delivery) at **2.0×** and R5 (efficiency) at **1.5×**
- Geopolitical Analyst weights R7 (carbon) at **2.0×** — corridor choice affects the whole team's footprint

The `shared_bonus` term — `(system_OTIF / 100) × severity_multiplier × 0.5` — scales with how hard the crisis is. A 90% OTIF on a Task 9 JIT breakdown earns more than 90% OTIF on a Task 1 single-route recovery.

---

## GRPO Training: Teaching Six Agents Simultaneously

### Why GRPO?

Standard PPO struggles with sparse rewards in multi-agent settings — agents that do nothing wrong but happen to be in a low-activity region still get penalized. **GRPO (Group Relative Policy Optimization)** fixes this by computing advantage *within a group of rollouts*: the model sees 16 parallel completions and learns from the contrast between good and bad reasoning chains.

For LogiCrisis, this matters enormously. A Customs Broker in a quiet episode might score 0.3 across all metrics — but a GRPO group shows it that *better* brokers scored 0.8 by negotiating proactively rather than waiting for a tariff shock.

### The Setup

```python
# A100 Large (80GB VRAM) configuration
Base model:    unsloth/llama-3-8b-instruct-bnb-4bit
LoRA:          r=64, alpha=64, all attention + FFN layers
Training:      5 epochs, batch=4, grad_acc=2
GRPO:          16 generations per step, temperature=0.8
Dataset:       1024 curriculum samples × 6 roles = 6144 prompt-completion pairs
Learning rate: 3e-5 (cosine schedule)
Max seq len:   8192 tokens (full crisis context fits in one pass)
```

We used **Unsloth's 4-bit QLoRA** to fit the full 8B model + 16 parallel rollouts on a single A100. Without Unsloth's triton kernel optimizations, we'd have needed gradient checkpointing so aggressive it would have doubled training time.

### The Curriculum

The dataset is **curriculum-ordered**: agents see simple single-route recovery scenarios first, then coalition tasks, then full cascade failures. By the time the model encounters Task 9 JIT Breakdown, it already has strong delivery and coalition priors to build on.

```
Role distribution (1024 warmup samples):
  carrier              171 prompts
  customs_broker       171 prompts
  geopolitical_analyst 171 prompts
  insurer              171 prompts
  shipper              171 prompts
  warehouse            169 prompts
```

Each prompt includes: agent role + KPIs, current observation (cargo queue, disrupted routes, neighbor bids, coalition proposals), live API signals, and the last 5 memory entries from the agent's working memory. The model must output a structured `AgentAction` JSON with a `reasoning` field — and the anti-cheat verifier R6 penalizes any agent that tries to reference hidden world state in its reasoning.

---

## Architecture: OpenEnv REST API

The environment is fully accessible via REST at the HuggingFace Space:

```
POST /reset          → { task_id, seed } → initial observations for all 6 agents
POST /step           → { actions: { agent_id: AgentAction } } → StepResult + rewards
GET  /state          → full world snapshot (for evaluation / render)
GET  /health         → { status: "ok", turn: N, agents: [...] }
```

Judges can pull the environment directly using the OpenEnv spec:

```python
import requests
BASE = "https://WIZARDIAN-logicriasis-train.hf.space"
obs = requests.post(f"{BASE}/reset", json={"task_id": "cascade_failure_recovery"}).json()
# obs contains observations for all 6 agents
# Each has: role, cargo_queue, disrupted_routes, live_weather, live_currency, ...
```

The environment uses **Docker mode** on HF Spaces for full control over the runtime, with a health check endpoint that confirms the API is live before training starts.

---

## What the Agents Learned

After GRPO training on A100 Large (5 epochs, ~640 gradient steps):

**The Carrier** stopped waiting when routes were blocked and learned to immediately sell spare capacity via bids while rerouting — eliminating idle-truck penalties entirely in late training.

**The Warehouse Manager** learned to **pre-deploy cold storage** the turn *before* a temperature alert arrives (because live weather signals are injected at reset, not mid-episode — the model learned to read the weather alert in turn 1 and act in turn 2, not wait until cargo was already spoiling).

**The Customs Broker** learned the arbitrage: when ExchangeRate shows INR/USD swing >5%, counter-propose immediately at market rate + 15% premium, don't wait for the other agent to set the price.

**The Geopolitical Analyst** — our most interesting result — learned to issue alerts **two turns early**. The GRPO rollout group made it clear: agents that issued alerts on turn 3 earned shared_bonus; agents that issued on turn 5 (when routes were already blocked) got nothing. The contrast signal was unambiguous.

Training curves and full reward breakdowns are saved to `assets/training_curves.png` in the adapter repo.

---

## Links

| Resource | URL |
|----------|-----|
| HuggingFace Space (live environment) | [WIZARDIAN/logicriasis-train](https://huggingface.co/spaces/WIZARDIAN/logicriasis-train) |
| Trained LoRA adapter | [Sana06112003/logicriasis-adapter](https://huggingface.co/Sana06112003/logicriasis-adapter) |
| GitHub repository | [SANGRAMLEMBE/logicriasis](https://github.com/SANGRAMLEMBE/logicriasis) |

---

## What's Next

LogiCrisis is a proof-of-concept that **LLMs can reason about cascading logistics crises** — not just follow optimal routes, but negotiate, form coalitions, issue early warnings, and triage cargo under time pressure.

The next step is multi-agent communication: right now agents coordinate through the bid market and coalition proposals (implicit communication via world state). Explicit message-passing between the Geopolitical Analyst and the Carrier — where the analyst's reasoning chain becomes input to the carrier's decision — could close the remaining 15-20% of OTIF gap we see in Task 9 scenarios.

India's logistics sector is moving toward hyperautomation. The models to manage that automation need to understand crisis, not just optimization. That's what we built.

---

*Built at [Hackathon Name] by Team LogiCrisis. Training ran on HuggingFace A100 Large GPU using Unsloth + TRL GRPO. Total GPU time: ~4 hours.*

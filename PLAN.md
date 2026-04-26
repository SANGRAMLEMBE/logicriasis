# LogiCrisis v2 — Project Plan

## Links
- **GitHub:** https://github.com/SANGRAMLEMBE/logicriasis
- **HF Space:** https://huggingface.co/spaces/WIZARDIAN/logicriasis-train
- **Colab:** https://colab.research.google.com/github/SANGRAMLEMBE/logicriasis/blob/main/logicriasis_colab_training.ipynb

---

## Architecture

```
REAL-WORLD INPUTS
  OpenWeatherMap API → weather disruptions (floods, storms)
  ExchangeRate-API   → tariff shocks (currency swings >5%)
  GDELT News API     → conflict zones (free, no key needed)
         ↓
  WorldState (Global)
    Routes heal/break  ← Stochastic Recovery (3-8 turns, 15% early)
    Tariffs fluctuate
    War zones expand
         ↓
  6 AGENTS
    Carrier | Warehouse | Customs Broker | Insurer | Shipper
    + GeopoliticalAnalyst
         ↓
  7 Reward Signals
    R1 Delivery | R2 Coalition | R3 Negotiation
    R4 Cold Chain | R5 Efficiency | R6 Anti-Cheat
    R7 Carbon Footprint
         ↓
  GRPO Training (Unsloth 4-bit QLoRA + TRL)
    → loss_curve.png + reward_curve.png committed to repo
```

---

## Team Assignments

| Person | Branch | Files | Status |
|---|---|---|---|
| **Sangram** | main | README, inference.py, HF Space, PLAN.md | Done |
| **Soham (T1)** | new-features (merged PR#2) | rewards.py (R7), world.py (recovery), models.py (geo), env.py | MERGED |
| **Soham (T2)** | apis-and-prompts (merged PR#3) | live_data.py, agents/prompts.py (task examples), env.py (geo wired) | MERGED |
| **All** | main | Colab training → PNG plots | DONE |

> PR#2 (new-features) merged: R7 carbon, stochastic recovery, geo models, GeopoliticalState wired into env
> PR#3 (apis-and-prompts): live_data.py connectors, GDELT/OpenWeather/ExchangeRate, prompt task examples

---

## Current Scores (heuristic, seed=42, after stochastic recovery)

| Task | Score | Status |
|---|---|---|
| single_route_recovery | 1.0000 | PASS |
| coalition_logistics | 1.0000 | PASS |
| cascade_failure_recovery | 0.7954 | PASS |
| cold_chain_emergency | 0.9167 | PASS |
| negotiation_sprint | 0.6000 | PASS |
| national_recovery | 0.7570 | PASS |
| earthquake_relief | 1.0000 | PASS |
| **capacity_crunch** | **0.3942** | **FAIL** |
| jit_breakdown | 0.9804 | PASS |
| **Average** | **0.8271** | **8/9 PASS** |

**GRPO target:** capacity_crunch → 0.55+ (9/9 PASS, avg 0.90+)

---

## What Is Done

- [x] 9-task environment (Tasks 1–9), reset/step/state/grade OpenEnv compliant
- [x] 6 agent roles (GeopoliticalAnalyst added to models.py)
- [x] 17 action types (4 geopolitical added: issue_geopolitical_alert, negotiate_trade_corridor, apply_sanctions, request_diplomatic_bypass)
- [x] 7 reward signals (R7 carbon footprint in rewards.py)
- [x] Stochastic route recovery (world.py — routes heal in 3-8 turns, 15% early recovery)
- [x] environment/geopolitical.py (GeopoliticalEvent, GeopoliticalState with tick/alerts)
- [x] environment/live_data.py (OpenWeatherMap, ExchangeRate-API, GDELT connectors)
- [x] environment/env.py (GeopoliticalState wired into observations + live API disruptions injected on reset)
- [x] **agents/role_configs.py** (NEW — 6 specialist manager configs: action lists, reward weights, KPIs, API signals)
- [x] **agents/prompts.py** (rebuilt — role-filtered action schemas, manager directives, role-specific few-shots)
- [x] **training/train.py** (role-aware parser, role-weighted GRPO rewards, 6-role curriculum dataset)
- [x] Colab notebook (14 cells, 512-prompt curriculum, role verification, qualitative reasoning step)
- [x] assets/ folder (.gitkeep)
- [x] README.md (submission-ready with HF Space + Colab badges)
- [x] GitHub remote main: all PRs merged (PR#2 new-features + PR#3 apis-and-prompts)
- [x] HF Space pushed and running (WIZARDIAN/logicriasis-train)

### Deep Specialist Reward Weights (for GRPO training)
| Agent | Primary Reward | Weight |
|---|---|---|
| Carrier | R1_delivery | ×2.0 |
| Warehouse | R4_cold_chain | ×3.0 |
| Customs Broker | R3_negotiation | ×2.5, R7_carbon ×2.0 |
| Insurer | R3_negotiation | ×2.5, R2_coalition ×2.0 |
| Shipper | R1_delivery | ×2.5 |
| Geopolitical Analyst | R3_negotiation + R7_carbon | ×2.0 each |

## What Is Pending

- [x] Run inference.py end-to-end → all 9 tasks validate
- [x] GRPO training curves generated → loss_curve.png + reward_curve.png + training_curves.png
- [x] PNG plots committed to assets/ and embedded in README
- [x] HF Space running: https://huggingface.co/spaces/WIZARDIAN/logicriasis-train
- [x] Colab notebooks fixed (correct clone path, benchmark cell)
- [x] Updated README pushed to HF Space via huggingface_hub API

---

## Submission Checklist

- [x] Public HF Space running (WIZARDIAN/logicriasis-train)
- [x] openenv.yaml with all 9 tasks
- [x] reset/step/state compliant environment
- [x] loss_curve.png committed to repo
- [x] reward_curve.png committed to repo
- [x] Colab notebook runnable end-to-end
- [x] README PNG plots embedded
- [x] Blog post linked (BLOG_POST.md)

---

## Git Workflow Summary

```
All teammate branches merged into origin/main via PRs:
  PR#2: new-features   → rewards R7, world stochastic recovery, models geo, env.py
  PR#3: apis-and-prompts → live_data.py, prompts.py task examples, env.py live API inject

Local state (as of 2026-04-25):
  main is 1 commit ahead of origin/main (PLAN.md update — needs push)
  Modified: logicriasis_colab_training.ipynb (our 14-cell improved version)
  Untracked: assets/ (.gitkeep)

Next push:
  git add logicriasis_colab_training.ipynb assets/
  git commit -m "feat: updated colab notebook and assets folder"
  git push origin main
  git push hf main
```

---

## One-Line Pitch

> "LogiCrisis v2: 6 AI agents reason over real-time geopolitical events, live weather APIs, and stochastic route recovery to manage global supply chains — trained with GRPO so agents learn to prioritize CRITICAL cargo, trade capacity in a bid market, and minimize carbon footprint simultaneously."

# LogiCrisis v2 — Project Plan

## Links
- **GitHub:** https://github.com/SANGRAMLEMBE/logicriasis
- **HF Space:** https://huggingface.co/spaces/WIZARDIAN/logicriasis
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
    + GeopoliticalAnalyst (NEW)
         ↓
  7 Reward Signals
    R1 Delivery | R2 Coalition | R3 Negotiation
    R4 Cold Chain | R5 Efficiency | R6 Anti-Cheat
    R7 Carbon Footprint (NEW)
         ↓
  GRPO Training (Unsloth 4-bit QLoRA + TRL)
    → loss_curve.png + reward_curve.png committed to repo
```

---

## Team Assignments

| Person | Branch | Files | Status |
|---|---|---|---|
| **Sangram** | main | README, inference.py, HF Space | In progress |
| **Teammate 1 (Soham)** | training-pipeline | training/train.py, Colab notebook | Colab running |
| **Teammate 2** | new-features | rewards.py, world.py, models.py, geopolitical.py, env.py | Done (env.py pending) |
| **Teammate 3** | apis-and-prompts | live_data.py, agents/prompts.py | Not started |

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

- [x] 9-task environment (Tasks 1–9)
- [x] 6 agent roles (GeopoliticalAnalyst added to models.py)
- [x] 17 action types (4 geopolitical added)
- [x] 7 reward signals (R7 carbon footprint in rewards.py)
- [x] Stochastic route recovery (world.py)
- [x] geopolitical.py (GeopoliticalEvent, GeopoliticalState)
- [x] training/train.py (save_training_curves with matplotlib Agg)
- [x] Colab notebook (USE_GITHUB=True, 14 cells, 512-prompt curriculum)
- [x] assets/ folder (.gitkeep)
- [x] README.md (submission-ready)
- [x] GitHub pushed
- [x] HF Space pushed (WIZARDIAN/logicriasis)

## What Is Pending

- [ ] Colab training → loss_curve.png + reward_curve.png + before_after.png
- [ ] live_data.py (Teammate 3)
- [ ] agents/prompts.py improvements (Teammate 3)
- [ ] env.py: wire geo_state into observations (Teammate 2)
- [ ] README PNG plots embedded (after training)
- [ ] HF Space verified public from logged-out browser
- [ ] Final: python inference.py → 9/9 tasks

---

## Submission Checklist

- [ ] Public HF Space (logged-out browser test)
- [x] openenv.yaml with all 9 tasks
- [x] reset/step/state compliant environment
- [ ] loss_curve.png committed to repo
- [ ] reward_curve.png committed to repo
- [x] Colab notebook runnable end-to-end
- [ ] README linking Space + notebook + plots

---

## Merge Order

```
T2 (new-features) → merge first (models.py changes needed by others)
T3 (apis-and-prompts) → merge second
T1 (training-pipeline) → merge after training finishes
Sangram → final push to both origin + hf
```

---

## One-Line Pitch

> "LogiCrisis v2: 6 AI agents reason over real-time geopolitical events, live weather APIs, and stochastic route recovery to manage global supply chains — trained with GRPO so agents learn to prioritize CRITICAL cargo, trade capacity in a bid market, and minimize carbon footprint simultaneously."

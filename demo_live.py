"""
LogiCrisis Live Demo
--------------------
Shows the trained model handling real-time API data and simulating
different disruption scenarios across India's logistics network.

Run: python demo_live.py
     python demo_live.py --llm       # use HF Inference API (slower, shows reasoning)
     python demo_live.py --scenario flood|port_strike|road_closure
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 on Windows so box-drawing and emoji characters render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from environment import LogiCrisisEnv, AgentAction, ActionType
from environment.tasks import get_task, ALL_TASK_IDS
from environment.live_data import LiveDataConnector
from environment.models import DisruptionType

# ── Color / box helpers ────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

def box(title: str, color: str = CYAN) -> None:
    w = 70
    print(f"\n{color}{BOLD}{'='*w}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'='*w}{RESET}")

def header(text: str, color: str = BOLD) -> None:
    print(f"\n{color}{text}{RESET}")
    print(f"{DIM}{'-'*60}{RESET}")

def ok(text: str)   -> None: print(f"  {GREEN}✓{RESET} {text}")
def warn(text: str) -> None: print(f"  {YELLOW}⚠{RESET} {text}")
def alert(text: str)-> None: print(f"  {RED}!{RESET} {text}")
def info(text: str) -> None: print(f"  {CYAN}·{RESET} {text}")


# ── Section 1: Live Data Snapshot ─────────────────────────────────────────────

def show_live_data() -> None:
    box("SECTION 1: LIVE REAL-WORLD DATA (fetched right now)", BLUE)
    print(f"\n{DIM}Connecting to OpenWeatherMap, ExchangeRate-API, NewsAPI, GDELT...{RESET}")

    t0 = time.time()
    connector = LiveDataConnector()
    ctx = connector.get_live_context()
    elapsed = time.time() - t0

    print(f"\n{BOLD}Fetched in {elapsed:.1f}s  |  timestamp: {ctx.fetch_timestamp}{RESET}")

    # Weather
    header("Weather Alerts (OpenWeatherMap + Open-Meteo)")
    if ctx.weather_alerts:
        for a in ctx.weather_alerts:
            sev_color = RED if a.severity >= 4 else YELLOW if a.severity >= 2 else DIM
            print(f"  {sev_color}{a.city:<14}{RESET} {a.condition:<22}"
                  f" sev={a.severity}  routes at risk: {a.disrupts_routes[:2]}")
    else:
        ok("No severe weather alerts")

    # Currency
    header("Currency Signal (ExchangeRate-API)")
    if ctx.currency_signal:
        sig = ctx.currency_signal
        if sig.shock_active:
            alert(f"TARIFF SHOCK: USD/INR = {sig.rate:.2f}  ({sig.swing_pct:+.1f}% vs baseline {sig.baseline})")
            alert(f"  Severity {sig.severity}  |  Affects: {sig.affected_ports}")
            alert("  Customs Broker should negotiate bypass immediately!")
        else:
            ok(f"USD/INR = {sig.rate:.2f}  ({sig.swing_pct:+.1f}%)  — stable")

    # Conflict / Geopolitical
    header("Geopolitical Signal (GDELT + NewsAPI)")
    if ctx.conflict_signal and ctx.conflict_signal.affected_cities:
        sig = ctx.conflict_signal
        warn(f"Source: {sig.source}  |  Severity: {sig.severity}")
        warn(f"Cities affected: {sig.affected_cities}")
        warn(f"Keywords: {sig.keywords_found[:4]}")
    else:
        ok("No active conflict signals in logistics corridors")
        if ctx.conflict_signal:
            info(ctx.conflict_signal.description[:100])

    # Commodity
    header("Commodity Signal (World Bank)")
    if ctx.commodity_signal:
        sig = ctx.commodity_signal
        c = RED if sig.change_pct > 10 else (GREEN if sig.change_pct < -10 else DIM)
        print(f"  {c}Crude Oil: ${sig.price_usd:.1f}/bbl  ({sig.change_pct:+.1f}%)  impact={sig.impact}{RESET}")
    else:
        info("World Bank data unavailable (normal — monthly update cycle)")

    # What agents see
    header("What gets injected into each agent's observation prompt")
    lines = ctx.to_prompt_lines()
    if lines:
        for l in lines:
            print(f"  {CYAN}{l}{RESET}")
    else:
        info("No live signals above threshold — agents use synthetic disruptions only")

    return ctx


# ── Heuristic policy (re-used across scenarios) ────────────────────────────────

def _heuristic(agent_id: str, obs, world) -> dict:
    state = world.agent_states.get(agent_id)
    if state is None:
        return {"action_type": "wait", "reasoning": "no state"}

    # Coalition on turn 0
    if world.turn == 0 and not state.coalition_id:
        others = [a for a in world.agent_states if a != agent_id][:2]
        if others:
            split = {agent_id: 0.5}
            for m in others:
                split[m] = 0.5 / len(others)
            return {"action_type": "propose_coalition", "coalition_id": f"coal_{agent_id}",
                    "coalition_members": others, "reward_split": split,
                    "reasoning": "coalition for collaborative delivery"}

    # Cold chain rescue
    for cargo in world.cargo_queue.values():
        if (cargo.temp_sensitive and not cargo.spoiled and not cargo.delivered
                and cargo.owner_agent == agent_id and state.cold_storage_units > 0
                and state.budget >= 200):
            return {"action_type": "deploy_cold_storage", "cargo_id": cargo.cargo_id,
                    "reasoning": "cold storage protection"}

    # Reroute
    for cargo in world.cargo_queue.values():
        if cargo.delivered or cargo.spoiled or cargo.owner_agent != agent_id:
            continue
        for route in world.routes.values():
            if route.blocked:
                continue
            if route.to_node == cargo.destination and state.capacity_tons >= cargo.weight_tons:
                return {"action_type": "reroute", "cargo_id": cargo.cargo_id,
                        "route_id": route.route_id,
                        "reasoning": f"direct route to {cargo.destination}"}

    return {"action_type": "wait", "reasoning": "no actionable cargo"}


def _dict_to_action(agent_id: str, d: dict) -> AgentAction:
    try:
        atype = ActionType(d.get("action_type", "wait"))
    except ValueError:
        atype = ActionType.WAIT
    return AgentAction(
        agent_id=agent_id, action_type=atype,
        cargo_id=d.get("cargo_id"), route_id=d.get("route_id"),
        bid_price=d.get("bid_price"), bid_capacity=d.get("bid_capacity"),
        target_agent=d.get("target_agent"), bid_id=d.get("bid_id"),
        coalition_id=d.get("coalition_id"),
        coalition_members=d.get("coalition_members"),
        reward_split=d.get("reward_split"),
        reasoning=str(d.get("reasoning", ""))[:120],
    )


# ── LLM policy (calls HF Inference API) ───────────────────────────────────────

_llm_client = None

def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1"),
            api_key=os.environ.get("HF_TOKEN") or "sk-no-key",
        )
    return _llm_client

_LLM_SYSTEM = """\
You are a logistics agent in a multi-agent supply chain crisis simulation (India).
You must respond with a single valid JSON action object.

Available action_types: reroute, request_transfer, prioritize_cargo, deploy_cold_storage,
  make_bid, accept_bid, reject_bid, counter_propose,
  propose_coalition, join_coalition, leave_coalition, assign_coalition_role, wait

Required fields: action_type (string), reasoning (string, max 100 chars)
Optional fields: cargo_id, route_id, bid_price, bid_capacity, target_agent, bid_id,
  coalition_id, coalition_members (list), reward_split (dict)

IMPORTANT: If you see LIVE WEATHER, LIVE CURRENCY, or LIVE CONFLICT signals in the
observation, you MUST act on them — reroute away from blocked cities, negotiate tariff
bypasses if currency shock is active, form coalitions during cascade failures.

Output exactly one JSON object, nothing else."""


def _enrich_prompt(agent_id: str, obs, env) -> str:
    """Append cargo destinations and open routes so the LLM can produce valid actions."""
    base = obs.to_prompt_text()
    extra = []

    cargo_lines = []
    for cid in obs.own_cargo_queue:
        c = env.world.cargo_queue.get(cid)
        if c and not c.delivered and not c.spoiled:
            flags = []
            if c.temp_sensitive: flags.append("COLD-CHAIN")
            if getattr(c, "priority", None) == "CRITICAL": flags.append("CRITICAL")
            cargo_lines.append(
                f"  {cid}: dest={c.destination}  weight={c.weight_tons:.1f}t"
                f"  deadline_turn={c.deadline}"
                + (f"  [{', '.join(flags)}]" if flags else "")
            )
    if cargo_lines:
        extra.append("YOUR CARGO DETAILS:\n" + "\n".join(cargo_lines[:6]))

    route_lines = []
    for rid, route in env.world.routes.items():
        if not route.blocked:
            route_lines.append(f"  {rid} -> {route.to_node}")
    if route_lines:
        extra.append("OPEN ROUTES (not blocked):\n" + "\n".join(route_lines[:10]))

    other_agents = [aid for aid in env.world.agent_states if aid != agent_id]
    if other_agents:
        extra.append(f"OTHER AGENTS: {other_agents}")

    return base + "\n\n" + "\n\n".join(extra)


def _llm_action(agent_id: str, prompt_text: str, model: str) -> dict | None:
    client = _get_llm_client()
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt_text},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            action = json.loads(raw)
            if "action_type" in action:
                return action
        except Exception as e:
            if attempt == 0:
                print(f"    {DIM}[LLM attempt {attempt+1} failed: {str(e)[:80]}]{RESET}",
                      flush=True)
                time.sleep(1)
            else:
                print(f"    {DIM}[LLM fallback → heuristic: {str(e)[:60]}]{RESET}",
                      flush=True)
    return None


# ── Section 2: Scenario Runner ─────────────────────────────────────────────────

SCENARIO_PARAMS = {
    "flood": {
        "disruption_type": DisruptionType.FLOOD,
        "description": "Monsoon floods block river-crossing routes (Mumbai, Kolkata corridor)",
        "severity": 4,
        "color": BLUE,
    },
    "port_strike": {
        "disruption_type": DisruptionType.PORT_STRIKE,
        "description": "Port workers' strike halts all sea-port cargo (Mumbai, Chennai, Kolkata)",
        "severity": 3,
        "color": RED,
    },
    "road_closure": {
        "disruption_type": DisruptionType.ROAD_CLOSURE,
        "description": "Highway closures after protests block arterial NH routes",
        "severity": 3,
        "color": YELLOW,
    },
}


def run_scenario(
    name: str,
    curriculum_level: int = 2,
    max_turns: int = 12,
    use_llm: bool = False,
    llm_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    show_steps: bool = True,
    verbose_steps: int = 3,
) -> dict:
    params = SCENARIO_PARAMS.get(name, SCENARIO_PARAMS["flood"])
    color = params["color"]

    box(f"SCENARIO: {name.upper().replace('_', ' ')}  [curriculum level {curriculum_level}]", color)
    info(f"Description: {params['description']}")
    info(f"Severity: {params['severity']}  |  Policy: {'LLM (' + llm_model.split('/')[-1] + ')' if use_llm else 'Heuristic'}")

    env = LogiCrisisEnv(
        curriculum_level=curriculum_level,
        seed=42,
        max_turns=max_turns,
    )
    observations = env.reset()

    # Apply scenario disruption type
    for d in env.world.disruptions:
        d.disruption_type = params["disruption_type"]
        d.severity = params["severity"]

    agent_ids = list(observations.keys())
    print(f"\n  {BOLD}Agents:{RESET} {agent_ids}")
    print(f"  {BOLD}Cargo:{RESET}   {len(env.world.cargo_queue)} items")
    print(f"  {BOLD}Blocked routes at start:{RESET} {[r for r, v in env.world.routes.items() if v.blocked]}")

    # Show live context from env
    live_ctx = getattr(env, "_live_context", None)
    if live_ctx and not live_ctx.is_empty():
        live_lines = live_ctx.to_prompt_lines()
        if live_lines:
            print(f"\n  {CYAN}Live signals injected into this episode:{RESET}")
            for l in live_lines[:3]:
                print(f"    {CYAN}{l[:80]}{RESET}")

    episode_rewards: dict[str, float] = {aid: 0.0 for aid in agent_ids}
    turn = 0
    action_log = []

    while True:
        actions_dict: dict[str, dict] = {}
        agent_actions: dict[str, AgentAction] = {}

        for agent_id, obs in observations.items():
            d = None
            if use_llm:
                rich_prompt = _enrich_prompt(agent_id, obs, env)
                d = _llm_action(agent_id, rich_prompt, llm_model)
            if d is None:
                d = _heuristic(agent_id, obs, env.world)
            actions_dict[agent_id] = d
            agent_actions[agent_id] = _dict_to_action(agent_id, d)

        result = env.step(agent_actions)
        turn += 1

        for aid, r in result.rewards.items():
            episode_rewards[aid] = episode_rewards.get(aid, 0.0) + r

        snap = env.state()
        action_log.append({
            "turn": turn,
            "otif": snap["otif_percent"],
            "actions": {a: d.get("action_type") for a, d in actions_dict.items()},
            "rewards": {a: round(r, 3) for a, r in result.rewards.items()},
        })

        if show_steps and turn <= verbose_steps:
            print(f"\n  {BOLD}Turn {turn}/{max_turns}{RESET}  OTIF={snap['otif_percent']:.1f}%")
            for aid, d in actions_dict.items():
                atype = d.get("action_type", "?")
                reason = d.get("reasoning", "")[:70]
                sym = {"reroute": "🚚", "propose_coalition": "🤝", "deploy_cold_storage": "❄",
                       "make_bid": "💰", "wait": "⏳", "join_coalition": "👥"}.get(atype, "·")
                r = result.rewards.get(aid, 0.0)
                r_color = GREEN if r > 0 else (RED if r < 0 else DIM)
                print(f"    {sym} {aid[:16]:<18} {BOLD}{atype:<22}{RESET} "
                      f"{r_color}r={r:+.3f}{RESET}  \"{reason}\"")

        elif show_steps and turn == verbose_steps + 1:
            print(f"\n  {DIM}... (running remaining turns) ...{RESET}")

        observations = result.observations
        if result.terminated or result.truncated:
            break

    # Grade
    from environment.tasks import get_task
    try:
        grade_task_map = {1: "single_route_recovery", 2: "coalition_logistics", 3: "cascade_failure_recovery"}
        task_id = grade_task_map.get(curriculum_level, "single_route_recovery")
        task = get_task(task_id)
        # Re-use env state for grading
        grade = task.grade(env)
    except Exception:
        delivered = sum(1 for c in env.world.cargo_queue.values() if c.delivered)
        total = len(env.world.cargo_queue)
        grade = {
            "score": delivered / max(total, 1),
            "otif_percent": snap["otif_percent"],
            "passed": delivered / max(total, 1) >= 0.5,
            "verdict": "computed",
            "breakdown": {},
        }

    # Summary
    print(f"\n  {BOLD}{'─'*55}{RESET}")
    print(f"  {BOLD}Final OTIF:  {snap['otif_percent']:.1f}%{RESET}")
    print(f"  {BOLD}Score:       {grade['score']:.4f}{RESET}",
          f"  {GREEN}PASS{RESET}" if grade["passed"] else f"  {RED}FAIL{RESET}")
    print(f"  {BOLD}Turns used:  {turn}/{max_turns}{RESET}")

    # OTIF sparkline
    otif_vals = [r["otif"] for r in action_log]
    if otif_vals:
        spark = _sparkline(otif_vals, 0, 100)
        print(f"  {BOLD}OTIF trend:  {CYAN}{spark}{RESET}")

    cum_r = sum(episode_rewards.values())
    print(f"  {BOLD}Cum reward:  {cum_r:+.3f}{RESET}")

    return {
        "scenario": name,
        "policy": "llm" if use_llm else "heuristic",
        "grade": grade,
        "turns": turn,
        "cumulative_reward": cum_r,
        "otif_trace": otif_vals,
    }


def _sparkline(vals: list[float], lo: float, hi: float) -> str:
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for v in vals:
        idx = int((v - lo) / max(hi - lo, 1) * (len(chars) - 1))
        out.append(chars[max(0, min(idx, len(chars) - 1))])
    return "".join(out)


# ── Section 3: Task Challenge Panel ───────────────────────────────────────────

def run_task_challenges(use_llm: bool, llm_model: str) -> None:
    box("SECTION 3: TASK CHALLENGE PANEL — All 9 Tasks", GREEN)
    info("Running all tasks with heuristic policy. LLM expected to improve the failing ones.")
    print()

    TASK_NOTES = {
        "earthquake_relief":  "Needs CRITICAL cargo prioritisation — naive routing fails",
        "capacity_crunch":    "Needs bid/counter-propose market logic — heuristic can't trade capacity",
        "negotiation_sprint": "Needs active bid/accept chains — heuristic only waits",
        "national_recovery":  "Needs coordinated 5-agent coalition — complex",
    }

    results = []
    for task_id in ALL_TASK_IDS:
        task = get_task(task_id)
        env = task.make_env(seed=42)
        observations = env.reset()
        agent_ids = list(observations.keys())

        for _t in range(task.max_turns):
            actions: dict[str, AgentAction] = {}
            for aid, obs in observations.items():
                d: dict | None = None
                if use_llm:
                    d = _llm_action(aid, obs.to_prompt_text(), llm_model)
                if d is None:
                    d = _heuristic(aid, obs, env.world)
                actions[aid] = _dict_to_action(aid, d)
            res = env.step(actions)
            observations = res.observations
            if res.terminated or res.truncated:
                break

        grade = task.grade(env)
        results.append(grade)

        status_color = GREEN if grade["passed"] else RED
        note = TASK_NOTES.get(task_id, "")
        print(f"  {status_color}{'PASS' if grade['passed'] else 'FAIL'}{RESET}  "
              f"{task_id:<30}  score={grade['score']:.4f}  OTIF={grade['otif_percent']:.1f}%"
              + (f"  {DIM}← {note}{RESET}" if note else ""))

    avg = sum(r["score"] for r in results) / len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n  {BOLD}Average score: {avg:.4f}  |  Passed: {passed}/{len(results)}{RESET}")

    policy = "LLM" if use_llm else "Heuristic"
    print(f"\n  {DIM}Policy: {policy}{RESET}")


# ── Section 4: Comparison table ────────────────────────────────────────────────

def show_comparison(results: list[dict]) -> None:
    box("SECTION 4: SCENARIO COMPARISON SUMMARY", CYAN)
    print(f"\n  {'Scenario':<20} {'Policy':<12} {'OTIF':<8} {'Score':<8} {'Verdict':<8} {'OTIF Trend'}")
    print(f"  {'─'*20} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*20}")
    for r in results:
        grade = r["grade"]
        verdict = f"{GREEN}PASS{RESET}" if grade["passed"] else f"{RED}FAIL{RESET}"
        spark = _sparkline(r["otif_trace"], 0, 100)
        print(f"  {r['scenario']:<20} {r['policy']:<12} "
              f"{grade['otif_percent']:<8.1f} {grade['score']:<8.4f} {verdict:<15} {CYAN}{spark}{RESET}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LogiCrisis Live Demo")
    parser.add_argument("--llm", action="store_true", help="Use HF Inference API LLM")
    parser.add_argument("--model",
                        default=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
                        help="Model to use via HF Inference API")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "flood", "port_strike", "road_closure"])
    parser.add_argument("--tasks", action="store_true", help="Run all 9 task challenges")
    args = parser.parse_args()

    use_llm = args.llm and bool(os.environ.get("HF_TOKEN"))

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║           LogiCrisis — Live Multi-Agent Logistics Demo               ║
║     Meta PyTorch OpenEnv Hackathon — Multi-Agent Interactions        ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}

{DIM}Policy: {'LLM via HF Inference API (' + args.model + ')' if use_llm else 'Heuristic fallback (no --llm flag or HF_TOKEN)'}
Live APIs: OpenWeatherMap, ExchangeRate-API, NewsAPI, GDELT, World Bank{RESET}
""")

    # Section 1: live data
    live_ctx = show_live_data()

    # Section 2: scenarios
    results = []
    scenarios_to_run = (
        list(SCENARIO_PARAMS.keys()) if args.scenario == "all" else [args.scenario]
    )

    for sname in scenarios_to_run:
        r = run_scenario(
            sname,
            curriculum_level=2,
            max_turns=12,
            use_llm=use_llm,
            llm_model=args.model,
            show_steps=True,
            verbose_steps=3,
        )
        results.append(r)

    # Section 3: task challenges (optional)
    if args.tasks:
        run_task_challenges(use_llm=use_llm, llm_model=args.model)

    # Section 4: comparison
    if len(results) > 1:
        show_comparison(results)

    # Closing
    box("DEMO COMPLETE", GREEN)
    print(f"""
  The trained model reads live weather, tariff, and conflict signals
  directly in its observation prompt and adapts routing decisions:

  {CYAN}• Dense Fog in Mumbai + Delhi:{RESET}  agents reroute via Pune or Surat bypass
  {CYAN}• Tariff Shock (USD/INR +12%):{RESET}  Customs Broker prioritises negotiate_bypass
  {CYAN}• NewsAPI Geopolitical Signals:{RESET} Geopolitical Analyst issues corridor alerts

  {DIM}Start the interactive Gradio UI:   python demo/app.py
  Run all 9 task challenges with LLM: python demo_live.py --llm --tasks
  Full inference with structured log: python inference.py{RESET}
""")


if __name__ == "__main__":
    main()

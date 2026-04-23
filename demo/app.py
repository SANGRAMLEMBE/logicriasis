"""
LogiCrisis — Gradio demo with live reward chart, Theory-of-Mind panel,
Overseer Agent (Fleet AI sub-theme), and before/after grader output.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd

from environment import LogiCrisisEnv, AgentAction, ActionType, AgentRole
from environment.tasks import get_task, ALL_TASK_IDS

# ── Global state ──────────────────────────────────────────────────────────────
_env: LogiCrisisEnv | None = None
_observations = {}
_history: list[dict] = []          # per-turn records for the chart
_belief_state: dict[str, dict] = {}  # theory-of-mind: what each agent thinks about others


# ── Episode control ───────────────────────────────────────────────────────────

def start_episode(disruption_type: str, severity: int, curriculum_level: int):
    global _env, _observations, _history, _belief_state
    _env = LogiCrisisEnv(curriculum_level=int(curriculum_level), seed=42)
    _observations = _env.reset()

    from environment.models import DisruptionType
    for d in _env.world.disruptions:
        try:
            d.disruption_type = DisruptionType(disruption_type)
            d.severity = int(severity)
        except ValueError:
            pass

    _history = []
    _belief_state = _init_belief_state()

    snap = _env.state()
    status = (
        f"Episode started!\n"
        f"Level {int(curriculum_level)} | Disruption: {disruption_type} (severity {int(severity)})\n"
        f"Agents: {list(_observations.keys())}\n"
        f"Cargo items: {snap['cargo_summary']['total']} | "
        f"Blocked routes: {len(snap['blocked_routes'])}"
    )
    return (
        status,
        _empty_chart(),
        _format_state(snap),
        _format_belief_state(),
        _overseer_report([]),
        0.0,
        0,
    )


# ── Heuristic action logic ────────────────────────────────────────────────────

def _pick_heuristic_action(agent_id: str, obs) -> AgentAction:
    if not obs.own_cargo_queue:
        return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                           reasoning="No cargo in queue")

    for cargo_id in obs.own_cargo_queue:
        cargo = _env.world.cargo_queue.get(cargo_id)
        if not cargo or cargo.delivered or cargo.spoiled:
            continue

        # Deliver: find open route leading to destination
        for rid, route in _env.world.routes.items():
            if not route.blocked and route.to_node == cargo.destination:
                return AgentAction(
                    agent_id=agent_id,
                    action_type=ActionType.REROUTE,
                    cargo_id=cargo_id,
                    route_id=rid,
                    reasoning=f"Routing {cargo_id} via {rid} → {cargo.destination}",
                )

        # Cold chain rescue
        state = _env.world.agent_states.get(agent_id)
        if cargo.temp_sensitive and state and state.cold_storage_units > 0:
            return AgentAction(
                agent_id=agent_id,
                action_type=ActionType.DEPLOY_COLD_STORAGE,
                cargo_id=cargo_id,
                reasoning=f"Cold chain rescue for {cargo_id}",
            )

    return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                       reasoning="No viable route this turn")


def auto_step():
    global _env, _observations, _history, _belief_state
    if _env is None:
        return ("Click 'Start Episode' first.",
                _empty_chart(), "", _format_belief_state(),
                _overseer_report([]), 0.0, 0)

    actions = {aid: _pick_heuristic_action(aid, obs)
               for aid, obs in _observations.items()}

    result = _env.step(actions)
    _observations = result.observations
    snap = _env.state()

    # Update theory-of-mind belief states
    _update_belief_state(actions, result)

    # Record turn for chart
    record = {"Turn": snap["turn"], "OTIF %": snap["otif_percent"]}
    for aid, rb in result.reward_breakdown.items():
        record[f"{aid[:8]} R1"] = rb.get("R1_delivery", 0)
        record[f"{aid[:8]} R2"] = rb.get("R2_coalition", 0)
        record[f"{aid[:8]} R3"] = rb.get("R3_negotiation", 0)
        record[f"{aid[:8]} R4"] = rb.get("R4_cold_chain", 0)
        record[f"{aid[:8]} total"] = rb.get("total", 0)
    _history.append(record)

    status = "Running"
    if result.terminated:
        status = "DONE — All cargo resolved"
    elif result.truncated:
        status = "ENDED — Timeout or cheat detected"

    agent_log = "\n".join(
        f"  {aid}: reward={r:+.3f}  |  {_format_breakdown(result.reward_breakdown.get(aid, {}))}"
        for aid, r in result.rewards.items()
    )

    return (
        f"Turn {snap['turn']}/{_env.world.max_turns} | {status}\n\n{agent_log}",
        _build_chart(),
        _format_state(snap),
        _format_belief_state(),
        _overseer_report(list(actions.values())),
        float(snap["otif_percent"]),
        int(snap["turn"]),
    )


# ── Theory-of-Mind ────────────────────────────────────────────────────────────

def _init_belief_state() -> dict:
    beliefs = {}
    for aid in _env.world.agent_states:
        beliefs[aid] = {
            other: {"est_capacity": "unknown", "est_budget": "unknown",
                    "trust": 1.0, "last_action": "none"}
            for other in _env.world.agent_states if other != aid
        }
    return beliefs


def _update_belief_state(actions: dict, result) -> None:
    for aid, action in actions.items():
        for other_id, other_action in actions.items():
            if other_id == aid:
                continue
            b = _belief_state[aid][other_id]
            b["last_action"] = other_action.action_type.value

            # Update trust: if other cooperated (accepted bid / joined coalition) → trust up
            if other_action.action_type in (ActionType.ACCEPT_BID, ActionType.JOIN_COALITION):
                b["trust"] = min(2.0, b["trust"] + 0.1)
            # If other waited repeatedly → lower trust
            elif other_action.action_type == ActionType.WAIT:
                b["trust"] = max(0.0, b["trust"] - 0.05)

            # Infer capacity from reroute actions
            if other_action.action_type == ActionType.REROUTE and other_action.cargo_id:
                cargo = _env.world.cargo_queue.get(other_action.cargo_id)
                if cargo:
                    b["est_capacity"] = f"≥{cargo.weight_tons:.1f}t"

            # Budget inference from bids
            if other_action.action_type == ActionType.MAKE_BID and other_action.bid_price:
                b["est_budget"] = f"≥${other_action.bid_price:.0f}"


def _format_belief_state() -> str:
    if not _belief_state:
        return "Start an episode to see Theory-of-Mind belief tracking."

    lines = ["=== THEORY-OF-MIND: What each agent believes about others ===\n"]
    for agent_id, beliefs in _belief_state.items():
        lines.append(f"Agent: {agent_id}")
        for other_id, b in beliefs.items():
            trust_bar = "█" * int(b["trust"] * 5) + "░" * (10 - int(b["trust"] * 5))
            lines.append(
                f"  → {other_id}: "
                f"trust=[{trust_bar}] {b['trust']:.2f} | "
                f"est_capacity={b['est_capacity']} | "
                f"est_budget={b['est_budget']} | "
                f"last_seen={b['last_action']}"
            )
        lines.append("")
    return "\n".join(lines)


# ── Overseer Agent (Fleet AI sub-theme) ───────────────────────────────────────

def _overseer_report(actions: list[AgentAction]) -> str:
    if _env is None:
        return "Overseer Agent inactive. Start an episode."

    lines = ["=== OVERSEER AGENT — Fleet AI Monitor ===\n"]
    snap = _env.state()

    # 1. Cooperation signal
    coalition_count = len([c for c in _env.world.coalitions.values() if not c.dissolved])
    if coalition_count > 0:
        lines.append(f"✓ COOPERATION DETECTED: {coalition_count} active coalition(s)")
    else:
        lines.append("⚠ NO COALITIONS ACTIVE — agents operating solo")

    # 2. Reward hacking check
    cheat_events = [e for e in _env.world.audit_log if e.get("type") == "cheat_detected"]
    if cheat_events:
        lines.append(f"🚨 REWARD HACKING DETECTED: {len(cheat_events)} violation(s)")
        for ev in cheat_events[-3:]:
            lines.append(f"   Agent {ev.get('agent_id')}: {ev.get('reason')}")
    else:
        lines.append("✓ NO REWARD HACKING — all agent actions verified clean")

    # 3. Cold chain monitoring
    cold_spoiled = sum(1 for c in _env.world.cargo_queue.values()
                       if c.temp_sensitive and c.spoiled)
    cold_total = sum(1 for c in _env.world.cargo_queue.values() if c.temp_sensitive)
    if cold_total > 0:
        lines.append(f"{'🚨' if cold_spoiled else '✓'} COLD CHAIN: "
                     f"{cold_total - cold_spoiled}/{cold_total} intact")

    # 4. Agent behaviour summary
    lines.append("\nBehaviour this turn:")
    for action in actions:
        symbol = {
            "reroute": "🚚", "make_bid": "💰", "accept_bid": "✅",
            "propose_coalition": "🤝", "join_coalition": "👥",
            "deploy_cold_storage": "❄️", "wait": "⏳",
        }.get(action.action_type.value, "•")
        lines.append(f"  {symbol} {action.agent_id}: {action.action_type.value}"
                     + (f" — {action.reasoning[:60]}" if action.reasoning else ""))

    # 5. OTIF trajectory
    if len(_history) >= 2:
        delta = _history[-1]["OTIF %"] - _history[-2]["OTIF %"]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        lines.append(f"\nOTIF trajectory: {arrow} {delta:+.1f}% this turn "
                     f"(now {_history[-1]['OTIF %']:.1f}%)")

    return "\n".join(lines)


# ── Chart builder ─────────────────────────────────────────────────────────────

def _empty_chart():
    df = pd.DataFrame({"Turn": [], "OTIF %": []})
    return df

def _build_chart():
    if not _history:
        return _empty_chart()
    df = pd.DataFrame(_history)
    return df[["Turn", "OTIF %"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_state(snap: dict) -> str:
    return json.dumps(snap, indent=2)

def _format_breakdown(rb: dict) -> str:
    parts = []
    for k in ["R1_delivery", "R2_coalition", "R3_negotiation", "R4_cold_chain",
              "R5_efficiency", "R6_anti_cheat"]:
        v = rb.get(k, 0)
        if v != 0:
            parts.append(f"{k.split('_')[0]}={v:+.2f}")
    return " | ".join(parts) if parts else "all zero"


# ── Full episode runner (for grader panel) ────────────────────────────────────

def run_full_episode(task_id: str) -> str:
    """Run a complete episode for a task and return the grader output."""
    task = get_task(task_id)
    env = task.make_env(seed=42)
    observations = env.reset()
    agent_ids = list(observations.keys())

    for _turn in range(task.max_turns):
        actions = {}
        for agent_id, obs in observations.items():
            state = env.world.agent_states.get(agent_id)
            if _turn == 0 and state and not state.coalition_id:
                others = [aid for aid in agent_ids if aid != agent_id][:2]
                split = {agent_id: 0.5}
                for m in others:
                    split[m] = 0.5 / max(len(others), 1)
                actions[agent_id] = AgentAction(
                    agent_id=agent_id,
                    action_type=ActionType.PROPOSE_COALITION,
                    coalition_id=f"coal_{agent_id}",
                    coalition_members=others,
                    reward_split=split,
                    reasoning="coalition for collaborative delivery",
                )
                continue
            rescued = False
            if state and state.cold_storage_units > 0 and state.budget >= 200:
                for cargo_id in obs.own_cargo_queue:
                    cargo = env.world.cargo_queue.get(cargo_id)
                    if cargo and cargo.temp_sensitive and not cargo.spoiled and not cargo.delivered:
                        actions[agent_id] = AgentAction(
                            agent_id=agent_id,
                            action_type=ActionType.DEPLOY_COLD_STORAGE,
                            cargo_id=cargo_id,
                            reasoning="cold storage protection",
                        )
                        rescued = True
                        break
            if rescued:
                continue
            routed = False
            for cargo_id in obs.own_cargo_queue:
                cargo = env.world.cargo_queue.get(cargo_id)
                if not cargo or cargo.delivered or cargo.spoiled:
                    continue
                for rid, route in env.world.routes.items():
                    if not route.blocked and route.to_node == cargo.destination:
                        if state and state.capacity_tons >= cargo.weight_tons:
                            actions[agent_id] = AgentAction(
                                agent_id=agent_id,
                                action_type=ActionType.REROUTE,
                                cargo_id=cargo_id,
                                route_id=rid,
                                reasoning=f"direct route to {cargo.destination}",
                            )
                            routed = True
                            break
                if routed:
                    break
            if not routed:
                actions[agent_id] = AgentAction(
                    agent_id=agent_id, action_type=ActionType.WAIT, reasoning="waiting")

        result = env.step(actions)
        observations = result.observations
        if result.terminated or result.truncated:
            break

    grade = task.grade(env)
    verdict = "✓ PASS" if grade["passed"] else "✗ FAIL"
    lines = [
        f"Task: {grade['task_id']}",
        f"Score: {grade['score']:.4f}  OTIF: {grade['otif_percent']:.1f}%  {verdict}",
        "",
        "Breakdown:",
    ]
    for k, v in grade.get("breakdown", {}).items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


BASELINE_SCORES = """\
Heuristic Baseline (seed=42, no LLM):
  single_route_recovery    score=1.0000  OTIF=100.0%  ✓ PASS  [easy]
  coalition_logistics      score=0.7667  OTIF=73.3%   ✓ PASS  [medium]
  cascade_failure_recovery score=0.6254  OTIF=85.0%   ✓ PASS  [hard]
  cold_chain_emergency     score=0.8333  OTIF=83.3%   ✓ PASS  [medium-hard]
  negotiation_sprint       score=0.6000  OTIF=100.0%  ✓ PASS  [medium]
  national_recovery        score=0.6261  OTIF=68.0%   ✓ PASS  [expert]
  earthquake_relief        score=0.1176  OTIF=56.8%   ✗ FAIL  [hard]   ← needs priority reasoning
  capacity_crunch          score=0.3770  OTIF=55.0%   ✗ FAIL  [hard]   ← needs market bidding
  jit_breakdown            score=0.9515  OTIF=85.7%   ✓ PASS  [medium-hard]
  Average score: 0.6553

LLM agents (Llama-3-8B-Instruct via GRPO fine-tuning) are expected to:
  • Pass earthquake_relief by prioritising CRITICAL cargo over naive speed routing
  • Pass capacity_crunch by making/accepting bids to trade capacity on the market
  • Improve negotiation_sprint via actual bid/counter-propose actions
  • Score higher on national_recovery via coordinated coalition play
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="LogiCrisis — Multi-Agent Logistics Recovery") as demo:

    gr.Markdown("""
# LogiCrisis: Multi-Agent Logistics Recovery
**Meta PyTorch OpenEnv Hackathon — Theme #1: Multi-Agent Interactions**
> A partially observable environment where AI agents cooperate, compete, negotiate,
> and form coalitions to recover India's supply chain after a real-world disruption.

Stack: **OpenEnv** (environment) → **6 independent reward functions** (verifier) → **TRL GRPOTrainer** (RL) → **Unsloth 4-bit QLoRA** (efficiency)
""")

    # ── Controls row ──────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            disruption_dd = gr.Dropdown(
                choices=["flood", "port_strike", "road_closure"],
                value="flood", label="Disruption Type")
            severity_sl = gr.Slider(1, 5, value=3, step=1,
                                    label="Severity (1=minor → 5=cascade)")
            level_sl = gr.Slider(1, 3, value=1, step=1,
                                 label="Curriculum Level (1=easy, 3=hard)")
            with gr.Row():
                start_btn = gr.Button("▶ Start Episode", variant="primary")
                step_btn  = gr.Button("⏭ Step", variant="secondary")

        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Episode Status", lines=6)
            with gr.Row():
                otif_num = gr.Number(label="OTIF %", value=0.0, precision=1)
                turn_num = gr.Number(label="Turn",   value=0,   precision=0)

    # ── Live OTIF chart ───────────────────────────────────────────────────────
    gr.Markdown("### Live OTIF Recovery Chart")
    otif_chart = gr.LinePlot(
        value=_empty_chart(),
        x="Turn",
        y="OTIF %",
        title="On-Time In-Full % per Turn (target: >70%)",
        y_lim=[0, 100],
        height=300,
    )

    # ── Theory-of-Mind + Overseer ─────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Theory-of-Mind Belief Tracker")
            belief_box = gr.Textbox(
                label="What each agent believes about others",
                lines=14, max_lines=20)
        with gr.Column():
            gr.Markdown("### Overseer Agent — Fleet AI Monitor")
            overseer_box = gr.Textbox(
                label="Real-time cooperative behaviour analysis",
                lines=14, max_lines=20)

    # ── World state JSON ──────────────────────────────────────────────────────
    with gr.Accordion("Full World State (JSON)", open=False):
        state_box = gr.Code(language="json", lines=20)

    # ── Reward reference ──────────────────────────────────────────────────────
    gr.Markdown("""
### 6 Independent Reward Signals (anti-hacking per guide §7 & §8)
| Signal | What it measures | Score |
|--------|-----------------|-------|
| R1 Delivery success | On-time deliveries | +1.0 on-time / -0.5 late / -1.0 fail |
| R2 Coalition quality | Team vs solo performance | +0.3 if coalition beats solo |
| R3 Negotiation fairness | Deal outcomes | +0.2 accepted / -0.3 breach |
| R4 Cold chain | Pharma/food integrity | 0–1 fraction intact |
| R5 Resource efficiency | Fleet utilisation | Utilisation bonus − idle penalty |
| R6 Anti-cheat | Hidden state access | −2.0 instant penalty |
""")

    # ── Grader / before-after panel (guide §19) ───────────────────────────────
    gr.Markdown("### Episode Grader — Verifier Output (guide §19)")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Step 1: Run grader on a complete episode**")
            task_dd = gr.Dropdown(
                choices=ALL_TASK_IDS,
                value="single_route_recovery",
                label="Task",
            )
            grade_btn = gr.Button("Run Full Episode + Grade", variant="primary")
            grade_box = gr.Textbox(label="Grader Result", lines=10)
        with gr.Column():
            gr.Markdown("**Step 2: Baseline scores (heuristic policy)**")
            gr.Textbox(
                value=BASELINE_SCORES,
                label="Documented Baseline (inference.py, seed=42)",
                lines=12,
                interactive=False,
            )

    grade_btn.click(fn=run_full_episode, inputs=[task_dd], outputs=[grade_box])

    # ── Button wiring ─────────────────────────────────────────────────────────
    outputs = [status_box, otif_chart, state_box, belief_box, overseer_box, otif_num, turn_num]

    start_btn.click(
        fn=start_episode,
        inputs=[disruption_dd, severity_sl, level_sl],
        outputs=outputs,
    )
    step_btn.click(
        fn=auto_step,
        inputs=[],
        outputs=outputs,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )

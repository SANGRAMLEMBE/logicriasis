"""
6 independent reward functions for LogiCrisis.
Each function is isolated — cannot be gamed by optimising one signal alone.
Per hackathon guide section 7 & 8.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import WorldState
    from .models import AgentAction


# ── R1 — Delivery success ─────────────────────────────────────────────────────

def r1_delivery_success(world: "WorldState", agent_id: str) -> float:
    score = 0.0
    agent_cargo = world.get_cargo_for_agent(agent_id)

    for cargo in world.cargo_queue.values():
        if cargo.owner_agent != agent_id:
            continue
        if cargo.delivered and not cargo.spoiled:
            turns_late = max(0, world.turn - cargo.deadline)
            if turns_late == 0:
                score += 1.0
            else:
                score -= 0.5 * min(turns_late, 2)
        elif cargo.spoiled or (not cargo.delivered and world.turn >= cargo.deadline):
            score -= 1.0

    return round(score, 3)


# ── R2 — Coalition quality ────────────────────────────────────────────────────

def r2_coalition_quality(world: "WorldState", agent_id: str) -> float:
    agent_state = world.agent_states.get(agent_id)
    if not agent_state or not agent_state.coalition_id:
        return 0.0

    coal = world.coalitions.get(agent_state.coalition_id)
    if not coal or coal.dissolved:
        return 0.0

    # Check if coalition delivered more than solo baseline
    coal_delivered = sum(
        1 for cid in coal.cargo_ids
        if world.cargo_queue.get(cid) and world.cargo_queue[cid].delivered
    )
    solo_baseline = max(1, len(coal.cargo_ids) // 2)

    if coal_delivered > solo_baseline:
        split = coal.reward_split.get(agent_id, 1.0 / max(len(coal.members), 1))
        return round(0.3 * split, 3)
    else:
        return -0.2


# ── R3 — Negotiation fairness ─────────────────────────────────────────────────

def r3_negotiation_fairness(world: "WorldState", agent_id: str,
                             action: "AgentAction") -> float:
    score = 0.0
    for bid in world.bids.values():
        if bid.from_agent != agent_id and bid.to_agent != agent_id:
            continue
        if bid.accepted:
            score += 0.2
        if bid.breached:
            score -= 0.3
    return round(score, 3)


# ── R4 — Cold chain integrity ─────────────────────────────────────────────────

def r4_cold_chain(world: "WorldState", agent_id: str) -> float:
    cold_cargo = [
        c for c in world.cargo_queue.values()
        if c.owner_agent == agent_id and c.temp_sensitive
    ]
    if not cold_cargo:
        return 0.0

    intact = sum(1 for c in cold_cargo if c.delivered and not c.spoiled)
    total = len(cold_cargo)
    return round(intact / total, 3) if total else 0.0


# ── R5 — Resource efficiency ──────────────────────────────────────────────────

def r5_resource_efficiency(world: "WorldState", agent_id: str) -> float:
    state = world.agent_states.get(agent_id)
    if not state:
        return 0.0

    # Utilisation bonus: higher is better
    delivered_weight = sum(
        c.weight_tons for c in world.cargo_queue.values()
        if c.owner_agent == agent_id and c.delivered
    )
    utilisation = delivered_weight / max(state.capacity_tons, 1)
    efficiency_bonus = min(utilisation * 0.1, 0.3)

    # Penalise idle trucks (counted via audit log)
    idle_events = sum(
        1 for ev in world.audit_log
        if ev.get("agent_id") == agent_id and ev.get("type") == "idle_truck"
    )
    penalty = idle_events * 0.05

    return round(efficiency_bonus - penalty, 3)


# ── R6 — Anti-cheat verifier (hard penalty) ───────────────────────────────────

def r6_anti_cheat(world: "WorldState", agent_id: str,
                   action: "AgentAction") -> float:
    """
    Returns -2.0 if cheating detected, 0.0 otherwise.
    Cheating: accessing hidden state, fake timestamps, action loops.
    """
    # Detect repeated identical actions in last 3 turns (loop exploit)
    recent = [
        ev for ev in world.audit_log[-15:]
        if ev.get("agent_id") == agent_id and ev.get("type") == "action"
    ]
    if len(recent) >= 5:
        last5 = [ev.get("action_type") for ev in recent[-5:]]
        if all(a == "wait" for a in last5):
            world.log({"agent_id": agent_id, "type": "cheat_detected",
                       "reason": "5 consecutive wait actions (loop exploit)"})
            return -1.0

    # Detect if reasoning mentions hidden world-state fields
    FORBIDDEN_HINTS = ["world.routes", "world.agent_states", "_hidden", "global_state"]
    reasoning_lower = action.reasoning.lower()
    for hint in FORBIDDEN_HINTS:
        if hint in reasoning_lower:
            world.log({"agent_id": agent_id, "type": "cheat_detected",
                       "reason": f"hidden-state hint: {hint}"})
            return -2.0

    return 0.0


# ── R7 — Carbon footprint (reroute penalty) ───────────────────────────────────

def r7_carbon_footprint(world: "WorldState", agent_id: str,
                         action: "AgentAction") -> float:
    from .models import ActionType
    if action.action_type != ActionType.REROUTE:
        return 0.0

    route = world.routes.get(action.route_id) if action.route_id else None
    cargo = world.cargo_queue.get(action.cargo_id) if action.cargo_id else None

    if route is None or cargo is None:
        return 0.0

    return round(-(route.distance_km * cargo.weight_tons * 0.001), 3)


# ── Composite reward ──────────────────────────────────────────────────────────

def compute_rewards(
    world: "WorldState",
    actions: dict[str, "AgentAction"],
) -> dict[str, dict[str, float]]:
    """
    Returns per-agent reward breakdown: {agent_id: {R1..R7, total, shared}}.
    """
    system_otif = world.otif_percent()
    severity_mult = world.severity_multiplier()
    shared_bonus = (system_otif / 100.0) * severity_mult * 0.5

    results: dict[str, dict[str, float]] = {}
    for agent_id in world.agent_states:
        action = actions.get(agent_id)
        if action is None:
            from .models import AgentAction, ActionType
            action = AgentAction(agent_id=agent_id, action_type=ActionType.WAIT)

        r1 = r1_delivery_success(world, agent_id)
        r2 = r2_coalition_quality(world, agent_id)
        r3 = r3_negotiation_fairness(world, agent_id, action)
        r4 = r4_cold_chain(world, agent_id)
        r5 = r5_resource_efficiency(world, agent_id)
        r6 = r6_anti_cheat(world, agent_id, action)
        r7 = r7_carbon_footprint(world, agent_id, action)

        total = r1 + r2 + r3 + r4 + r5 + r6 + r7 + shared_bonus

        results[agent_id] = {
            "R1_delivery": r1,
            "R2_coalition": r2,
            "R3_negotiation": r3,
            "R4_cold_chain": r4,
            "R5_efficiency": r5,
            "R6_anti_cheat": r6,
            "R7_carbon": r7,
            "shared_bonus": round(shared_bonus, 3),
            "total": round(total, 3),
        }

    return results

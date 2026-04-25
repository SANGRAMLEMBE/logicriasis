"""
Per-role manager configurations for LogiCrisis.

Each agent is a deep specialist with:
- A restricted action space (only domain-relevant actions)
- Role-weighted reward multipliers for GRPO training
- KPI definitions used in grading and prompt context
- API data signals it pays attention to
"""
from __future__ import annotations

# ── Role configurations ────────────────────────────────────────────────────────

ROLE_CONFIGS: dict[str, dict] = {

    "carrier": {
        "title": "Carrier Manager",
        "specialty": "Route optimization and fleet utilization",
        "allowed_actions": [
            "reroute",
            "request_transfer",
            "make_bid",
            "accept_bid",
            "reject_bid",
            "propose_coalition",
            "join_coalition",
            "leave_coalition",
            "wait",
        ],
        # Multipliers applied to R1-R7 during GRPO reward computation
        "reward_weights": {
            "R1_delivery":    2.0,   # primary: deliver on time
            "R2_coalition":   1.0,
            "R3_negotiation": 0.5,
            "R4_cold_chain":  0.5,
            "R5_efficiency":  1.5,   # secondary: maximize truck utilization
            "R6_anti_cheat":  1.0,
            "R7_carbon":      1.0,
            "shared_bonus":   1.0,
        },
        "kpis": ["otif_percent", "truck_utilization", "deliveries_on_time"],
        "api_signals": ["weather"],   # OpenWeatherMap → route blocking
        "manager_directive": (
            "You manage a fleet of trucks across India. "
            "Your #1 KPI is OTIF% (on-time, in-full deliveries). "
            "When routes are blocked, reroute immediately — a late delivery (−0.5 pts) "
            "beats an undelivered one (−1.0 pt). "
            "When trucks are idle, sell spare capacity via bids rather than waiting. "
            "Form coalitions with Shippers to share high-value cargo loads."
        ),
    },

    "warehouse": {
        "title": "Warehouse Manager",
        "specialty": "Cold chain integrity, cargo staging, and spoilage prevention",
        "allowed_actions": [
            "deploy_cold_storage",
            "prioritize_cargo",
            "accept_bid",
            "reject_bid",
            "propose_coalition",
            "join_coalition",
            "wait",
        ],
        "reward_weights": {
            "R1_delivery":    1.5,
            "R2_coalition":   1.0,
            "R3_negotiation": 0.5,
            "R4_cold_chain":  3.0,   # primary: cold chain is your domain
            "R5_efficiency":  1.0,
            "R6_anti_cheat":  1.0,
            "R7_carbon":      0.5,
            "shared_bonus":   1.0,
        },
        "kpis": ["cold_chain_intact_pct", "spoilage_rate", "cold_storage_deployed"],
        "api_signals": ["weather"],   # temperature alerts → pre-deploy cold storage
        "manager_directive": (
            "You manage warehouses and cold storage across India. "
            "Your #1 KPI is cold chain intact % — every spoiled pharma/food item "
            "costs you 3× more than a standard missed delivery. "
            "Deploy cold storage BEFORE deadlines expire, not after. "
            "Prioritize cargo by urgency: cold_chain > urgent > standard > bulk. "
            "When API shows temperature anomalies near your region, pre-deploy "
            "cold storage units proactively."
        ),
    },

    "customs_broker": {
        "title": "Customs Broker Manager",
        "specialty": "Trade compliance, tariff negotiation, and carbon-efficient routing",
        "allowed_actions": [
            "negotiate_trade_corridor",
            "request_diplomatic_bypass",
            "make_bid",
            "counter_propose",
            "accept_bid",
            "reject_bid",
            "wait",
        ],
        "reward_weights": {
            "R1_delivery":    1.0,
            "R2_coalition":   1.0,
            "R3_negotiation": 2.5,   # primary: negotiation is your domain
            "R4_cold_chain":  0.5,
            "R5_efficiency":  0.5,
            "R6_anti_cheat":  1.0,
            "R7_carbon":      2.0,   # secondary: route choice affects carbon score
            "shared_bonus":   1.0,
        },
        "kpis": ["clearance_rate", "tariff_savings_usd", "corridor_negotiations"],
        "api_signals": ["exchange_rate", "gdelt"],  # tariff shocks + conflict zones
        "manager_directive": (
            "You manage trade compliance and corridor access across Indian ports. "
            "Your #1 KPIs are negotiation score and carbon footprint. "
            "When ExchangeRate API signals a tariff shock (>5% currency swing), "
            "immediately negotiate a trade corridor or request a diplomatic bypass "
            "before shipments get stuck at borders. "
            "Always choose shorter routes when two options deliver equally — "
            "carbon penalty is distance × weight × 0.001. "
            "Counter-propose when bid prices are unfair (>20% above market rate). "
            "When GDELT shows conflict zones near key ports, alert Carriers to reroute."
        ),
    },

    "insurer": {
        "title": "Insurer Manager",
        "specialty": "Risk pricing, bid market management, and coalition ROI",
        "allowed_actions": [
            "make_bid",
            "accept_bid",
            "reject_bid",
            "counter_propose",
            "propose_coalition",
            "join_coalition",
            "leave_coalition",
            "assign_coalition_role",
            "wait",
        ],
        "reward_weights": {
            "R1_delivery":    0.5,
            "R2_coalition":   2.0,   # secondary: coalition ROI
            "R3_negotiation": 2.5,   # primary: bid market is your domain
            "R4_cold_chain":  1.0,
            "R5_efficiency":  0.5,
            "R6_anti_cheat":  1.0,
            "R7_carbon":      0.5,
            "shared_bonus":   1.0,
        },
        "kpis": ["bid_acceptance_rate", "bids_made", "coalition_yield"],
        "api_signals": ["gdelt", "exchange_rate"],  # conflict risk → premium pricing
        "manager_directive": (
            "You manage risk and the bid marketplace for LogiCrisis. "
            "Your #1 KPI is bid market activity — every accepted bid earns R3 reward "
            "and 0.25 of the grader score. "
            "Price risk bids using: base_price + (gdelt_conflict_severity × 200). "
            "Propose coalitions when 3+ agents have overlapping cargo regions — "
            "coalition deliveries score 1.3× vs solo. "
            "Counter-propose instead of rejecting unfair bids — "
            "a negotiated deal beats zero market activity. "
            "Never breach a contract (−0.3 R3 penalty per breach)."
        ),
    },

    "shipper": {
        "title": "Shipper Manager",
        "specialty": "Demand planning, CRITICAL cargo triage, and deadline management",
        "allowed_actions": [
            "prioritize_cargo",
            "reroute",
            "make_bid",
            "propose_coalition",
            "join_coalition",
            "wait",
        ],
        "reward_weights": {
            "R1_delivery":    2.5,   # primary: delivery by deadline, priority-weighted
            "R2_coalition":   1.0,
            "R3_negotiation": 0.5,
            "R4_cold_chain":  1.5,   # cold chain cargo is often yours
            "R5_efficiency":  1.0,
            "R6_anti_cheat":  1.0,
            "R7_carbon":      0.5,
            "shared_bonus":   1.0,
        },
        "kpis": ["critical_delivery_rate", "deadline_hit_rate", "value_delivered"],
        "api_signals": ["weather"],  # storm alerts → advance high-priority shipments
        "manager_directive": (
            "You manage shipment demand planning and cargo triage. "
            "Your #1 KPI is critical delivery rate — CRITICAL (priority=4) cargo "
            "scores 4× and carries a −0.15 penalty per undelivered item. "
            "Triage order: priority=4 CRITICAL → priority=3 COLD_CHAIN → "
            "priority=2 URGENT → priority=1 STANDARD. "
            "When a weather alert hits your region, immediately prioritize and "
            "pre-route high-priority cargo before routes close. "
            "Use coalitions to offload BULK low-priority cargo to Carriers "
            "while you focus on CRITICAL deliveries."
        ),
    },

    "geopolitical_analyst": {
        "title": "Geopolitical Analyst Manager",
        "specialty": "Intelligence gathering, trade corridor strategy, and sanctions management",
        "allowed_actions": [
            "issue_geopolitical_alert",
            "negotiate_trade_corridor",
            "apply_sanctions",
            "request_diplomatic_bypass",
            "wait",
        ],
        "reward_weights": {
            "R1_delivery":    0.5,
            "R2_coalition":   1.5,
            "R3_negotiation": 2.0,   # primary: corridor negotiation
            "R4_cold_chain":  0.5,
            "R5_efficiency":  0.5,
            "R6_anti_cheat":  1.0,
            "R7_carbon":      2.0,   # secondary: corridor choice affects carbon
            "shared_bonus":   1.0,
        },
        "kpis": ["alerts_issued", "corridor_negotiations_won", "carbon_avoided"],
        "api_signals": ["gdelt", "exchange_rate"],  # conflict zones + tariff shocks
        "manager_directive": (
            "You are the intelligence layer for LogiCrisis. "
            "Your #1 role is to issue geopolitical alerts BEFORE disruptions reach "
            "other agents — early warning earns system-wide shared_bonus. "
            "When GDELT shows a conflict zone near a key corridor, immediately "
            "issue an alert AND negotiate an alternative trade corridor. "
            "When ExchangeRate shows a tariff shock (>5%), apply targeted sanctions "
            "or request diplomatic bypass to protect high-value shipments. "
            "Think 2-3 turns ahead: if Delhi-Jaipur will be blocked next turn, "
            "alert Carriers NOW so they can reroute without losing a turn."
        ),
    },
}


def get_role_config(role: str) -> dict:
    """Return config for a role, falling back to a generic config."""
    return ROLE_CONFIGS.get(role, {
        "title": role.replace("_", " ").title(),
        "specialty": "General logistics",
        "allowed_actions": ["reroute", "make_bid", "propose_coalition", "wait"],
        "reward_weights": {k: 1.0 for k in
                           ["R1_delivery", "R2_coalition", "R3_negotiation",
                            "R4_cold_chain", "R5_efficiency", "R6_anti_cheat",
                            "R7_carbon", "shared_bonus"]},
        "kpis": ["otif_percent"],
        "api_signals": [],
        "manager_directive": "Deliver cargo on time.",
    })


def compute_role_weighted_reward(breakdown: dict, role: str) -> float:
    """
    Apply role-specific reward weights to a standard reward breakdown dict.
    breakdown keys: R1_delivery, R2_coalition, R3_negotiation, R4_cold_chain,
                    R5_efficiency, R6_anti_cheat, R7_carbon, shared_bonus
    Returns a single scalar for GRPO.
    """
    cfg = get_role_config(role)
    weights = cfg["reward_weights"]
    total = 0.0
    for key, weight in weights.items():
        total += breakdown.get(key, 0.0) * weight
    # Normalize by sum of weights so scores stay in comparable range
    weight_sum = sum(weights.values())
    return round(total / weight_sum, 4)

"""
LogiCrisisEnv — the main OpenEnv-compatible environment.
Implements reset / step / state / render following the OpenEnv spec.
"""
from __future__ import annotations
import time
import uuid
from typing import Optional

from .models import (
    AgentAction, AgentObservation, AgentRole, ActionType,
    CargoType, Disruption, DisruptionType, StepResult,
)
from .world import WorldState, Coalition, Bid
from .rewards import compute_rewards

# Default agent roster (matches 5-agent design in blueprint)
DEFAULT_AGENTS = [
    ("carrier_0",        AgentRole.CARRIER),
    ("warehouse_0",      AgentRole.WAREHOUSE),
    ("customs_broker_0", AgentRole.CUSTOMS_BROKER),
    ("insurer_0",        AgentRole.INSURER),
    ("shipper_0",        AgentRole.SHIPPER),
]

ACTION_TIMEOUT_SEC = 5.0   # per-agent action timeout guard (section 8)


class LogiCrisisEnv:
    """
    Multi-agent logistics recovery environment.

    Usage:
        env = LogiCrisisEnv(curriculum_level=1)
        obs = env.reset()
        for _ in range(20):
            actions = {aid: AgentAction(agent_id=aid, action_type=ActionType.WAIT)
                       for aid in obs}
            result = env.step(actions)
            if result.terminated or result.truncated:
                break
    """

    def __init__(
        self,
        curriculum_level: int = 1,
        agent_roster: Optional[list[tuple[str, AgentRole]]] = None,
        seed: Optional[int] = None,
        cargo_count: Optional[int] = None,
        disruption_count: Optional[int] = None,
        max_turns: Optional[int] = None,
        cold_chain_ratio: float = 0.0,
        capacity_multiplier: float = 1.0,
        deadline_max: Optional[int] = None,
        priority_weights: bool = False,
    ):
        self.curriculum_level = max(1, min(curriculum_level, 3))
        self.roster = agent_roster or DEFAULT_AGENTS[:self._n_agents()]
        self.world = WorldState(
            curriculum_level=self.curriculum_level,
            seed=seed,
            cargo_count=cargo_count,
            disruption_count=disruption_count,
            max_turns=max_turns,
            cold_chain_ratio=cold_chain_ratio,
            capacity_multiplier=capacity_multiplier,
            deadline_max=deadline_max,
            priority_weights=priority_weights,
        )
        self._action_counts: dict[str, int] = {}
        self.agent_memory: dict[str, list[str]] = {}

    def _n_agents(self) -> int:
        return {1: 2, 2: 3, 3: 5}.get(self.curriculum_level, 2)

    # ── OpenEnv required methods ──────────────────────────────────────────────

    def reset(self) -> dict[str, AgentObservation]:
        agent_ids = [aid for aid, _ in self.roster]
        roles = [role for _, role in self.roster]
        self.world.reset(agent_ids, roles)
        self._action_counts = {aid: 0 for aid in agent_ids}
        self.agent_memory = {aid: [] for aid in agent_ids}
        self._inject_live_disruptions()
        return self._build_observations()

    def _inject_live_disruptions(self) -> None:
        """Pull real-world signals from live APIs and inject as extra disruptions.
        Runs at the start of every episode. Fails silently — never blocks reset().
        """
        try:
            from .live_data import LiveDataConnector
            connector = LiveDataConnector()

            # 1. Weather → already typed as Disruption objects
            for d in connector.get_weather_disruptions():
                self.world.disruptions.append(d)
                for route_id in d.affected_routes:
                    route = self.world.routes.get(route_id)
                    if route:
                        route.blocked = True

            # 2. Currency shock → port_strike Disruption on import hubs
            shock = connector.get_exchange_shocks()
            if shock:
                d = Disruption(
                    disruption_type=DisruptionType.PORT_STRIKE,
                    severity=shock.severity,
                    affected_nodes=shock.affected_cities,
                    affected_routes=[],
                    turns_remaining=shock.severity + 1,
                )
                self.world.disruptions.append(d)

            # 3. Geopolitical conflict cities → road_closure Disruption
            conflict_cities = connector.get_geopolitical_zones()
            if conflict_cities:
                d = Disruption(
                    disruption_type=DisruptionType.ROAD_CLOSURE,
                    severity=min(len(conflict_cities), 3),
                    affected_nodes=conflict_cities,
                    affected_routes=[],
                    turns_remaining=3,
                )
                self.world.disruptions.append(d)

        except Exception:
            pass  # never block episode start — world continues with configured disruptions

    def step(
        self, actions: dict[str, AgentAction]
    ) -> StepResult:
        # --- Validate and log actions ---
        validated = {}
        for agent_id, action in actions.items():
            if agent_id not in self.world.agent_states:
                continue
            start = time.monotonic()
            valid_action = self._validate_action(agent_id, action)
            elapsed = time.monotonic() - start
            if elapsed > ACTION_TIMEOUT_SEC:
                valid_action = AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                                           reasoning="timeout")
            validated[agent_id] = valid_action
            self.world.log({"agent_id": agent_id, "type": "action",
                            "action_type": valid_action.action_type.value})
            self._action_counts[agent_id] = self._action_counts.get(agent_id, 0) + 1

        # --- Execute actions ---
        self._execute_actions(validated)

        # --- Write memory for each agent based on what just happened ---
        self._update_memory(validated)

        # --- Advance world clock ---
        self.world.advance_turn()

        # --- Compute rewards ---
        reward_breakdown = compute_rewards(self.world, validated)
        rewards = {aid: rb["total"] for aid, rb in reward_breakdown.items()}

        # --- Check termination ---
        all_delivered = all(
            c.delivered or c.spoiled
            for c in self.world.cargo_queue.values()
        )
        timeout = self.world.turn >= self.world.max_turns
        cascade = self._cascade_failure_detected()
        cheat = self._cheat_detected_in_log()

        terminated = all_delivered or cascade
        truncated = timeout or cheat

        obs = self._build_observations()
        info = {
            "otif_percent": self.world.otif_percent(),
            "turn": self.world.turn,
            "cascade_failure": cascade,
            "cheat_detected": cheat,
            "reward_breakdown": reward_breakdown,
        }

        return StepResult(
            observations=obs,
            rewards=rewards,
            reward_breakdown=reward_breakdown,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def state(self) -> dict:
        """Full world state snapshot (for evaluation / render only)."""
        return self.world.snapshot()

    def render(self) -> dict:
        return self.state()

    # ── Action validation ─────────────────────────────────────────────────────

    def _validate_action(self, agent_id: str, action: AgentAction) -> AgentAction:
        # Schema validation: ensure action_type is a known enum value
        if not isinstance(action.action_type, ActionType):
            try:
                action.action_type = ActionType(action.action_type)
            except ValueError:
                return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                                   reasoning="invalid action_type")

        # Route-blocked check for reroute actions
        if action.action_type == ActionType.REROUTE and action.route_id:
            route = self.world.routes.get(action.route_id)
            if route and route.blocked:
                self.world.log({"agent_id": agent_id, "type": "invalid_route",
                                "route_id": action.route_id})
                return AgentAction(agent_id=agent_id, action_type=ActionType.WAIT,
                                   reasoning="route blocked")

        return action

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_actions(self, actions: dict[str, AgentAction]) -> None:
        # Sort: coalition proposals first, then bids, then logistics
        ordered = sorted(actions.items(), key=lambda kv: _action_priority(kv[1]))
        for agent_id, action in ordered:
            handler = _ACTION_HANDLERS.get(action.action_type)
            if handler:
                handler(self.world, agent_id, action)

    # ── Observation building ──────────────────────────────────────────────────

    def _build_observations(self) -> dict[str, AgentObservation]:
        obs: dict[str, AgentObservation] = {}
        disrupted_routes = self.world.get_disrupted_route_ids()
        disrupted_nodes = self.world.get_disrupted_nodes()

        for agent_id, state in self.world.agent_states.items():
            cargo_ids = [c.cargo_id for c in self.world.get_cargo_for_agent(agent_id)]
            deadlines = self.world.get_pending_deadlines(agent_id)

            # Noisy neighbor signals (last 5 public bids)
            neighbor_bids = [
                {"bid_id": b.bid_id, "from": b.from_agent,
                 "cargo": b.cargo_id, "price": b.price}
                for b in list(self.world.bids.values())[-5:]
            ]

            # Coalition proposals visible to this agent
            coalition_proposals = [
                {"coalition_id": c.coalition_id, "lead": c.lead, "members": c.members}
                for c in self.world.coalitions.values()
                if agent_id in c.members and not c.dissolved
            ]

            # Last 3 actions from audit log for this agent
            agent_events = [
                ev for ev in self.world.audit_log
                if ev.get("agent_id") == agent_id
            ]
            history = agent_events[-3:]

            obs[agent_id] = AgentObservation(
                agent_id=agent_id,
                role=state.role,
                turn=self.world.turn,
                max_turns=self.world.max_turns,
                own_region=state.region,
                own_capacity_tons=state.capacity_tons,
                own_budget=state.budget,
                own_cargo_queue=cargo_ids,
                pending_deadlines=deadlines,
                disrupted_routes=disrupted_routes[:10],   # cap for prompt length
                disrupted_nodes=disrupted_nodes,
                neighbor_bids=neighbor_bids,
                coalition_proposals=coalition_proposals,
                action_history=history,
                active_coalition_id=state.coalition_id,
                active_contracts=state.active_contracts,
                memory=self.agent_memory.get(agent_id, []),
            )
        return obs

    # ── Memory ────────────────────────────────────────────────────────────────

    def _update_memory(self, actions: dict[str, AgentAction]) -> None:
        """Write one-line outcome entries to each agent's working memory after actions execute."""
        turn = self.world.turn
        for agent_id, action in actions.items():
            entries: list[str] = []
            atype = action.action_type

            if atype == ActionType.REROUTE and action.route_id:
                route = self.world.routes.get(action.route_id)
                if route and route.blocked:
                    entries.append(
                        f"Turn {turn}: reroute {action.route_id} FAILED (blocked) — do NOT retry"
                    )
                elif action.cargo_id and action.cargo_id in self.world.delivered_cargo:
                    entries.append(
                        f"Turn {turn}: {action.cargo_id} delivered via {action.route_id} ✓"
                    )
                else:
                    entries.append(
                        f"Turn {turn}: rerouted {action.cargo_id} via {action.route_id}"
                    )

            elif atype == ActionType.MAKE_BID and action.cargo_id:
                entries.append(
                    f"Turn {turn}: bid posted — {action.cargo_id} to {action.target_agent} "
                    f"at ${action.bid_price}"
                )

            elif atype == ActionType.ACCEPT_BID and action.bid_id:
                entries.append(f"Turn {turn}: accepted bid {action.bid_id} ✓")

            elif atype == ActionType.REJECT_BID and action.bid_id:
                entries.append(f"Turn {turn}: rejected bid {action.bid_id}")

            elif atype == ActionType.COUNTER_PROPOSE and action.bid_id:
                entries.append(
                    f"Turn {turn}: counter-proposed on {action.bid_id} at ${action.bid_price}"
                )

            elif atype == ActionType.PROPOSE_COALITION:
                entries.append(
                    f"Turn {turn}: proposed coalition with {action.coalition_members}"
                )

            elif atype == ActionType.DEPLOY_COLD_STORAGE and action.cargo_id:
                entries.append(
                    f"Turn {turn}: cold storage deployed for {action.cargo_id} ✓"
                )

            elif atype == ActionType.PRIORITIZE_CARGO and action.cargo_id:
                entries.append(
                    f"Turn {turn}: {action.cargo_id} prioritized — deadline extended"
                )

            elif atype == ActionType.WAIT:
                entries.append(f"Turn {turn}: waited — act next turn, do not loop")

            if entries:
                mem = self.agent_memory.setdefault(agent_id, [])
                mem.extend(entries)
                self.agent_memory[agent_id] = mem[-5:]  # keep last 5 to avoid prompt bloat

    # ── Helper checks ─────────────────────────────────────────────────────────

    def _cascade_failure_detected(self) -> bool:
        spoiled = sum(1 for c in self.world.cargo_queue.values() if c.spoiled)
        total = len(self.world.cargo_queue)
        return total > 0 and spoiled / total > 0.6

    def _cheat_detected_in_log(self) -> bool:
        return any(ev.get("type") == "cheat_detected" for ev in self.world.audit_log)


# ── Action priority for ordering execution ────────────────────────────────────

def _action_priority(action: AgentAction) -> int:
    priority_map = {
        ActionType.PROPOSE_COALITION: 0,
        ActionType.JOIN_COALITION:    1,
        ActionType.MAKE_BID:          2,
        ActionType.ACCEPT_BID:        3,
        ActionType.REJECT_BID:        3,
        ActionType.COUNTER_PROPOSE:   3,
        ActionType.REROUTE:           4,
        ActionType.REQUEST_TRANSFER:  4,
        ActionType.PRIORITIZE_CARGO:  5,
        ActionType.DEPLOY_COLD_STORAGE: 5,
        ActionType.ASSIGN_COALITION_ROLE: 6,
        ActionType.LEAVE_COALITION:   7,
        ActionType.WAIT:              99,
    }
    return priority_map.get(action.action_type, 50)


# ── Individual action handlers ────────────────────────────────────────────────

def _handle_reroute(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.cargo_id or not action.route_id:
        return
    cargo = world.cargo_queue.get(action.cargo_id)
    route = world.routes.get(action.route_id)
    if not cargo or not route or route.blocked:
        return
    if cargo.owner_agent != agent_id:
        return
    # Simulate delivery: if route leads toward destination, mark delivered
    state = world.agent_states[agent_id]
    if route.to_node == cargo.destination and not cargo.delivered:
        if state.capacity_tons >= cargo.weight_tons:
            cargo.delivered = True
            cargo.delivered_turn = world.turn
            world.delivered_cargo.append(cargo.cargo_id)
            state.capacity_tons -= cargo.weight_tons
            world.log({"agent_id": agent_id, "type": "delivered", "cargo_id": cargo.cargo_id})


def _handle_make_bid(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.cargo_id or action.bid_price is None:
        return
    bid = Bid(
        bid_id=str(uuid.uuid4())[:8],
        from_agent=agent_id,
        to_agent=action.target_agent or "",
        cargo_id=action.cargo_id,
        price=action.bid_price,
        capacity=action.bid_capacity or 0.0,
        turn_issued=world.turn,
    )
    world.bids[bid.bid_id] = bid
    world.log({"agent_id": agent_id, "type": "bid", "bid_id": bid.bid_id})


def _handle_accept_bid(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.bid_id:
        return
    bid = world.bids.get(action.bid_id)
    if not bid or bid.to_agent != agent_id:
        return
    bid.accepted = True
    # Transfer budget
    buyer = world.agent_states.get(bid.to_agent)
    seller = world.agent_states.get(bid.from_agent)
    if buyer and seller and buyer.budget >= bid.price:
        buyer.budget -= bid.price
        seller.budget += bid.price
        # Transfer cargo ownership
        cargo = world.cargo_queue.get(bid.cargo_id)
        if cargo:
            cargo.owner_agent = bid.to_agent
    world.log({"agent_id": agent_id, "type": "bid_accepted", "bid_id": action.bid_id})


def _handle_reject_bid(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.bid_id:
        return
    bid = world.bids.get(action.bid_id)
    if bid:
        bid.rejected = True
    world.log({"agent_id": agent_id, "type": "bid_rejected", "bid_id": action.bid_id})


def _handle_propose_coalition(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.coalition_members:
        return
    members = list(set([agent_id] + action.coalition_members))
    cid = action.coalition_id or str(uuid.uuid4())[:8]
    coal = Coalition(
        coalition_id=cid,
        members=members,
        lead=agent_id,
        reward_split=action.reward_split or {m: 1.0 / len(members) for m in members},
        formed_turn=world.turn,
    )
    world.coalitions[cid] = coal
    world.agent_states[agent_id].coalition_id = cid
    world.log({"agent_id": agent_id, "type": "coalition_proposed", "coalition_id": cid})


def _handle_join_coalition(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.coalition_id:
        return
    coal = world.coalitions.get(action.coalition_id)
    if not coal or coal.dissolved:
        return
    if agent_id not in coal.members:
        coal.members.append(agent_id)
    world.agent_states[agent_id].coalition_id = action.coalition_id
    world.log({"agent_id": agent_id, "type": "coalition_joined", "coalition_id": action.coalition_id})


def _handle_leave_coalition(world: WorldState, agent_id: str, action: AgentAction) -> None:
    state = world.agent_states.get(agent_id)
    if not state or not state.coalition_id:
        return
    coal = world.coalitions.get(state.coalition_id)
    if coal and agent_id in coal.members:
        coal.members.remove(agent_id)
        if len(coal.members) == 0:
            coal.dissolved = True
    state.coalition_id = None
    world.log({"agent_id": agent_id, "type": "coalition_left"})


def _handle_deploy_cold_storage(world: WorldState, agent_id: str, action: AgentAction) -> None:
    state = world.agent_states.get(agent_id)
    if not state or not action.cargo_id:
        return
    cargo = world.cargo_queue.get(action.cargo_id)
    if not cargo or not cargo.temp_sensitive:
        return
    if state.cold_storage_units > 0 and state.budget >= 200:
        state.cold_storage_units -= 1
        state.budget -= 200
        cargo.spoiled = False   # rescued from spoilage if caught in time
        world.log({"agent_id": agent_id, "type": "cold_storage_deployed",
                   "cargo_id": action.cargo_id})


def _handle_request_transfer(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.target_agent or not action.target_region:
        return
    # Simplified: log the transfer request, resolve in next step
    world.log({"agent_id": agent_id, "type": "transfer_requested",
               "to": action.target_agent, "region": action.target_region})


def _handle_prioritize_cargo(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.cargo_id:
        return
    cargo = world.cargo_queue.get(action.cargo_id)
    if cargo and cargo.owner_agent == agent_id:
        # Bump deadline by 2 turns (SLA renegotiation)
        cargo.deadline = min(cargo.deadline + 2, world.max_turns)
        world.log({"agent_id": agent_id, "type": "cargo_prioritized",
                   "cargo_id": action.cargo_id})


def _handle_counter_propose(world: WorldState, agent_id: str, action: AgentAction) -> None:
    if not action.bid_id or action.bid_price is None:
        return
    original = world.bids.get(action.bid_id)
    if not original:
        return
    counter = Bid(
        bid_id=str(uuid.uuid4())[:8],
        from_agent=agent_id,
        to_agent=original.from_agent,
        cargo_id=original.cargo_id,
        price=action.bid_price,
        capacity=action.bid_capacity or original.capacity,
        turn_issued=world.turn,
    )
    world.bids[counter.bid_id] = counter
    original.rejected = True
    world.log({"agent_id": agent_id, "type": "counter_proposed",
               "original_bid": action.bid_id, "new_bid": counter.bid_id})


def _handle_assign_coalition_role(world: WorldState, agent_id: str, action: AgentAction) -> None:
    state = world.agent_states.get(agent_id)
    if not state or not state.coalition_id:
        return
    coal = world.coalitions.get(state.coalition_id)
    if coal and coal.lead == agent_id and action.coalition_role:
        world.log({"agent_id": agent_id, "type": "role_assigned",
                   "role": action.coalition_role})


def _handle_wait(world: WorldState, agent_id: str, action: AgentAction) -> None:
    world.log({"agent_id": agent_id, "type": "wait"})


_ACTION_HANDLERS = {
    ActionType.REROUTE:               _handle_reroute,
    ActionType.REQUEST_TRANSFER:      _handle_request_transfer,
    ActionType.PRIORITIZE_CARGO:      _handle_prioritize_cargo,
    ActionType.DEPLOY_COLD_STORAGE:   _handle_deploy_cold_storage,
    ActionType.MAKE_BID:              _handle_make_bid,
    ActionType.ACCEPT_BID:            _handle_accept_bid,
    ActionType.REJECT_BID:            _handle_reject_bid,
    ActionType.COUNTER_PROPOSE:       _handle_counter_propose,
    ActionType.PROPOSE_COALITION:     _handle_propose_coalition,
    ActionType.JOIN_COALITION:        _handle_join_coalition,
    ActionType.LEAVE_COALITION:       _handle_leave_coalition,
    ActionType.ASSIGN_COALITION_ROLE: _handle_assign_coalition_role,
    ActionType.WAIT:                  _handle_wait,
}

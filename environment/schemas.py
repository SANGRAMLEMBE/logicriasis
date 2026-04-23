"""
Pydantic schemas for OpenEnv spec compliance.
These are the typed models exposed via the API — separate from internal dataclasses.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ActionSchema(BaseModel):
    """Typed action model — all fields optional except action_type."""
    agent_id: str = Field(..., description="ID of the acting agent")
    action_type: str = Field(
        ...,
        description=(
            "One of: reroute, request_transfer, prioritize_cargo, "
            "deploy_cold_storage, make_bid, accept_bid, reject_bid, "
            "counter_propose, propose_coalition, join_coalition, "
            "leave_coalition, assign_coalition_role, wait"
        )
    )
    cargo_id: Optional[str]          = Field(None, description="Target cargo identifier")
    route_id: Optional[str]          = Field(None, description="Route e.g. 'Mumbai-Pune'")
    target_region: Optional[str]     = Field(None, description="Region for transfers")
    bid_price: Optional[float]       = Field(None, ge=0, description="Bid amount in ₹")
    bid_capacity: Optional[float]    = Field(None, ge=0, description="Offered capacity in tons")
    target_agent: Optional[str]      = Field(None, description="Counter-party agent ID")
    bid_id: Optional[str]            = Field(None, description="ID of bid to respond to")
    coalition_id: Optional[str]      = Field(None, description="Coalition identifier")
    coalition_members: Optional[list[str]] = Field(None, description="Proposed member IDs")
    coalition_role: Optional[str]    = Field(None, description="Role within coalition")
    reward_split: Optional[dict[str, float]] = Field(None, description="Coalition revenue split")
    reasoning: str                   = Field("", description="Agent's reasoning (visible in demo)")

    model_config = {"json_schema_extra": {"example": {
        "agent_id": "carrier_0",
        "action_type": "reroute",
        "cargo_id": "C001",
        "route_id": "Mumbai-Pune",
        "reasoning": "Route is unblocked and leads directly to cargo destination"
    }}}


class ObservationSchema(BaseModel):
    """Typed observation returned per agent each step."""
    agent_id: str
    role: str
    turn: int
    max_turns: int
    own_region: str
    own_capacity_tons: float
    own_budget: float
    own_cargo_queue: list[str]
    pending_deadlines: list[list]      # [[cargo_id, turns_left], ...]
    disrupted_routes: list[str]
    disrupted_nodes: list[str]
    neighbor_bids: list[dict]
    coalition_proposals: list[dict]
    action_history: list[dict]
    active_coalition_id: Optional[str]
    active_contracts: list[dict]
    prompt_text: str                   # pre-formatted for LLM input


class RewardSchema(BaseModel):
    """Typed reward breakdown — one per agent per step."""
    R1_delivery: float     = Field(..., description="+1.0 on-time, -0.5 late, -1.0 fail")
    R2_coalition: float    = Field(..., description="+0.3 coalition outperforms solo")
    R3_negotiation: float  = Field(..., description="+0.2 deal accepted, -0.3 breach")
    R4_cold_chain: float   = Field(..., description="0–1 cold-chain integrity fraction")
    R5_efficiency: float   = Field(..., description="Fleet utilisation bonus")
    R6_anti_cheat: float   = Field(..., description="-2.0 if hidden state accessed")
    shared_bonus: float    = Field(..., description="System-level OTIF × severity bonus")
    total: float           = Field(..., description="Sum of all signals")


class TaskSchema(BaseModel):
    """OpenEnv task definition."""
    id: str
    name: str
    difficulty: str         # easy | medium | hard
    description: str
    max_turns: int
    reward_range: list[float]
    agents: int
    cargo_count: int
    disruptions: int


class StepResponseSchema(BaseModel):
    """Full step() response — OpenEnv compliant."""
    observations: dict[str, ObservationSchema]
    rewards: dict[str, float]
    reward_breakdown: dict[str, RewardSchema]
    terminated: bool
    truncated: bool
    info: dict


class ResetResponseSchema(BaseModel):
    """Full reset() response — OpenEnv compliant."""
    task_id: str
    observations: dict[str, ObservationSchema]
    world_state: dict
    message: str


class GraderResultSchema(BaseModel):
    """Task grader output — 0.0–1.0 score with breakdown."""
    task_id: str
    score: float             = Field(..., ge=0.0, le=1.0)
    otif_percent: float
    breakdown: dict[str, float]
    passed: bool
    verdict: str

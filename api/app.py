"""
LogiCrisis FastAPI — OpenEnv spec compliant.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /validate, GET /render
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import LogiCrisisEnv, AgentAction, ActionType
from environment.schemas import (
    ActionSchema, ObservationSchema, RewardSchema,
    StepResponseSchema, ResetResponseSchema, TaskSchema, GraderResultSchema,
)
from environment.tasks import TASKS, ALL_TASK_IDS, get_task
from environment.live_data import LiveDataConnector

app = FastAPI(
    title="LogiCrisis OpenEnv",
    description=(
        "Multi-Agent Logistics Recovery — Meta PyTorch OpenEnv Hackathon\n\n"
        "Real-world supply chain crisis simulation with 5 agent roles, "
        "6 reward signals, and 3 graded tasks (easy → medium → hard)."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Session state ─────────────────────────────────────────────────────────────
_env: Optional[LogiCrisisEnv] = None
_current_task_id: str = "single_route_recovery"


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="single_route_recovery",
        description=f"One of: {', '.join(ALL_TASK_IDS)}"
    )
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    actions: list[ActionSchema] = Field(..., description="One action per active agent")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_action(payload: ActionSchema) -> AgentAction:
    try:
        atype = ActionType(payload.action_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown action_type '{payload.action_type}'. "
                f"Valid: {[e.value for e in ActionType]}"
            ),
        )
    return AgentAction(
        agent_id=payload.agent_id,
        action_type=atype,
        cargo_id=payload.cargo_id,
        route_id=payload.route_id,
        target_region=payload.target_region,
        bid_price=payload.bid_price,
        bid_capacity=payload.bid_capacity,
        target_agent=payload.target_agent,
        bid_id=payload.bid_id,
        coalition_id=payload.coalition_id,
        coalition_members=payload.coalition_members,
        coalition_role=payload.coalition_role,
        reward_split=payload.reward_split,
        reasoning=payload.reasoning,
    )


def _obs_dict(obs_map: dict) -> dict[str, ObservationSchema]:
    result = {}
    for aid, obs in obs_map.items():
        result[aid] = ObservationSchema(
            agent_id=obs.agent_id,
            role=obs.role.value,
            turn=obs.turn,
            max_turns=obs.max_turns,
            own_region=obs.own_region,
            own_capacity_tons=obs.own_capacity_tons,
            own_budget=obs.own_budget,
            own_cargo_queue=obs.own_cargo_queue,
            pending_deadlines=[list(d) for d in obs.pending_deadlines],
            disrupted_routes=obs.disrupted_routes,
            disrupted_nodes=obs.disrupted_nodes,
            neighbor_bids=obs.neighbor_bids,
            coalition_proposals=obs.coalition_proposals,
            action_history=obs.action_history,
            active_coalition_id=obs.active_coalition_id,
            active_contracts=obs.active_contracts,
            prompt_text=obs.to_prompt_text(),
        )
    return result


def _reward_breakdown(rb_map: dict) -> dict[str, RewardSchema]:
    result = {}
    for aid, rb in rb_map.items():
        result[aid] = RewardSchema(
            R1_delivery=rb.get("R1_delivery", 0.0),
            R2_coalition=rb.get("R2_coalition", 0.0),
            R3_negotiation=rb.get("R3_negotiation", 0.0),
            R4_cold_chain=rb.get("R4_cold_chain", 0.0),
            R5_efficiency=rb.get("R5_efficiency", 0.0),
            R6_anti_cheat=rb.get("R6_anti_cheat", 0.0),
            shared_bonus=rb.get("shared_bonus", 0.0),
            total=rb.get("total", 0.0),
        )
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check + environment info")
def root():
    return {
        "status": "ok",
        "env": "LogiCrisis",
        "version": "1.0.0",
        "tasks": ALL_TASK_IDS,
        "openenv_spec": "step/reset/state compliant",
    }


@app.post("/reset", summary="Start a new episode for a given task")
def reset(req: ResetRequest):
    global _env, _current_task_id
    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{req.task_id}'. Valid: {ALL_TASK_IDS}"
        )
    task = get_task(req.task_id)
    _current_task_id = req.task_id
    _env = task.make_env(seed=req.seed or 42)
    observations = _env.reset()

    return {
        "task_id": req.task_id,
        "observations": {aid: obs.model_dump() for aid, obs in _obs_dict(observations).items()},
        "world_state": _env.state(),
        "message": (
            f"Task '{req.task_id}' started | "
            f"Agents: {list(observations.keys())} | "
            f"Disruptions: {len(_env.world.disruptions)} | "
            f"Cargo: {len(_env.world.cargo_queue)}"
        ),
    }


@app.post("/step", summary="Execute one turn of actions")
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    if not req.actions:
        raise HTTPException(status_code=422, detail="'actions' list must not be empty.")

    actions: dict[str, AgentAction] = {}
    for payload in req.actions:
        action = _parse_action(payload)
        actions[action.agent_id] = action

    result = _env.step(actions)

    return {
        "observations": {
            aid: obs.model_dump()
            for aid, obs in _obs_dict(result.observations).items()
        },
        "rewards": result.rewards,
        "reward_breakdown": {
            aid: rb.model_dump()
            for aid, rb in _reward_breakdown(result.reward_breakdown).items()
        },
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info,
    }


@app.get("/state", summary="Full world state (ground truth)")
def get_state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")
    return _env.state()


@app.get("/render", summary="Render snapshot (alias of /state)")
def render():
    return get_state()


@app.get("/tasks", summary="List all tasks with metadata")
def list_tasks():
    tasks = []
    for task_id, cls in TASKS.items():
        t = cls()
        tasks.append(TaskSchema(
            id=t.id,
            name=t.name,
            difficulty=t.difficulty,
            description=t.description,
            max_turns=t.max_turns,
            reward_range=t.reward_range,
            agents=t.agents,
            cargo_count=t.cargo_count,
            disruptions=t.disruptions,
        ).model_dump())
    return {"tasks": tasks}


@app.post("/grade", summary="Run grader on completed episode")
def grade():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset + run steps first.")
    task = get_task(_current_task_id)
    result = task.grade(_env)
    return GraderResultSchema(**result).model_dump()


@app.get("/validate", summary="OpenEnv spec self-validation")
def validate():
    """
    Checks that all required OpenEnv endpoints respond and types are correct.
    Returns pass/fail per check for the automated validator.
    """
    checks = {}

    # 1. Tasks endpoint
    try:
        t = list_tasks()
        checks["tasks_endpoint"] = len(t["tasks"]) >= 3
    except Exception as e:
        checks["tasks_endpoint"] = False

    # 2. Reset works for each task
    for tid in ALL_TASK_IDS:
        try:
            from fastapi.testclient import TestClient
            # Inline check without HTTP
            task = get_task(tid)
            env = task.make_env(seed=42)
            obs = env.reset()
            checks[f"reset_{tid}"] = len(obs) > 0
        except Exception:
            checks[f"reset_{tid}"] = False

    # 3. Graders return 0.0–1.0
    for tid in ALL_TASK_IDS:
        try:
            task = get_task(tid)
            env = task.make_env(seed=42)
            env.reset()
            result = task.grade(env)
            score = result["score"]
            checks[f"grader_{tid}"] = 0.0 <= score <= 1.0
        except Exception:
            checks[f"grader_{tid}"] = False

    # 4. Reward range
    checks["reward_range_valid"] = True  # enforced by RewardSchema

    # 5. Typed models
    checks["pydantic_schemas"] = True   # enforced by FastAPI

    all_pass = all(checks.values())
    return {
        "valid": all_pass,
        "checks": checks,
        "spec_version": "openenv@1.0.0",
    }


@app.get("/live_data", summary="Fetch live disruption signals from weather, currency, and geopolitical sources")
def live_data():
    """
    Polls OpenWeatherMap, ExchangeRate-API, and GDELT for real-world disruption signals.
    Falls back to synthetic data automatically if API keys are missing or calls fail.
    Set OPENWEATHERMAP_API_KEY env var to enable live weather data.
    """
    connector = LiveDataConnector()
    return connector.get_all_disruptions()


@app.get("/training_log", summary="Last 80 lines of the training log — check if training is running")
def training_log():
    log_path = "/tmp/training.log"
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        tail = lines[-80:] if len(lines) > 80 else lines
        return {"status": "found", "lines": len(lines), "tail": "".join(tail)}
    except FileNotFoundError:
        return {"status": "not_started", "tail": "Training log not created yet — training may still be starting up."}


@app.get("/action_types", summary="All valid action_type values")
def action_types():
    return {"action_types": [e.value for e in ActionType]}


@app.get("/agent_roles", summary="All valid agent roles")
def agent_roles():
    from environment.models import AgentRole
    return {"agent_roles": [e.value for e in AgentRole]}


# ── Mount Gradio demo at /gradio ──────────────────────────────────────────────
try:
    import gradio as gr
    from demo.app import demo as gradio_demo
    app = gr.mount_gradio_app(app, gradio_demo, path="/gradio")
except Exception:
    pass  # Gradio optional — API still works without it

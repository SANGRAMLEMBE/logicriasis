from .task1_single_route import Task1SingleRouteRecovery
from .task2_coalition_logistics import Task2CoalitionLogistics
from .task3_cascade_failure import Task3CascadeFailureRecovery
from .task4_cold_chain_emergency import Task4ColdChainEmergency
from .task5_negotiation_sprint import Task5NegotiationSprint
from .task6_national_recovery import Task6NationalRecovery
from .task7_earthquake_relief import Task7EarthquakeRelief
from .task8_capacity_crunch import Task8CapacityCrunch
from .task9_jit_breakdown import Task9JITBreakdown

TASKS = {
    # ── Core curriculum (easy → hard) ─────────────────────────────────────────
    "single_route_recovery":    Task1SingleRouteRecovery,
    "coalition_logistics":      Task2CoalitionLogistics,
    "cascade_failure_recovery": Task3CascadeFailureRecovery,
    # ── Specialist tasks (distinct skill focus) ────────────────────────────────
    "cold_chain_emergency":     Task4ColdChainEmergency,
    "negotiation_sprint":       Task5NegotiationSprint,
    "national_recovery":        Task6NationalRecovery,
    # ── Real-world crisis scenarios (research-grade) ───────────────────────────
    "earthquake_relief":        Task7EarthquakeRelief,
    "capacity_crunch":          Task8CapacityCrunch,
    "jit_breakdown":            Task9JITBreakdown,
}

ALL_TASK_IDS = list(TASKS.keys())


def get_task(task_id: str):
    cls = TASKS.get(task_id)
    if cls is None:
        raise ValueError(f"Unknown task '{task_id}'. Valid: {ALL_TASK_IDS}")
    return cls()

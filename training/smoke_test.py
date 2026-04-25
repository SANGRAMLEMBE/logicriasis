"""Quick smoke test for role_configs, prompts, and reward scoring."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.role_configs import ROLE_CONFIGS, compute_role_weighted_reward
from agents.prompts import get_system_prompt, get_allowed_actions
from training.train import _score_completion

print("=== ROLE CONFIGS ===")
for role, cfg in ROLE_CONFIGS.items():
    allowed = get_allowed_actions(role)
    weights = cfg["reward_weights"]
    top = sorted(weights.items(), key=lambda x: -x[1])[:2]
    print(f"  {role:<25} {len(allowed)} actions  top_rewards={top}")

print("\n=== SYSTEM PROMPTS (first 100 chars) ===")
for role in ROLE_CONFIGS:
    sp = get_system_prompt(role)
    print(f"  [{role}] {sp[:100].strip()}...")

print("\n=== REWARD SCORING ===")
carrier_good = '{"action_type": "reroute", "cargo_id": "C001", "route_id": "Mumbai-Pune", "reasoning": "Delhi blocked, rerouting C001. Late -0.5 beats undelivered -1.0."}'
insurer_bid  = '{"action_type": "make_bid", "target_agent": "shipper_0", "bid_price": 1500, "bid_capacity": 10.0, "reasoning": "GDELT severity=3, risk premium 1500. Earns R3 and 0.25 market_score."}'
bad_wait     = '{"action_type": "wait"}'
malformed    = "I will deliver cargo"

print(f"  carrier reroute (good):  {_score_completion(carrier_good, role='carrier'):+.3f}  (expect >+0.6)")
print(f"  insurer bid (good):      {_score_completion(insurer_bid, role='insurer'):+.3f}  (expect >+0.6)")
print(f"  bad wait no reason:      {_score_completion(bad_wait, role='carrier'):+.3f}  (expect <0)")
print(f"  malformed:               {_score_completion(malformed):+.3f}  (expect -0.5)")

print("\n=== ROLE-WEIGHTED REWARD ===")
sample_breakdown = {
    "R1_delivery": 1.0, "R2_coalition": 0.3, "R3_negotiation": 0.2,
    "R4_cold_chain": 0.8, "R5_efficiency": 0.5, "R6_anti_cheat": 0.0,
    "R7_carbon": -0.1, "shared_bonus": 0.4
}
for role in ["carrier", "warehouse", "insurer", "customs_broker"]:
    w = compute_role_weighted_reward(sample_breakdown, role)
    print(f"  {role:<20}: weighted_reward={w:+.4f}")

print("\n=== DATASET BUILD TEST ===")
from training.train import build_curriculum_dataset
dataset = build_curriculum_dataset(warmup_samples=20)
from collections import Counter
role_counts = Counter(dataset["role"])
print(f"  Total prompts: {len(dataset)}")
for role, cnt in sorted(role_counts.items()):
    print(f"  {role:<25} {cnt} prompts")

print("\nAll checks passed.")

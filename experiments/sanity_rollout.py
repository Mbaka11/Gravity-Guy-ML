# experiments/sanity_rollout.py
"""
Run a few headless episodes with different policies and save per-episode metrics.
Output: experiments/runs/<timestamp>_<policy>.jsonl  (one JSON object per episode)
"""

from __future__ import annotations
import os, time, json, random
from typing import Optional
from src.env.gg_env import GGEnv

def heuristic_action(obs, info_prev: dict, state: dict) -> int:
    """
    Original heuristic - simple and effective.
    Flip when danger (lookahead clearance) is clearly high (HI).
    Keep requesting action=1 every frame until a flip actually occurs (sticky).
    Disarm once it's clearly safe again (LO).
    """
    _, _, _, p1, p2, p3 = obs
    # weight nearer probes more → earlier warning
    danger = max(0.8 * p1, 0.5 * p2, 0.3 * p3)

    HI, LO = 0.30, 0.15
    did_flip = bool(info_prev.get("did_flip", False))

    # If we were waiting for a flip, keep requesting until it actually fires.
    if state.get("pending_flip", False):
        if did_flip:                  # flip executed last step
            state["pending_flip"] = False
            return 0
        return 1                      # keep trying every frame

    # Arm when it looks dangerous; start requesting immediately
    if danger > HI:
        state["pending_flip"] = True
        return 1

    # Disarm when clearly safe again
    if danger < LO:
        state["pending_flip"] = False

    return 0

def improved_heuristic_action(obs, info_prev: dict, state: dict) -> int:
    """
    Improved heuristic - more conservative and smarter about timing.
    Key improvements:
    1. Slightly more aggressive thresholds
    2. Velocity awareness
    3. Better flip management
    """
    y_norm, vy_norm, grav_dir, p1, p2, p3 = obs
    
    did_flip = bool(info_prev.get("did_flip", False))
    
    # === ENHANCED DANGER CALCULATION ===
    # Base danger with slightly different weighting
    base_danger = max(0.85 * p1, 0.55 * p2, 0.25 * p3)
    
    # Velocity consideration: if moving fast toward obstacles, be more cautious
    velocity_factor = 1.0
    if abs(vy_norm) > 0.4:  # Moving fast in any direction
        velocity_factor = 1.2
    
    danger = min(1.0, base_danger * velocity_factor)
    
    # === SLIGHTLY MORE AGGRESSIVE THRESHOLDS ===
    HI, LO = 0.25, 0.12  # vs original 0.30, 0.15
    
    # === ANTI-THRASHING LOGIC ===
    flip_count = state.get("recent_flips", 0)
    
    # Decay flip count over time
    if "last_flip_time" not in state:
        state["last_flip_time"] = 0
    
    state["last_flip_time"] += 1
    if state["last_flip_time"] > 240:  # Reset every 2 seconds at 120fps
        state["recent_flips"] = max(0, flip_count - 1)
        state["last_flip_time"] = 0
    
    # === SAME STICKY LOGIC AS ORIGINAL ===
    pending = state.get("pending_flip", False)
    
    if pending:
        if did_flip:
            state["pending_flip"] = False
            state["recent_flips"] = state.get("recent_flips", 0) + 1
            return 0
        return 1
    
    # Only flip if we haven't been thrashing
    if danger > HI:
        if state.get("recent_flips", 0) < 5:  # Reasonable flip limit
            state["pending_flip"] = True
            return 1
    
    if danger < LO:
        state["pending_flip"] = False
    
    return 0

def conservative_heuristic_action(obs, info_prev: dict, state: dict) -> int:
    """
    Ultra-conservative heuristic - only flips when absolutely necessary.
    Should be very stable but might miss some opportunities.
    """
    _, _, _, p1, p2, p3 = obs
    
    did_flip = bool(info_prev.get("did_flip", False))
    
    # Only care about immediate danger (nearest probe)
    danger = p1
    
    # Very tight thresholds - only flip when obstacle is very close
    HI, LO = 0.20, 0.08
    
    pending = state.get("pending_flip", False)
    
    if pending:
        if did_flip:
            state["pending_flip"] = False
            return 0
        return 1
    
    if danger > HI:
        state["pending_flip"] = True
        return 1
    
    if danger < LO:
        state["pending_flip"] = False
    
    return 0

def aggressive_heuristic_action(obs, info_prev: dict, state: dict) -> int:
    """
    More aggressive heuristic - flips earlier to stay safer.
    Might flip more often but should avoid close calls.
    """
    _, _, _, p1, p2, p3 = obs
    
    did_flip = bool(info_prev.get("did_flip", False))
    
    # Weight all probes more heavily for earlier warning
    danger = max(0.9 * p1, 0.7 * p2, 0.4 * p3)
    
    # More aggressive thresholds
    HI, LO = 0.40, 0.20  # vs original 0.30, 0.15
    
    pending = state.get("pending_flip", False)
    
    if pending:
        if did_flip:
            state["pending_flip"] = False
            return 0
        return 1
    
    if danger > HI:
        state["pending_flip"] = True
        return 1
    
    if danger < LO:
        state["pending_flip"] = False
    
    return 0

def run_policy_test(
    n_episodes: int = 8,
    steps_per_ep: int = 1200,
    flip_prob: float = 0.12,
    level_seed: Optional[int] = None,
    rng_seed: Optional[int] = 0,
    policy: str = "random",
    max_time_s: float = 10.0,
):
    """
    Run episodes with a specific policy and save results.
    
    Args:
        n_episodes: Number of episodes to run
        steps_per_ep: Maximum steps per episode
        flip_prob: Probability of random flip (only used for random policy)
        level_seed: Seed for level generation (None = random each episode)
        rng_seed: Seed for random actions (None = don't seed)
        policy: Policy name ("random", "heuristic", "improved", "conservative", "aggressive")
        max_time_s: Maximum time per episode in seconds
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    os.makedirs("experiments/runs", exist_ok=True)
    out_path = f"experiments/runs/{int(time.time())}_{policy}.jsonl"

    # Policy function mapping
    policy_functions = {
        "heuristic": heuristic_action,
        "improved": improved_heuristic_action,
        "conservative": conservative_heuristic_action,
        "aggressive": aggressive_heuristic_action,
    }

    totals, dists, all_flips = [], [], []
    
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            # Create fresh environment for each episode
            env = GGEnv(level_seed=level_seed, max_time_s=max_time_s, flip_penalty=0.01, dt=1/120)
            obs = env.reset()

            total_r, flips = 0.0, 0
            
            # CRITICAL: Fresh state for each episode
            state = {"pending_flip": False}
            info = {}
            
            for t in range(steps_per_ep):
                # Get action based on policy
                if policy == "random":
                    a = 1 if random.random() < flip_prob else 0
                elif policy in policy_functions:
                    a = policy_functions[policy](obs, info, state)
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                obs, r, done, info = env.step(a)

                # Count actual flips, not attempts
                if info.get("did_flip", False):
                    flips += 1

                total_r += r
                if done:
                    break

            # Record episode results
            rec = {
                "policy": policy,
                "episode": ep,
                "steps": t + 1,
                "flips": flips,
                "total_return": round(float(total_r), 3),
                "time_s": round(float(info.get("time_s", 0.0)), 3),
                "distance_px": int(info.get("distance_px", 0)),
                "level_seed": info.get("level_seed"),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "time_up": bool(info.get("time_up", False)),
                # Final state instrumentation:
                "grounded": bool(info.get("grounded", False)),
                "cooldown": float(info.get("cooldown", 0.0)),
                "grav_dir": int(info.get("grav_dir", 0)),
                "probes": list(info.get("probes", [])),
            }
            f.write(json.dumps(rec) + "\n")

            totals.append(total_r)
            dists.append(info.get("distance_px", 0))
            all_flips.append(flips)

    # Print summary
    print(f"Saved {n_episodes} episodes to {out_path}")
    if totals:
        avg_r = sum(totals) / len(totals)
        avg_d = sum(dists) / len(dists)
        avg_f = sum(all_flips) / len(all_flips)
        max_d = max(dists)
        print(f"{policy.upper():>12}: avg_dist={avg_d:6.1f}px  avg_return={avg_r:7.2f}  avg_flips={avg_f:4.1f}  max_dist={max_d:4.0f}px")

def run_comparison():
    """Run all policies for comparison"""
    policies = [
        ("random", {"flip_prob": 0.12}),
        ("heuristic", {}),
        ("improved", {}), 
        ("conservative", {}),
        ("aggressive", {}),
    ]
    
    print("=== POLICY COMPARISON ===")
    print("Running 8 episodes per policy with 30s time limit...")
    print()
    
    for policy, kwargs in policies:
        run_policy_test(
            n_episodes=8,
            steps_per_ep=3600,  # 30 seconds at 120fps
            level_seed=None,    # Random levels
            rng_seed=0,         # Reproducible random actions
            policy=policy,
            max_time_s=30.0,    # Longer time limit to see true potential
            **kwargs
        )

if __name__ == "__main__":
    # Quick test with short time limit
    print("=== QUICK TEST (10s limit) ===")
    run_policy_test(policy="random", n_episodes=3, max_time_s=10.0)
    run_policy_test(policy="heuristic", n_episodes=3, max_time_s=10.0)
    run_policy_test(policy="improved", n_episodes=3, max_time_s=10.0)
    
    print("\n" + "="*50)
    
    # Full comparison with longer time limit
    run_comparison()
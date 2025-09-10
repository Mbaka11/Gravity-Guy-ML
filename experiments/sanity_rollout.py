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
    Heuristique simple et efficace, 100% côté rollout :
    - On s'arme (arm) quand on détecte un trou à l'avance via les probes (déjà fournies dans info_prev).
    - Une fois armé, on renvoie 1 (flip) à chaque frame jusqu'à ce que le flip se produise (did_flip).
    - Puis on relâche.
    """
    grounded = info_prev.get("grounded", False)
    # Les "probes" (p1, p2, p3) sont déjà calculées dans l'env et mises dans info_prev.
    # On les interprète simplement : plus petit => plus dangereux.
    p1, p2, p3 = info_prev.get("probes", (1.0, 1.0, 1.0))

    # Armement assez tôt : loin (p3), puis moyen (p2), puis près (p1)
    # Seuils prudents (à ajuster si besoin) :
    ARM_FAR  = 0.35
    ARM_MID  = 0.25
    ARM_NEAR = 0.18

    if grounded and (p3 < ARM_FAR or p2 < ARM_MID or p1 < ARM_NEAR):
        state["arm"] = True

    if state.get("arm", False):
        # Sticky jusqu’à ce que l’env confirme que le flip a été exécuté
        if info_prev.get("did_flip", False):
            state["arm"] = False
            return 0
        return 1

    return 0

def run_policy_test(
    n_episodes: int = 8,
    steps_per_ep: int = 1200,
    flip_prob: float = 0.12,
    level_seed: Optional[int] = None,
    rng_seed: Optional[int] = None,
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
    }

    totals, dists, all_flips = [], [], []
    
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            # Create fresh environment for each episode
            env = GGEnv(level_seed=level_seed, max_time_s=max_time_s, flip_penalty=0.01, dt=1/120)
            obs = env.reset()

            total_r, flips = 0.0, 0
            
            # CRITICAL: Fresh state for each episode
            state = {"pending_flip": False, "action_history": []}
            info = {}
            
            for t in range(steps_per_ep):
                # Get action based on policy
                if policy == "random":
                    a = 1 if random.random() < flip_prob else 0
                elif policy in policy_functions:
                    a = policy_functions[policy](obs, info, state)
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                # Record the action
                state["action_history"].append(a)

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
                # Full action history for replay
                "actions": state["action_history"],
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
    print("\n" + "="*50)
    # Full comparison with longer time limit
    run_comparison()
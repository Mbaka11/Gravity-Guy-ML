# experiments/sanity_rollout.py
"""
Run a few headless episodes with a random policy and save per-episode metrics.
Output: experiments/runs/<timestamp>_random.jsonl  (one JSON object per episode)
"""

from __future__ import annotations
import os, time, json, random
from typing import Optional
from src.env.gg_env import GGEnv

def heuristic_action(obs, info_prev: dict, state: dict) -> int:
    """
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

def run_random(
    n_episodes: int = 8,
    steps_per_ep: int = 2000,
    flip_prob: float = 0.10,
    level_seed=None,
    rng_seed: int | None = 0,
    policy: str = "random",
):
    if rng_seed is not None:
        random.seed(rng_seed)

    os.makedirs("experiments/runs", exist_ok=True)
    out_path = f"experiments/runs/{int(time.time())}_{policy}.jsonl"

    totals, dists = [], []
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            env = GGEnv(level_seed=level_seed, max_time_s=10.0, flip_penalty=0.01, dt=1/120)
            obs = env.reset()

            total_r, flips = 0.0, 0
            state, info = {"pending_flip": False}, {}
            for t in range(steps_per_ep):
                if policy == "heuristic":
                    a = heuristic_action(obs, info, state)
                else:
                    a = 1 if random.random() < flip_prob else 0

                obs, r, done, info = env.step(a)

                # count actual flips, not attempts
                if info.get("did_flip", False):
                    flips += 1

                total_r += r
                if done:
                    break

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
                # instrumentation at episode end:
                "grounded": bool(info.get("grounded", False)),
                "cooldown": float(info.get("cooldown", 0.0)),
                "grav_dir": int(info.get("grav_dir", 0)),
                "probes": list(info.get("probes", [])),
            }
            f.write(json.dumps(rec) + "\n")

            totals.append(total_r)
            dists.append(info.get("distance_px", 0))

    print(f"Saved {n_episodes} episodes to {out_path}")
    if totals:
        avg_r = sum(totals) / len(totals)
        avg_d = sum(dists) / max(1, len(dists))
        print(f"Avg return: {avg_r:.2f}   Avg distance: {avg_d:.1f}px   (flip_prob={flip_prob})")
        

if __name__ == "__main__":
    # Reproducible action sequence, random layout each episode:
    # Random
    run_random(n_episodes=8, steps_per_ep=1200, flip_prob=0.12, level_seed=None, rng_seed=0, policy="random")
    # Heuristic
    run_random(n_episodes=8, steps_per_ep=1200, flip_prob=0.12, level_seed=None, rng_seed=0, policy="heuristic")

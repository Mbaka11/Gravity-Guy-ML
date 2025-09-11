# /experiments/sanity_rollout.py
"""
Sanity rollouts for GGEnv v2:
- Runs RANDOM and/or TINY-HEURISTIC policies over fixed seeds
- Writes an episodes CSV for notebook analysis
- Saves per-episode action sequences (and optionally observations) for exact replay

Usage examples (from repo root):
  # Run both policies over 20 default seeds, frame_skip=4, save traces:
  python -m experiments.sanity_rollout --policies both --save-traces

  # Only heuristic, custom seeds, also save observations for deeper debugging:
  python -m experiments.sanity_rollout --policies heuristic --seeds 111,222,333 --save-traces --save-obs

  # Quick random-only smoke with fewer steps and no files:
  python -m experiments.sanity_rollout --policies random --steps 300 --out-dir /tmp/sanity --save-traces
"""

from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from src.env.gg_env_v2 import GGEnv  # assumes you placed the env here


# ------------------------ Policies ------------------------

def random_policy_init(action_seed: int):
    rng = np.random.RandomState(action_seed)
    def act(_obs: np.ndarray) -> int:
        return int(rng.randint(0, 2))  # 0 or 1
    return act

def tiny_heuristic_policy_init():
    """
    Very small rule:
      - If gravity is down: flip only when bottom near-probe shows danger
        (bottom spike at +120 OR no floor at +120), and target lane not spiky at +120.
      - If gravity is up: mirror for top.
    """
    def act(obs: np.ndarray) -> int:
        grav = obs[2]  # +1 down, -1 up
        # Near probe (+120) lives at indices 3..6
        ceil_n, floor_n, spike_top, spike_bot = obs[3], obs[4], obs[5], obs[6]
        if grav > 0:  # currently on bottom lane
            cur_danger = (spike_bot == 1.0) or (floor_n >= 0.999)  # no floor sentinel
            target_safe = (spike_top == 0.0) and (ceil_n > 0.0)    # ceiling exists & no spike
            return 1 if (cur_danger and target_safe) else 0
        else:         # currently on top lane
            cur_danger = (spike_top == 1.0) or (ceil_n <= 0.001)   # no ceiling sentinel
            target_safe = (spike_bot == 0.0) and (floor_n < 1.0)   # floor exists & no spike
            return 1 if (cur_danger and target_safe) else 0
    return act


# ------------------------ Rollout core ------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_episode_row(csv_path: Path, header: List[str], row: List):
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

def run_one_episode(policy_name: str,
                    seed: int,
                    frame_skip: int,
                    steps_limit: int,
                    save_traces: bool,
                    save_obs: bool,
                    out_dir: Path) -> Tuple[int, float, float, bool, bool, Optional[str], float]:
    """
    Returns: (ep_len, ret_sum, distance_px, terminated, truncated, death_cause, grounded_ratio)
    Also writes traces to disk if requested.
    """
    # For fair parity with training cadence, let env carry its own time limit (30s default).
    env = GGEnv(frame_skip=frame_skip)

    # Policy init
    if policy_name == "random":
        # Make action RNG seed a function of seed for determinism
        action_seed = 10_000 + seed
        policy = random_policy_init(action_seed)
    elif policy_name == "heuristic":
        action_seed = -1
        policy = tiny_heuristic_policy_init()
    else:
        raise ValueError("Unknown policy")

    # Storage for replay
    actions: List[int] = []
    obs_list: List[np.ndarray] = [] if save_obs else []

    ret_sum = 0.0
    grounded_count = 0
    ep_len = 0
    death_cause = None

    try:
        obs, info = env.reset(seed=seed)
        if save_obs:
            obs_list.append(obs.copy())

        for t in range(steps_limit):
            a = policy(obs)
            actions.append(int(a))

            obs, r, term, trunc, info = env.step(a)
            ret_sum += float(r)
            ep_len += 1
            grounded_count += int(bool(info.get("grounded", False)))
            death_cause = info.get("death_cause", None)

            if save_obs:
                obs_list.append(obs.copy())

            if term or trunc:
                break

        distance_px = float(info.get("distance_px", 0.0))
        terminated = bool(term)
        truncated = bool(trunc)
        grounded_ratio = grounded_count / max(1, ep_len)

    finally:
        env.close()

    # Save traces
    if save_traces:
        trace_dir = out_dir / "traces" / policy_name
        ensure_dir(trace_dir)
        np.save(trace_dir / f"{seed}_actions.npy", np.asarray(actions, dtype=np.int8))
        if save_obs:
            np.save(trace_dir / f"{seed}_obs.npy", np.asarray(obs_list, dtype=np.float32))

        # Also write a tiny metadata sidecar for convenience
        meta_lines = [
            f"seed={seed}",
            f"frame_skip={frame_skip}",
            f"policy={policy_name}",
            f"action_rng_seed={action_seed}",
            f"steps_limit={steps_limit}",
        ]
        (trace_dir / f"{seed}_meta.txt").write_text("\n".join(meta_lines), encoding="utf-8")

    return ep_len, ret_sum, distance_px, terminated, truncated, death_cause, grounded_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, default="both",
                    choices=["random", "heuristic", "both"],
                    help="Which policy to run")
    ap.add_argument("--seeds", type=str, default="",
                    help="Comma-separated seeds. If empty, uses 20 defaults: 101..120")
    ap.add_argument("--frame-skip", type=int, default=4,
                    help="Sim frames per decision step")
    ap.add_argument("--steps", type=int, default=10_000,
                    help="Hard cap on decision steps (env may truncate earlier)")
    ap.add_argument("--out-dir", type=str, default="experiments/runs",
                    help="Directory to store episodes.csv and traces/")
    ap.add_argument("--save-traces", action="store_true",
                    help="Save action sequences (and optional obs) for replay")
    ap.add_argument("--save-obs", action="store_true",
                    help="Also save observations per step (larger files)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Seeds
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(101, 121))  # 20 fixed eval seeds by default

    # Episode CSV
    episodes_csv = out_dir / "episodes.csv"
    header = [
        "env_name", "env_version", "obs_version",
        "policy_name", "seed",
        "frame_skip", "sim_fps", "decision_hz",
        "episode_len_decisions", "return_sum", "distance_px",
        "terminated", "truncated", "death_cause",
        "grounded_ratio"
    ]
    # Constants for metadata rows
    env_name = "GGEnv"
    env_version = "v2"
    obs_version = "v2"
    sim_fps = 60
    decision_hz = sim_fps / max(1, args.frame_skip)

    to_run = ["random", "heuristic"] if args.policies == "both" else [args.policies]

    print(f"Running policies={to_run} on {len(seeds)} seeds "
          f"(frame_skip={args.frame_skip}, decision_hz≈{decision_hz:.1f})")
    print(f"Writing summaries to {episodes_csv} and traces under {out_dir}/traces/")

    for policy_name in to_run:
        for seed in seeds:
            ep_len, ret_sum, dist, terminated, truncated, death_cause, g_ratio = run_one_episode(
                policy_name=policy_name,
                seed=seed,
                frame_skip=args.frame_skip,
                steps_limit=args.steps,
                save_traces=args.save_traces,
                save_obs=args.save_obs,
                out_dir=out_dir
            )

            row = [
                env_name, env_version, obs_version,
                policy_name, seed,
                args.frame_skip, sim_fps, decision_hz,
                ep_len, f"{ret_sum:.1f}", f"{dist:.1f}",
                int(terminated), int(truncated), (death_cause or ""),
                f"{g_ratio:.3f}",
            ]
            write_episode_row(episodes_csv, header, row)

            print(f"[{policy_name}] seed={seed}  len={ep_len}  dist={dist:.1f}  "
                  f"ret={ret_sum:.1f}  term={terminated} trunc={truncated}  cause={death_cause}")

    print("✓ Sanity rollouts complete")


if __name__ == "__main__":
    main()

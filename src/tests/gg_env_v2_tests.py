# src/tests/gg_env_v2_tests.py
"""
Quick tests for GGEnv v2 (Gymnasium environment).

Usage (from repo root):
  python -m src.tests.gg_env_v2_tests
  python -m src.tests.gg_env_v2_tests --render
  python -m src.tests.gg_env_v2_tests --no-api-check --no-determinism
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
except Exception as e:
    print("ERROR: gymnasium not installed. `pip install gymnasium`", file=sys.stderr)
    raise

try:
    from src.env.gg_env_v2 import GGEnv
except Exception as e:
    print(
        "ERROR: Couldn't import GGEnv. Make sure the file exists at src/env/gg_env_v2.py\n"
        "and run this script with `python -m src.tests.gg_env_v2_tests` from the project root.",
        file=sys.stderr,
    )
    raise


def api_check(frame_skip: int) -> None:
    """Verify Gym API contract (spaces, step/reset signatures, types)."""
    env = GGEnv(frame_skip=frame_skip)
    try:
        check_env(env, warn=True)
    finally:
        env.close()
    print("✓ API check ok")


def smoke_test(steps: int, seed: int, frame_skip: int) -> None:
    """Short random rollout: no crashes, obs in space, reward type, proper terminations."""
    env = GGEnv(frame_skip=frame_skip)
    try:
        obs, info = env.reset(seed=seed)
        assert env.observation_space.contains(obs), "Initial observation not in space"

        for t in range(steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            assert isinstance(r, float), "Reward must be a float"
            assert env.observation_space.contains(obs), f"Step {t}: observation out of bounds"
            if term or trunc:
                break
    finally:
        env.close()
    print("✓ Smoke test ok")


def determinism_test(steps: int, seed: int, frame_skip: int) -> None:
    """Same seed + same action sequence => identical obs/reward/terminal flags."""
    def rollout(seed_val: int, action_seq: List[int]) -> List[Tuple[np.ndarray, float, bool, bool]]:
        env = GGEnv(frame_skip=frame_skip)
        traj: List[Tuple[np.ndarray, float, bool, bool]] = []
        try:
            obs, _ = env.reset(seed=seed_val)
            for a in action_seq:
                obs, r, term, trunc, _ = env.step(int(a))
                traj.append((obs.copy(), float(r), bool(term), bool(trunc)))
                if term or trunc:
                    break
        finally:
            env.close()
        return traj

    # Fixed action sequence using a local RNG (not numpy global)
    rng = np.random.RandomState(42)
    action_seq = [int(rng.randint(0, 2)) for _ in range(steps)]

    t1 = rollout(seed, action_seq)
    t2 = rollout(seed, action_seq)

    assert len(t1) == len(t2), "Determinism: trajectory length mismatch"
    for i, ((o1, r1, te1, tr1), (o2, r2, te2, tr2)) in enumerate(zip(t1, t2)):
        if not np.allclose(o1, o2):
            raise AssertionError(f"Determinism: obs mismatch at step {i}")
        if not (r1 == r2 and te1 == te2 and tr1 == tr2):
            raise AssertionError(f"Determinism: transition mismatch at step {i}")

    print("✓ Determinism ok")


def render_demo(steps: int, seed: int, frame_skip: int) -> None:
    """Open a window and run a short NOOP demo so you can visually verify behavior."""
    env = GGEnv(render_mode="human", frame_skip=frame_skip)
    try:
        obs, info = env.reset(seed=seed)
        for _ in range(steps):
            # NOOP to just watch scrolling, collisions, and deaths
            obs, r, term, trunc, info = env.step(0)
            if term or trunc:
                break
    finally:
        env.close()
    print("✓ Render demo finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123, help="Episode seed for tests")
    ap.add_argument("--steps", type=int, default=300, help="Max decision steps per test")
    ap.add_argument("--frame-skip", type=int, default=4, help="Sim frames per decision step")
    ap.add_argument("--render", action="store_true", help="Run a short visual demo")
    ap.add_argument("--no-api-check", action="store_true", help="Skip Gym API compliance check")
    ap.add_argument("--no-smoke", action="store_true", help="Skip smoke test")
    ap.add_argument("--no-determinism", action="store_true", help="Skip determinism test")
    args = ap.parse_args()

    try:
        if not args.no_api_check:
            api_check(frame_skip=args.frame_skip)
        if not args.no_smoke:
            smoke_test(steps=args.steps, seed=args.seed, frame_skip=args.frame_skip)
        if not args.no_determinism:
            determinism_test(steps=args.steps, seed=args.seed, frame_skip=args.frame_skip)
        if args.render:
            # Keep the render short; you can bump steps if you want to watch longer.
            render_demo(steps=min(args.steps, 600), seed=args.seed, frame_skip=args.frame_skip)
    except AssertionError as e:
        print(f"✗ Test failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        raise
    else:
        print("🎉 All selected tests passed")


if __name__ == "__main__":
    main()

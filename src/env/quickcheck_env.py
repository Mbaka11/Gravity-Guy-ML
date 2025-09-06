# src/env/quickcheck_env.py
#command is python -m src.env.quickcheck_env
from __future__ import annotations
import random
from typing import List
from .gg_env import GGEnv

def run_once(seed=None, steps=600, flip_prob=0.1):
    env = GGEnv(level_seed=seed, max_time_s=10.0, flip_penalty=0.01, dt=1/120)
    obs = env.reset()               # first observation
    assert isinstance(obs, list) and len(obs) == 6, "Obs shape must be 6"
    total_r = 0.0
    flips = 0

    for t in range(steps):
        a = 1 if random.random() < flip_prob else 0
        if a == 1: flips += 1
        obs, r, done, info = env.step(a)

        # --- invariants / sanity ---
        assert len(obs) == 6, "Obs length changed"
        y, vy, g, p1, p2, p3 = obs
        assert 0.0 <= y <= 1.0, f"y out of range: {y}"
        assert -1.0 <= vy <= 1.0, f"vy out of range: {vy}"
        assert g in (-1.0, 1.0), f"grav_dir must be ±1, got {g}"
        for p in (p1, p2, p3):
            assert 0.0 <= p <= 1.0, f"probe out of range: {p}"
            
        if t % 200 == 0:
            y, vy, g, p1, p2, p3 = obs
            print(f"t={t} y={y:.2f} vy={vy:.2f} g={g:+.0f} p={p1:.2f},{p2:.2f},{p3:.2f}")

        total_r += r
        if done:
            print(f"[DONE] time={info['time_s']:.2f}s dist={int(info['distance_px'])} seed={info['level_seed']} outOfBounds={info['out_of_bounds']}")
            break

    print(f"steps={t+1} flips={flips} total_r={total_r:.3f}")


if __name__ == "__main__":
    # same seed should be reproducible (action sequence fixed by flip_prob RNG)
    run_once(seed=12345)
    # different layout each time with seed=None
    run_once(seed=None)
    
    run_once(seed=12345, steps=2000)

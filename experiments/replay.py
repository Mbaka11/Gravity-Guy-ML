# experiments/replay.py
"""
Replay tool for GGEnv v2 — quick command cheat sheet

# Typical usage (run from REPO ROOT so `src/...` imports work)

# Replay a HEURISTIC episode by seed (uses actions at experiments/runs/traces/heuristic/<seed>_actions.npy)
python -m experiments.replay --policy heuristic --seed 105

# Replay a RANDOM episode by seed
python -m experiments.replay --policy random --seed 112

# Replay by pointing directly to a specific actions file (bypasses --policy/--seed lookup)
python -m experiments.replay --trace experiments/runs/traces/heuristic/105_actions.npy --frame-skip 4

# Slow the display to ~decision rate (~15 fps) for readability
python -m experiments.replay --policy heuristic --seed 105 --slow

# Override frame skip for viewing at normal human speed
python -m experiments.replay --policy heuristic --seed 105 --frame-skip 1

# If running the script FROM INSIDE the experiments/ folder, set out-dir accordingly
python ./.replay.py --policy heuristic --seed 105 --out-dir ./runs
python ./.replay.py --trace ./runs/traces/random/112_actions.npy --frame-skip 4 --slow


# Arguments
--policy {random,heuristic,rl}   # which trace subfolder to use (ignored if --trace is provided)
--seed INT                       # episode seed to locate the trace (required unless --trace)
--trace PATH                     # explicit path to a 1D *_actions.npy file
--out-dir PATH                   # base folder containing runs/ (default: experiments/runs)
--frame-skip INT                 # sim frames per decision; default 4 (use 1 for normal-speed viewing)
--slow                           # limit display to ~15 fps for readability

# Controls during replay
SPACE = pause/resume
N     = single step (when paused)
R     = restart episode
ESC   = quit

# Notes / gotchas
- Deterministic: given the same seed, frame_skip, and action sequence, replay matches the original run.
- If you pass --trace, the script does not read meta; supply --frame-skip if different from 4.
- Expected trace layout from sanity rollouts: experiments/runs/traces/<policy>/<seed>_actions.npy
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pygame

from src.env.gg_env_v2 import GGEnv
from src.env.observations_v2 import build_observation_v2
from src.game.config import (
    WIDTH, HEIGHT, PLAYER_X, PLAYER_W, PLAYER_H,
    COLOR_BG, COLOR_PLAT, COLOR_ACCENT, COLOR_DANGER, COLOR_FG
)

DEFAULT_OUT_DIR = "experiments/runs"
DEFAULT_PROBES = (120, 240, 360)

def _find_trace(out_dir: Path, policy: str, seed: int) -> Path:
    p = out_dir / "traces" / policy / f"{seed}_actions.npy"
    if not p.exists():
        raise FileNotFoundError(f"Trace not found: {p}")
    return p

def _read_meta(out_dir: Path, policy: str, seed: int) -> dict:
    meta_path = out_dir / "traces" / policy / f"{seed}_meta.txt"
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta

def _draw_overlay(env: GGEnv, step_idx: int, action: Optional[int]):
    # Draw extra overlay on top of env's render using current pygame surface
    surf = pygame.display.get_surface()
    if surf is None:
        return
    font = pygame.font.SysFont("jetbrainsmono", 16)

    # Build obs for display (15-dim)
    try:
        # Use env internals to compute obs to avoid drift
        obs = env._get_obs()
        plat_rects = [p.rect for p in env.level.platforms] if env.level else []
    except Exception:
        obs = None
        plat_rects = []

    # Probe guide lines
    for dx in DEFAULT_PROBES:
        x = PLAYER_X + dx
        pygame.draw.line(surf, (90, 180, 255), (x, 0), (x, HEIGHT), 1)

    # Panel
    lines: List[str] = []
    lines.append(f"Step={step_idx}  Action={'NOOP' if action==0 else ('FLIP' if action==1 else '-')}")
    # Distance / death cause
    dist = getattr(env, "distance_px", 0.0)
    cause = getattr(env, "death_cause", None)
    lines.append(f"Dist={dist:.1f}px  Cause={cause or '—'}")


    # Grav and near-probe block if obs available
    if obs is not None and len(obs) == 15:
        grav = int(obs[2])
        lines.append(f"y={obs[0]:.2f}  vy={obs[1]:.2f}  g={grav:+d}")
        for i, dx in enumerate(DEFAULT_PROBES):
            b = 3 + 4*i
            ceil_n, floor_n, st, sb = obs[b:b+4]
            lines.append(f"+{dx:>3}: ceil={ceil_n:.2f} floor={floor_n:.2f}  T={int(st)} B={int(sb)}")

    # Translucent background
    panel_w = 360
    panel_h = 20 * (len(lines) + 1)
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((10, 20, 35, 160))
    surf.blit(panel, (12, 12))

    # Text
    y0 = 18
    for i, txt in enumerate(lines):
        surf.blit(font.render(txt, True, (210, 230, 255)), (20, y0 + i*20))

    pygame.display.flip()

def replay_episode(
    seed: int,
    actions: np.ndarray,
    frame_skip: int,
    slow: bool = False
):
    """
    Replays an episode deterministically using GGEnv v2 with on-screen overlay.
    Controls:
      SPACE: pause/resume   N: single-step when paused
      R: restart episode    ESC: quit
    """
    env = GGEnv(render_mode="human", frame_skip=frame_skip)
    obs, info = env.reset(seed=seed)
    env.render()
    
    paused = False
    step_idx = 0
    action: Optional[int] = None
    clock = pygame.time.Clock()

    try:
        running = True
        while running and step_idx < len(actions):
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        # single step when paused
                        if paused:
                            paused = False
                            # fall through to one step; we'll re-pause after
                            do_single = True
                        else:
                            do_single = False
                        # handle after stepping
                    elif event.key == pygame.K_r:
                        obs, info = env.reset(seed=seed)
                        step_idx = 0
                        paused = False

            if paused:
                # Draw current frame with overlay and idle
                env.render()
                _draw_overlay(env, step_idx, action=None)
                clock.tick(60)
                continue

            # Step with next recorded action
            action = int(actions[step_idx])
            obs, r, term, trunc, info = env.step(action)

            # Draw with overlay
            env.render()
            _draw_overlay(env, step_idx, action)

            step_idx += 1

            # Re-pause if we just did a single-step
            keys = pygame.key.get_pressed()
            # (We detect N via event, but simple approach: hold no extra state.)
            # Optional slow mode: cap to ~15 fps (decision rate) for readability
            if slow:
                clock.tick(15)
            else:
                clock.tick(60)

            if term or trunc:
                # Final frame is already drawn; wait a moment
                pygame.time.delay(600)
                break
    finally:
        env.close()

def main():
    ap = argparse.ArgumentParser(description="Replay a recorded GGEnv v2 episode with overlay.")
    ap.add_argument("--seed", type=int, help="Episode seed")
    ap.add_argument("--policy", type=str, default="random",
                    help="Trace subfolder name, e.g. random / heuristic / rl")
    ap.add_argument("--trace", type=str, default="",
                    help="Optional explicit path to a .npy action file")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                    help="Base directory where experiments/runs live")
    ap.add_argument("--frame-skip", type=int, default=-1,
                    help="Override frame_skip. If <0, use meta or default=4")
    ap.add_argument("--slow", action="store_true", help="Slow display (~15 fps) for readability")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # Resolve action file
    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        # Try to infer seed/policy from file name when possible
        if args.seed is None:
            try:
                args.seed = int(trace_path.stem.split("_")[0])
            except Exception:
                pass
    else:
        if args.seed is None:
            raise SystemExit("Please provide --seed or --trace")
        trace_path = _find_trace(out_dir, args.policy, args.seed)

    # Load actions
    actions = np.load(trace_path)
    if actions.ndim != 1:
        raise ValueError(f"Expected 1D action array, got shape {actions.shape}")

    # Determine frame_skip
    fs = args.frame_skip
    if fs < 0:
        fs = 4  # default
        if not args.trace:
            meta = _read_meta(out_dir, args.policy, args.seed)
            if "frame_skip" in meta:
                try:
                    fs = int(meta["frame_skip"])
                except Exception:
                    pass

    print(f"Replaying seed={args.seed}  policy={args.policy}  steps={len(actions)}  frame_skip={fs}")
    print("Controls: SPACE pause/resume | N step (when paused) | R restart | ESC quit")

    replay_episode(seed=args.seed, actions=actions, frame_skip=fs, slow=args.slow)

if __name__ == "__main__":
    main()

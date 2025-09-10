# src/env/obs_v2.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import pygame

from src.game.config import (
    WIDTH, HEIGHT, PLAYER_X, PLAYER_H, MAX_VY, SPIKE_BASE
)

# Probe positions ahead of the player (world space)
PROBE_OFFSETS_V2: Tuple[int, int, int] = (120, 240, 360)
# Horizontal window around a probe x within which a spike is considered "near"
SPIKE_WINDOW_PX: int = max(16, SPIKE_BASE // 2)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _norm_top_y(y_top: float) -> float:
    """Normalize a top coordinate into [0,1] using [0, HEIGHT-PLAYER_H]."""
    denom = max(1, HEIGHT - PLAYER_H)
    return _clamp01(y_top / denom)

def _norm_vy(vy: float, vy_max: float = MAX_VY) -> float:
    """Clip vy to [-vy_max, vy_max] and scale to [-1,1]."""
    vy_max = float(max(1.0, vy_max))
    vv = max(-vy_max, min(vy, vy_max))
    return np.float32(vv / vy_max)

def _surfaces_at_x(platform_rects: List[pygame.Rect], x: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (ceiling_y, floor_y) at vertical ray x.
    - ceiling_y: y of the underside (bottom) of top-lane platform covering x (min bottom).
    - floor_y  : y of the top of bottom-lane platform covering x (max top).
    None if absent.
    """
    ceil_y: Optional[int] = None
    floor_y: Optional[int] = None
    mid = HEIGHT * 0.5

    for r in platform_rects:
        # strict left<=x<right to avoid double-counting vertical edges
        if r.left <= x < r.right:
            if r.centery < mid:
                # top lane candidate -> use bottom as ceiling
                y = r.bottom
                ceil_y = y if (ceil_y is None or y < ceil_y) else ceil_y
            else:
                # bottom lane candidate -> use top as floor
                y = r.top
                floor_y = y if (floor_y is None or y > floor_y) else floor_y

    return ceil_y, floor_y

def _spike_presence_near_x(
    spikes: Optional[Iterable],
    x: int,
    window_px: int = SPIKE_WINDOW_PX
) -> Tuple[int, int]:
    """
    Detects spike presence near x (±window_px) by lane.
    Returns (has_top, has_bot) as 0/1.
    Accepts Spike objects with .platform/.local_x/.lane or dict/tuple equivalents.
    """
    if not spikes:
        return 0, 0

    has_top = 0
    has_bot = 0

    for sp in spikes:
        # Try to support both object and tuple/dict forms
        try:
            plat = getattr(sp, "platform", None)
            lane = getattr(sp, "lane", None)
            local_x = getattr(sp, "local_x", None)
            if plat is not None and local_x is not None:
                world_x = plat.rect.left + int(local_x)
                lane_id = str(lane)
            else:
                raise AttributeError
        except Exception:
            # Tuple form like (lane, world_x) or (lane, world_x, base)
            if isinstance(sp, dict):
                lane_id = str(sp.get("lane", "bot"))
                world_x = int(sp.get("x", x + 10_000))
            elif isinstance(sp, (tuple, list)) and len(sp) >= 2:
                lane_id = str(sp[0])
                world_x = int(sp[1])
            else:
                continue  # unknown spike format

        if abs(world_x - x) <= window_px:
            if lane_id == "top":
                has_top = 1
            elif lane_id == "bot":
                has_bot = 1

        # Early exit if both found
        if has_top and has_bot:
            break

    return has_top, has_bot

def build_observation_v2(
    player,
    platform_rects: List[pygame.Rect],
    spikes: Optional[Iterable] = None,
    probe_offsets: Tuple[int, int, int] = PROBE_OFFSETS_V2
) -> np.ndarray:
    """
    Returns a fixed (15,) float32 vector:
      [ y_top_norm, vy_norm, grav_dir,
        ceil@120, floor@120, spikeTop@120, spikeBot@120,
        ceil@240, floor@240, spikeTop@240, spikeBot@240,
        ceil@360, floor@360, spikeTop@360, spikeBot@360 ]
    - y_top_norm in [0,1]
    - vy_norm    in [-1,1]
    - grav_dir   in {-1,+1} (float)
    - ceil/floor normalized to [0,1] using screen coordinates;
      sentinel: ceil=0.0 if no ceiling; floor=1.0 if no floor.
    - spike flags are 0.0/1.0
    """
    # Top-based player.y (we assume you've standardized this)
    y_top_norm = np.float32(_norm_top_y(float(player.y)))
    vy_norm = np.float32(_norm_vy(float(player.vy)))
    grav = np.float32(+1.0 if getattr(player, "grav_dir", 1) > 0 else -1.0)

    # Build probes from the fixed player anchor
    base_x = int(PLAYER_X)

    feats: List[float] = [y_top_norm, vy_norm, grav]

    for dx in probe_offsets:
        px = base_x + int(dx)

        ceil_y, floor_y = _surfaces_at_x(platform_rects, px)

        # Normalize surfaces into [0,1] (screen space y)
        if ceil_y is None:
            ceil_norm = np.float32(0.0)   # "no ceiling" sentinel
        else:
            ceil_norm = np.float32(_clamp01(ceil_y / float(HEIGHT)))

        if floor_y is None:
            floor_norm = np.float32(1.0)  # "no floor" sentinel
        else:
            floor_norm = np.float32(_clamp01(floor_y / float(HEIGHT)))

        st, sb = _spike_presence_near_x(spikes, px)
        feats.extend([ceil_norm, floor_norm, float(st), float(sb)])

    return np.asarray(feats, dtype=np.float32)

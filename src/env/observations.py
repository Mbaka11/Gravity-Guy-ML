# src/env/observations.py
"""
Purpose
-------
Utilities to build a compact, Gym-like observation vector for the agent.
This file converts the current game state (player + platforms) into a small,
normalized feature vector that an ML policy can consume.

The current design (no stairs) returns a 6-D vector:
    [ y_norm, vy_norm, grav_dir, probe120, probe240, probe360 ]

- y_norm   : player vertical position normalized to [0, 1] (top -> 0, bottom -> 1)
- vy_norm  : vertical velocity normalized to [-1, 1]
- grav_dir : +1 if gravity pulls down, -1 if gravity pulls up
- probe*   : normalized vertical clearance (in the gravity direction) measured
             at x offsets +120, +240, +360 pixels ahead of the player
"""

from __future__ import annotations
from typing import List
import pygame
from ..game.config import (HEIGHT, WIDTH, PLAYER_H, PLAYER_W, MAX_VY)

PROBE_OFFSETS = (120, 240, 360) # Horizontal lookahead offsets (in pixels) where we "peek" ahead of the player.
MAX_CLEARANCE = HEIGHT # Maximum clearance used to clamp/normalize probe distances into [0, 1].
# If we don't find a platform at that x-column, we treat clearance as this cap.

def build_observation(player, platforms: List[pygame.Rect]) -> list : 
    """
    Build a compact observation vector for the agent.

    Summary
    -------
    Computes normalized player state (y, vy, gravity direction) plus three
    "lookahead" clearance probes in the gravity direction at fixed x-offsets.

    Args
    ----
    player : object
        Any object exposing:
          - x (float), y (float): top-left position in world coordinates
          - vy (float): vertical velocity
          - grav_dir (int): +1 (down) or -1 (up)
          - rect (pygame.Rect): current player rect (derived from x, y, width, height)
    platforms : List[pygame.Rect]
        Platform rectangles in world coordinates.

    Returns
    -------
    List[float]
        A 6-element observation vector:
          [ y_norm, vy_norm, grav_dir, probe120, probe240, probe360 ]
        where all values are in [-1, 1] or [0, 1] as documented above.
    """
    y_norm = max(0.0, min(1.0, player.y / max(1.0, HEIGHT - PLAYER_H)))
    vy_norm = max(-1.0, min(1.0, player.vy / MAX_VY))
    g = 1.0 if player.grav_dir > 0 else -1.0

    me = pygame.Rect(player.x, player.y, PLAYER_W, PLAYER_H)
    cx, top, bottom = me.centerx, me.top, me.bottom
    
    probes: List[float] = []
    
    # For each lookahead column, find the nearest platform surface in the gravity direction
    # and convert that distance to a normalized clearance in [0, 1].   
    for dx in PROBE_OFFSETS: 
        x = cx + dx 
        best = None # smallest non-negative distance along gravity direction
        for pr in platforms: # Only consider platforms that span this x-column.
            if pr.left <= x < pr.right:
                d = (pr.top - bottom) if g > 0 else (top - pr.bottom)
                if d >= 0 and (best is None or d < best):
                    best = d
        clear = MAX_CLEARANCE if best is None else best
        probes.append(max(0.0, min(1.0, clear / float(MAX_CLEARANCE))))

    return [y_norm, vy_norm, g] + probes
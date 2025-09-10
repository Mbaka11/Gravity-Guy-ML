# src/env/observations.py
"""_summary_

    Returns:
        _type_: _description_
"""
    
from __future__ import annotations
from typing import List
import pygame
from ..game.config import (HEIGHT, WIDTH, PLAYER_H, PLAYER_W, MAX_VY)

PROBE_OFFSETS = (120, 240, 360)  # Horizontal lookahead offsets (in pixels)
MAX_CLEARANCE = HEIGHT  # Maximum clearance for normalization
DEBUG_PROBES = False  # Set to True for detailed debugging

def build_observation(player, platforms: List[pygame.Rect]) -> list:
    """
    Build a compact observation vector for the agent.
    
    FIXED: Now properly calculates different clearances for each probe distance
    
    Returns: [y_norm, vy_norm, grav_dir, probe120, probe240, probe360]
    where probe values represent CLEARANCE (0=collision, 1=max safe distance)
    """
    # Normalize player position and velocity
    y_norm = max(0.0, min(1.0, player.y / max(1.0, HEIGHT - PLAYER_H)))
    vy_norm = max(-1.0, min(1.0, player.vy / MAX_VY))
    g = 1.0 if player.grav_dir > 0 else -1.0

    # Player bounds
    me = pygame.Rect(player.x, player.y, PLAYER_W, PLAYER_H)
    cx, top, bottom = me.centerx, me.top, me.bottom
    
    probes: List[float] = []
    
    if DEBUG_PROBES:
        print(f"\n=== PROBE DEBUG ===")
        print(f"Player at ({cx}, {player.y}), gravity={g:+.0f}")
        print(f"Player bounds: top={top}, bottom={bottom}")
    
    # For each lookahead distance, find nearest obstacle
    for i, dx in enumerate(PROBE_OFFSETS):
        probe_x = cx + dx
        min_clearance = None
        
        if DEBUG_PROBES:
            print(f"\nProbe {i+1} at x={probe_x}:")
        
        # Check all platforms for potential collisions at this x-coordinate
        for pi, pr in enumerate(platforms):
            # Check if this platform intersects our probe line
            # Use <= and >= to be more inclusive
            if pr.left <= probe_x <= pr.right:
                if g > 0:  # Gravity pulls down - look for platforms below
                    if pr.top >= bottom:  # Platform is below player bottom
                        clearance = pr.top - bottom
                        if DEBUG_PROBES:
                            print(f"  Platform {pi}: top={pr.top}, clearance={clearance:.1f}")
                        if clearance >= 0 and (min_clearance is None or clearance < min_clearance):
                            min_clearance = clearance
                else:  # Gravity pulls up - look for platforms above
                    if pr.bottom <= top:  # Platform is above player top
                        clearance = top - pr.bottom
                        if DEBUG_PROBES:
                            print(f"  Platform {pi}: bottom={pr.bottom}, clearance={clearance:.1f}")
                        if clearance >= 0 and (min_clearance is None or clearance < min_clearance):
                            min_clearance = clearance
        
        # Convert to normalized clearance (0=collision, 1=max safe)
        if min_clearance is None:
            normalized_clearance = 1.0  # No obstacle = maximum clearance
            if DEBUG_PROBES:
                print(f"  No obstacle found -> clearance = 1.0")
        else:
            normalized_clearance = max(0.0, min(1.0, min_clearance / MAX_CLEARANCE))
            if DEBUG_PROBES:
                print(f"  Min clearance = {min_clearance:.1f} -> normalized = {normalized_clearance:.3f}")
        
        probes.append(normalized_clearance)
    
    observation = [y_norm, vy_norm, g] + probes
    
    if DEBUG_PROBES:
        print(f"Final observation: {observation}")
    
    return observation


def debug_probe_calculation(player, platforms: List[pygame.Rect]):
    """
    Standalone function to debug probe calculation issues
    """
    global DEBUG_PROBES
    DEBUG_PROBES = True
    
    print("=== DETAILED PROBE DEBUG ===")
    print(f"Total platforms: {len(platforms)}")
    for i, p in enumerate(platforms):
        print(f"Platform {i}: left={p.left}, right={p.right}, top={p.top}, bottom={p.bottom}")
    
    obs = build_observation(player, platforms)
    
    DEBUG_PROBES = False
    return obs
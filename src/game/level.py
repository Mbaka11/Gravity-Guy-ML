# src/game/level.py
from __future__ import annotations
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame
from .config import (
    WIDTH, HEIGHT, PLATFORM_THICKNESS, LANE_TOP_Y, LANE_BOT_Y,
    SEGMENT_MIN_W, SEGMENT_MAX_W, GAP_MIN_W, GAP_MAX_W, SCROLL_PX_PER_S,
    MOVING_PLATFORM_SPEED, MOVING_PLATFORM_RANGE, MOVING_PLATFORM_CHANCE,
    MIN_STATIC_BETWEEN_MOVING
)

@dataclass
class Platform:
    rect: pygame.Rect
    lane: str  # "top" or "bot"
    platform_type: str = "static"  # "static" or "moving"
    move_range: float = 0.0  # How far it moves up/down
    move_speed: float = 0.0  # Speed of movement
    move_time: float = 0.0   # Current time in movement cycle
    original_y: float = 0.0  # Original Y position to oscillate around

class LevelGen:
    """
    Generates an endless ribbon of platforms scrolling left.
    Motifs: flat segments, gaps, and simple 'stairs'.
    Two lanes (top / bottom) to match gravity flip gameplay.
    """
    def __init__(self, seed: int | None):
        # If seed is None → randomize once (per level)
        if seed is None:
            # 32-bit seed is enough and printable
            seed = random.randrange(0, 2**32 - 1)
        self.seed = seed
        self.rng = random.Random(seed)
        self.platforms: List[Platform] = []
        self.static_platforms_since_moving = 0  # Counter for platform spacing
        self._init_start()

    def _init_start(self):
        # Fill the initial screen with some flat segments on both lanes
        # Force static platforms at the start for safety
        x = 0
        while x < WIDTH * 1.5:
            self.platforms.append(self._make_seg(x, "top", WIDTH // 3, force_static=True))
            self.platforms.append(self._make_seg(x, "bot", WIDTH // 3, force_static=True))
            x += WIDTH // 3

    def _make_seg(self, x: int, lane: str, w: int, force_static: bool = False, safe_path: bool = False) -> Platform:
        # Create a platform rectangle anchored on lane line
        y_line = LANE_TOP_Y if lane == "top" else LANE_BOT_Y
        if lane == "top":
            rect = pygame.Rect(x, y_line - PLATFORM_THICKNESS, w, PLATFORM_THICKNESS)
        else:
            rect = pygame.Rect(x, y_line, w, PLATFORM_THICKNESS)
        
        # Decide if this platform should move (unless forced static)
        platform_type = "static"
        move_range = 0.0
        move_speed = 0.0
        
        if not force_static and self.static_platforms_since_moving >= MIN_STATIC_BETWEEN_MOVING:
            # Check if we can safely make this a moving platform
            can_move = False
            
            # Only allow moving platforms if there's a guaranteed escape route
            if safe_path:
                # For top lane platforms, need static platform below
                if lane == "top":
                    # Look for static platforms below in the next segment
                    for p in self.platforms:
                        if (p.lane == "bot" and p.platform_type == "static" and
                            abs(p.rect.x - x) < SEGMENT_MAX_W):
                            can_move = True
                            break
                # For bottom lane platforms, need static platform above
                else:
                    # Look for static platforms above in the next segment
                    for p in self.platforms:
                        if (p.lane == "top" and p.platform_type == "static" and
                            abs(p.rect.x - x) < SEGMENT_MAX_W):
                            can_move = True
                            break
            
            if can_move and self.rng.random() < MOVING_PLATFORM_CHANCE:
                platform_type = "moving"
                # Reduce movement range when there might not be a safe path
                move_range = MOVING_PLATFORM_RANGE * (0.7 if not safe_path else 1.0)
                move_speed = MOVING_PLATFORM_SPEED
                self.static_platforms_since_moving = 0
                # Randomize starting phase
                move_time = self.rng.random() * 6.28  # random phase in radians
            else:
                self.static_platforms_since_moving += 1
        else:
            self.static_platforms_since_moving += 1
        
        return Platform(
            rect=rect, 
            lane=lane,
            platform_type=platform_type,
            move_range=move_range,
            move_speed=move_speed,
            move_time=0.0,
            original_y=float(rect.y)
        )

    def _make_gap(self, x: int, lane: str, w: int) -> Tuple[int, Optional[Platform]]:
        # A gap means 'reserve empty space'
        return (x + w, None)
    
    def _opposite_lane(self, lane: str) -> str:
        return "bot" if lane == "top" else "top"

    def _next_motif(self) -> str:
        # Only flat segments or gaps
        return "flat" if self.rng.random() < 0.6 else "gap"

    def _rand_w(self, lo: int, hi: int) -> int:
        return int(self.rng.uniform(lo, hi))

    def update_and_generate(self, dt: float):
        """
        Scroll existing platforms and extend the level on the right so
        that the visible region + some margin is always filled.
        Update moving platform positions.
        """
        dx = int(SCROLL_PX_PER_S * dt)
        for p in self.platforms:
            # Horizontal scroll
            p.rect.x -= dx
            
            # Update moving platforms
            if p.platform_type == "moving":
                # Update movement time
                p.move_time += dt * p.move_speed
                
                # Calculate new y position using sine wave
                offset = p.move_range * math.sin(p.move_time)
                p.rect.y = int(p.original_y + offset)

        # Drop off-screen platforms
        self.platforms = [p for p in self.platforms if p.rect.right > -200]

        # Extend to ~1.5x screen width
        right_edge = max((p.rect.right for p in self.platforms), default=0)
        target = max(right_edge, 0)
        
        while target < WIDTH * 1.5:
            # Generate platforms in pairs to ensure safe paths
            motif = self._next_motif()
            if motif == "flat":
                w = self._rand_w(SEGMENT_MIN_W, SEGMENT_MAX_W)
                
                # Always create both top and bottom platforms for safety
                # Bottom platform first so top platform can check for safe paths
                bottom_plat = self._make_seg(target, "bot", w, force_static=False, safe_path=True)
                self.platforms.append(bottom_plat)
                
                # Top platform can move if there's a safe bottom platform
                top_plat = self._make_seg(target, "top", w, force_static=False, safe_path=bottom_plat.platform_type == "static")
                self.platforms.append(top_plat)
                
                target += w
            else:  # gap
                w = self._rand_w(GAP_MIN_W, GAP_MAX_W)
                # For gaps, ensure the next segment will have safe platforms
                target += w

    def draw(self, surf: pygame.Surface, color: Tuple[int, int, int]):
        for p in self.platforms:
            pygame.draw.rect(surf, color, p.rect)

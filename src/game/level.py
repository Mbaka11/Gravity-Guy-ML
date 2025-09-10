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
    
    def update_movement(self, dt: float):
        """Update moving platform position"""
        if self.platform_type == "moving":
            self.move_time += dt * self.move_speed
            offset = self.move_range * math.sin(self.move_time)
            self.rect.y = int(self.original_y + offset)

class LevelGen:
    """
    Generates an endless ribbon of platforms scrolling left.
    Improved moving platform generation with better safety guarantees.
    """
    def __init__(self, seed: int | None):
        if seed is None:
            seed = random.randrange(0, 2**32 - 1)
        self.seed = seed
        self.rng = random.Random(seed)
        self.platforms: List[Platform] = []
        self.consecutive_moving = 0  # Track consecutive moving platforms
        self.last_safe_x = 0  # X position of last guaranteed safe platform pair
        self._init_start()

    def _init_start(self):
        """Initialize the starting area with safe static platforms"""
        x = 0
        while x < WIDTH * 1.5:
            # Always create safe static platforms at the start
            w = WIDTH // 3
            self.platforms.append(self._create_platform(x, "top", w, force_static=True))
            self.platforms.append(self._create_platform(x, "bot", w, force_static=True))
            x += w
        self.last_safe_x = x

    def _create_platform(self, x: int, lane: str, w: int, force_static: bool = False) -> Platform:
        """Create a platform with proper positioning and movement parameters"""
        y_line = LANE_TOP_Y if lane == "top" else LANE_BOT_Y
        
        if lane == "top":
            rect = pygame.Rect(x, y_line - PLATFORM_THICKNESS, w, PLATFORM_THICKNESS)
        else:
            rect = pygame.Rect(x, y_line, w, PLATFORM_THICKNESS)
        
        # Decide on movement
        platform_type = "static"
        move_range = 0.0
        move_speed = 0.0
        move_time = 0.0
        
        if not force_static and self._should_make_moving_platform():
            platform_type = "moving"
            # Use full movement range but ensure it doesn't go outside reasonable bounds
            # Calculate safe movement range based on lane position
            if lane == "top":
                # Top platforms can move down more than up
                max_down = (LANE_BOT_Y - LANE_TOP_Y) * 0.6  # Don't go too close to bottom lane
                max_up = LANE_TOP_Y * 0.8  # Don't go too close to top of screen
                safe_range = min(MOVING_PLATFORM_RANGE, max_down, max_up)
            else:
                # Bottom platforms can move up more than down
                max_up = (LANE_BOT_Y - LANE_TOP_Y) * 0.6  # Don't go too close to top lane  
                max_down = (HEIGHT - LANE_BOT_Y) * 0.8  # Don't go too close to bottom of screen
                safe_range = min(MOVING_PLATFORM_RANGE, max_up, max_down)
            
            move_range = safe_range
            move_speed = MOVING_PLATFORM_SPEED
            # Random starting phase to vary movement patterns
            move_time = self.rng.random() * 6.28
            self.consecutive_moving += 1
        else:
            self.consecutive_moving = 0
        
        return Platform(
            rect=rect,
            lane=lane,
            platform_type=platform_type,
            move_range=move_range,
            move_speed=move_speed,
            move_time=move_time,
            original_y=float(rect.y)
        )

    def _should_make_moving_platform(self) -> bool:
        """Determine if the next platform should be moving based on safety rules"""
        # Don't allow too many consecutive moving platforms
        if self.consecutive_moving >= 2:
            return False
        
        # Probability check
        if self.rng.random() > MOVING_PLATFORM_CHANCE:
            return False
        
        return True

    def _generate_segment_pair(self, x: int) -> Tuple[List[Platform], int]:
        """Generate a pair of platforms (top and bottom) ensuring at least one is accessible"""
        w = self._rand_w(SEGMENT_MIN_W, SEGMENT_MAX_W)
        platforms = []
        
        # Strategy: ensure at least one platform is static for safety
        force_one_static = self.consecutive_moving > 0 or self.rng.random() < 0.7
        
        if force_one_static:
            # Randomly choose which lane gets the static platform
            static_lane = self.rng.choice(["top", "bot"])
            
            for lane in ["top", "bot"]:
                is_static = (lane == static_lane)
                platform = self._create_platform(x, lane, w, force_static=is_static)
                platforms.append(platform)
        else:
            # Both can potentially be moving (rare case)
            for lane in ["top", "bot"]:
                platform = self._create_platform(x, lane, w, force_static=False)
                platforms.append(platform)
        
        return platforms, w

    def _generate_gap(self, x: int) -> Tuple[List[Platform], int]:
        """Generate a gap section - no platforms, just advance X"""
        w = self._rand_w(GAP_MIN_W, GAP_MAX_W)
        return [], w

    def _rand_w(self, lo: int, hi: int) -> int:
        return int(self.rng.uniform(lo, hi))

    def update_and_generate(self, dt: float):
        """
        Update existing platforms and generate new ones as needed
        """
        # Scroll all platforms left
        dx = int(SCROLL_PX_PER_S * dt)
        for platform in self.platforms:
            platform.rect.x -= dx
            platform.update_movement(dt)

        # Remove off-screen platforms
        self.platforms = [p for p in self.platforms if p.rect.right > -200]

        # Generate new platforms as needed
        right_edge = max((p.rect.right for p in self.platforms), default=0)
        target_x = max(right_edge, 0)
        
        while target_x < WIDTH * 1.5:
            # Decide between segment or gap
            if self.rng.random() < 0.65:  # 65% chance for platform segment
                new_platforms, width = self._generate_segment_pair(target_x)
                self.platforms.extend(new_platforms)
            else:  # 35% chance for gap
                new_platforms, width = self._generate_gap(target_x)
                # Reset moving platform counter after gaps for variety
                self.consecutive_moving = 0
            
            target_x += width

    def draw(self, surf: pygame.Surface, color: Tuple[int, int, int]):
        """Draw all platforms"""
        for platform in self.platforms:
            # Optional: Draw moving platforms in a different shade
            if platform.platform_type == "moving":
                # Slightly brighter color for moving platforms
                r, g, b = color
                moving_color = (min(255, r + 20), min(255, g + 20), min(255, b + 20))
                pygame.draw.rect(surf, moving_color, platform.rect)
            else:
                pygame.draw.rect(surf, color, platform.rect)
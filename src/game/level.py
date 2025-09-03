# src/game/level.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame
from .config import (
    WIDTH, HEIGHT, PLATFORM_THICKNESS, LANE_TOP_Y, LANE_BOT_Y,
    SEGMENT_MIN_W, SEGMENT_MAX_W, GAP_MIN_W, GAP_MAX_W, STAIR_STEP,
    SCROLL_PX_PER_S
)

@dataclass
class Platform:
    rect: pygame.Rect
    lane: str  # "top" or "bot"

class LevelGen:
    """
    Generates an endless ribbon of platforms scrolling left.
    Motifs: flat segments, gaps, and simple 'stairs'.
    Two lanes (top / bottom) to match gravity flip gameplay.
    """
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.platforms: List[Platform] = []
        self._init_start()

    def _init_start(self):
        # Fill the initial screen with some flat segments on both lanes
        x = 0
        while x < WIDTH * 1.5:
            self.platforms.append(self._make_seg(x, "top", WIDTH // 3))
            self.platforms.append(self._make_seg(x, "bot", WIDTH // 3))
            x += WIDTH // 3

    def _make_seg(self, x: int, lane: str, w: int) -> Platform:
        # Create a platform rectangle anchored on lane line
        y_line = LANE_TOP_Y if lane == "top" else LANE_BOT_Y
        if lane == "top":
            rect = pygame.Rect(x, y_line - PLATFORM_THICKNESS, w, PLATFORM_THICKNESS)
        else:
            rect = pygame.Rect(x, y_line, w, PLATFORM_THICKNESS)
        return Platform(rect=rect, lane=lane)

    def _make_gap(self, x: int, lane: str, w: int) -> Tuple[int, Optional[Platform]]:
        # A gap means 'reserve empty space'
        return (x + w, None)

    def _stairs(self, x: int, lane: str, total_w: int, steps: int = 3) -> List[Platform]:
        """
        Build a simple staircase (visual variety).
        We keep the lane the same; only the y shifts a bit.
        """
        seg_w = max(SEGMENT_MIN_W, total_w // steps)
        out: List[Platform] = []
        cur_x = x
        for i in range(steps):
            if lane == "top":
                y = LANE_TOP_Y - PLATFORM_THICKNESS - i * (STAIR_STEP // 2)
            else:
                y = LANE_BOT_Y + i * (STAIR_STEP // 2)
            rect = pygame.Rect(cur_x, y, seg_w, PLATFORM_THICKNESS)
            out.append(Platform(rect=rect, lane=lane))
            cur_x += seg_w
        return out

    def _next_motif(self) -> str:
        # Light bias towards flat and gap
        r = self.rng.random()
        if r < 0.5:
            return "flat"
        elif r < 0.8:
            return "gap"
        else:
            return "stairs"

    def _rand_w(self, lo: int, hi: int) -> int:
        return int(self.rng.uniform(lo, hi))

    def update_and_generate(self, dt: float):
        """
        Scroll existing platforms and extend the level on the right so
        that the visible region + some margin is always filled.
        """
        dx = int(SCROLL_PX_PER_S * dt)
        for p in self.platforms:
            p.rect.x -= dx

        # Drop off-screen platforms
        self.platforms = [p for p in self.platforms if p.rect.right > -200]

        # Extend to ~1.5x screen width
        right_edge = max((p.rect.right for p in self.platforms), default=0)
        target = max(right_edge, 0)
        while target < WIDTH * 1.5:
            lane = "top" if self.rng.random() < 0.5 else "bot"
            motif = self._next_motif()
            if motif == "flat":
                w = self._rand_w(SEGMENT_MIN_W, SEGMENT_MAX_W)
                self.platforms.append(self._make_seg(target, lane, w))
                target += w
            elif motif == "gap":
                w = self._rand_w(GAP_MIN_W, GAP_MAX_W)
                target += w  # empty space
            else:  # stairs
                total_w = self._rand_w(SEGMENT_MIN_W * 2, SEGMENT_MAX_W * 2)
                self.platforms.extend(self._stairs(target, lane, total_w, steps=3))
                target += total_w

    def draw(self, surf: pygame.Surface, color: Tuple[int, int, int]):
        for p in self.platforms:
            pygame.draw.rect(surf, color, p.rect)

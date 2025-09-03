# src/game/player.py
from __future__ import annotations
import pygame
from dataclasses import dataclass
from typing import List, Tuple
from .config import (
    PLAYER_X, PLAYER_W, PLAYER_H, G_ABS, MAX_VY, JUMP_COOLDOWN_S
)

@dataclass
class Player:
    """
    Minimal player with gravity flip:
    - grav_dir = +1 means gravity pulls down
    - grav_dir = -1 means gravity pulls up
    """
    x: float
    y: float
    vy: float
    grav_dir: int   # +1 down, -1 up
    grounded: bool
    _flip_cooldown: float = 0.0

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), PLAYER_W, PLAYER_H)

    def try_flip(self):
        """Attempt to flip gravity; obey cooldown to avoid rapid toggling."""
        if self._flip_cooldown <= 0.0:
            self.grav_dir *= -1
            self._flip_cooldown = JUMP_COOLDOWN_S

    def update_physics(self, dt: float):
        """Integrate vertical motion under signed gravity, clamp velocity."""
        if self._flip_cooldown > 0:
            self._flip_cooldown -= dt

        # Apply gravity along current direction
        self.vy += self.grav_dir * G_ABS * dt

        # Clamp vertical speed for stability
        if self.vy > MAX_VY: self.vy = MAX_VY
        if self.vy < -MAX_VY: self.vy = -MAX_VY

        # Integrate position
        self.y += self.vy * dt

    def resolve_collisions_swept(self, prev_y: float, platforms: List[pygame.Rect]) -> Tuple[bool, bool]:
        """
        Swept vertical collision:
        Uses previous and current rects to detect surface crossings even if
        the player would have tunneled between frames.
        Returns (grounded, crushed).
        """
        me_now = self.rect
        me_prev = pygame.Rect(int(self.x), int(prev_y), PLAYER_W, PLAYER_H)

        grounded = False
        crushed = False

        # Candidates for first hit (distance traveled to the contact)
        hit_distance = None
        hit_surface_y = None
        hitting_from_above = None  # True if coming down onto a top; False if going up into a bottom

        moving_down = self.vy >= 0

        for pr in platforms:
            # Horizontal overlap check is necessary for a vertical surface hit
            horiz_overlap = (me_now.right > pr.left) and (me_now.left < pr.right)
            if not horiz_overlap:
                continue

            if moving_down:
                # Did we cross a platform TOP? (prev bottom <= top && now bottom >= top)
                if me_prev.bottom <= pr.top and me_now.bottom >= pr.top:
                    dist = pr.top - me_prev.bottom
                    if dist >= 0 and (hit_distance is None or dist < hit_distance):
                        hit_distance = dist
                        hit_surface_y = pr.top
                        hitting_from_above = True
            else:
                # Moving up: Did we cross a platform BOTTOM? (prev top >= bottom && now top <= bottom)
                if me_prev.top >= pr.bottom and me_now.top <= pr.bottom:
                    dist = me_prev.top - pr.bottom
                    if dist >= 0 and (hit_distance is None or dist < hit_distance):
                        hit_distance = dist
                        hit_surface_y = pr.bottom
                        hitting_from_above = False

        if hit_surface_y is not None:
            # Snap to contact and zero vertical velocity
            if moving_down and hitting_from_above:
                # Rest on platform top
                self.y = hit_surface_y - PLAYER_H
                self.vy = 0.0
                grounded = True
            elif (not moving_down) and (hitting_from_above is False):
                # Rest against platform bottom
                self.y = hit_surface_y
                self.vy = 0.0
                grounded = True

        # After snapping, if any overlap remains, flag crushed (rare edge case)
        me_fix = self.rect
        for pr in platforms:
            if me_fix.colliderect(pr):
                crushed = True
                break

        self.grounded = grounded
        return grounded, crushed

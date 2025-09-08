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
    
    def _inner_rect(self, rect: pygame.Rect) -> pygame.Rect:
        r = rect.copy()
        # shrink the hitbox so that touching side walls doesn’t register as a blocking overlap
        return r

    def try_flip(self):
        """Flip gravity only if grounded and cooldown elapsed."""
        if self._flip_cooldown <= 0.0 and self.grounded:
            self.grav_dir *= -1
            self._flip_cooldown = JUMP_COOLDOWN_S
            self.vy = 0.0
            self.grounded = False

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
        Vertical swept test (top/bottom only).
        Returns (grounded, crushed=False).
        """
        me_prev = pygame.Rect(int(self.x), int(prev_y), PLAYER_W, PLAYER_H)
        me_now = self.rect

        grounded = False

        # Check vertical crossings (top/bottom) only
        moving_down = self.vy >= 0

        hit_surface_y = None
        hitting_from_above = None  # True if landing on a top; False if hitting a bottom

        for pr in platforms:
            horiz_overlap = (me_now.right > pr.left) and (me_now.left < pr.right)
            if not horiz_overlap:
                continue

            if moving_down:
                # crossing a platform TOP
                if me_prev.bottom <= pr.top and me_now.bottom >= pr.top:
                    dist = pr.top - me_prev.bottom
                    if dist >= 0 and (hit_surface_y is None or dist < (hit_surface_y - me_prev.bottom)):
                        hit_surface_y = pr.top
                        hitting_from_above = True
            else:
                # crossing a platform BOTTOM
                if me_prev.top >= pr.bottom and me_now.top <= pr.bottom:
                    dist = me_prev.top - pr.bottom
                    if dist >= 0 and (hit_surface_y is None or dist < (me_prev.top - hit_surface_y)):
                        hit_surface_y = pr.bottom
                        hitting_from_above = False

        # Snap to contact and zero vertical velocity
        if hit_surface_y is not None:
            if moving_down and hitting_from_above:
                # rest on platform top
                self.y = hit_surface_y - PLAYER_H
                self.vy = 0.0
                grounded = True
            elif (not moving_down) and (hitting_from_above is False):
                # rest against platform bottom
                self.y = hit_surface_y
                self.vy = 0.0
                grounded = True

        self.grounded = grounded
        return grounded, False  
    


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
        # No hitbox shrinkage needed now; just return the rect as-is
        return rect

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
        Vertical swept test (top/bottom only). Side overlaps are ignored.
        Returns (grounded, crushed=False).
        """
        # raw rects at previous and current positions
        me_prev_raw = pygame.Rect(int(self.x), int(prev_y), PLAYER_W, PLAYER_H)
        me_now_raw  = self.rect

        # inner rects used for collision checks (soften side contacts)
        me_prev = self._inner_rect(me_prev_raw)
        me_now  = self._inner_rect(me_now_raw)

        grounded = False

        # Check vertical crossings (top/bottom) only
        moving_down = self.vy >= 0

        hit_surface_y = None
        hitting_from_above = None  # True if landing on a top; False if hitting a bottom

        for pr in platforms:
            pr_inner = self._inner_rect(pr)  # also soften platforms slightly
            horiz_overlap = (me_now.right > pr_inner.left) and (me_now.left < pr_inner.right)
            if not horiz_overlap:
                continue

            if moving_down:
                # crossing a platform TOP
                if me_prev.bottom <= pr_inner.top and me_now.bottom >= pr_inner.top:
                    dist = pr_inner.top - me_prev.bottom
                    if dist >= 0 and (hit_surface_y is None or dist < (hit_surface_y - me_prev.bottom)):
                        hit_surface_y = pr_inner.top
                        hitting_from_above = True
            else:
                # crossing a platform BOTTOM
                if me_prev.top >= pr_inner.bottom and me_now.top <= pr_inner.bottom:
                    dist = me_prev.top - pr_inner.bottom
                    if dist >= 0 and (hit_surface_y is None or dist < (me_prev.top - hit_surface_y)):
                        hit_surface_y = pr_inner.bottom
                        hitting_from_above = False

        # Snap to contact and zero vertical velocity
        if hit_surface_y is not None:
            if moving_down and hitting_from_above:
                # rest on platform top: set player bottom == surface top
                self.y = hit_surface_y - PLAYER_H
                self.vy = 0.0
                grounded = True
            elif (not moving_down) and (hitting_from_above is False):
                # rest against platform bottom: set player top == surface bottom
                self.y = hit_surface_y
                self.vy = 0.0
                grounded = True

        self.grounded = grounded
        return grounded, False  
    
    def _min_axis_penetration(self, a: pygame.Rect, b: pygame.Rect):
        """Return (pen_x, pen_y, side), where side∈{'left','right','top','bottom'} is the direction to push the player out."""
        dx_left   = b.right - a.left      # if positive, overlap from left
        dx_right  = a.right - b.left      # if positive, overlap from right
        dy_top    = b.bottom - a.top      # overlap from top
        dy_bottom = a.bottom - b.top      # overlap from bottom

        pen_x = min(dx_left, dx_right)
        pen_y = min(dy_top, dy_bottom)

        if pen_x < pen_y:
            # horizontal resolution
            side = 'left' if dx_left < dx_right else 'right'
        else:
            side = 'top' if dy_top < dy_bottom else 'bottom'
        return pen_x, pen_y, side
    
    def resolve_side_collisions(self, platforms: List[pygame.Rect]) -> bool:
        """
        Make platform SIDES solid without killing:
        - Prefer horizontal resolution (no corner slip).
        - Clamp when very close to the side even if overlap is ~0 (prevents slow bleed-through).
        - Skip pushing when airborne so flips don't feel delayed.
        Returns True if we pushed horizontally.
        """
        # If we're clearly airborne and moving away, do nothing (flips feel snappy)
        if not self.grounded and abs(self.vy) > 10:
            return False

        pushed = False
        me = self.rect

        # How close counts as "touching" even if not overlapping
        PROX_EPS = 2  # px
        # Horizontal bias: if widths/heights are similar, treat as side contact
        H_BIAS = 4    # px

        for pr in platforms:
            pr_in = pr

            # Compute intersection; may be empty
            inter = me.clip(pr_in)

            # If there is *some* vertical overlap region, we may need a horizontal clamp
            vertical_overlap = inter.height > 0

            if inter.width > 0 and inter.height > 0:
                # Real overlap: choose axis with a bias toward horizontal
                if inter.width + H_BIAS <= inter.height:
                    # Horizontal resolution
                    if me.centerx < pr_in.centerx:
                        # player is left of wall → push left
                        self.x = pr_in.left - PLAYER_W - 1
                    else:
                        # player is right of wall → push right
                        self.x = pr_in.right + 1
                    pushed = True
                    me = self.rect
                else:
                    # Mostly vertical overlap — vertical swept solver already handled it.
                    pass

        # Optional: very light recenter to PLAYER_X when not touching anything,
        # so pushes don't drift you away forever (comment out if you prefer raw drift):
        # if not pushed:
        #     anchor = PLAYER_X
        #     snap_speed = 800.0  # px/s; call this with dt if you want smooth
        #     dx = anchor - self.x
        #     if abs(dx) < 1.0:
        #         self.x = anchor

        return pushed

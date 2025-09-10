# src/game/player.py
from __future__ import annotations
import pygame
from dataclasses import dataclass
from typing import List, Tuple, Optional
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

    # Platform tracking for moving platforms
    _standing_on_platform: Optional[object] = None
    _platform_contact_buffer: float = 0.0  # Small buffer time to maintain contact

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), PLAYER_W, PLAYER_H)

    def can_flip(self) -> bool:
        return (self._flip_cooldown <= 0.0) and (self.grounded or self._platform_contact_buffer > 0.0)

    def try_flip(self) -> bool:
        """Flip gravity only if grounded and cooldown elapsed. Returns True if performed."""
        if self.can_flip():
            self.grav_dir *= -1
            self._flip_cooldown = JUMP_COOLDOWN_S
            self.vy = 0.0
            self.grounded = False
            # Clear platform tracking when flipping
            self._standing_on_platform = None
            self._platform_contact_buffer = 0.0
            return True
        return False

    def update_physics(self, dt: float):
        """Integrate vertical motion under signed gravity, clamp velocity."""
        if self._flip_cooldown > 0:
            self._flip_cooldown -= dt
            if self._flip_cooldown < 0.0:
                self._flip_cooldown = 0.0

        # Apply gravity
        self.vy += self.grav_dir * G_ABS * dt

        # Clamp vertical speed
        self.vy = max(-MAX_VY, min(MAX_VY, self.vy))

        # Integrate position
        self.y += self.vy * dt

        # Update platform contact buffer
        if self._platform_contact_buffer > 0.0:
            self._platform_contact_buffer = max(0.0, self._platform_contact_buffer - dt)

    def resolve_collisions_with_platforms(self, platforms: List[object]) -> Tuple[bool, bool]:
        """
        Convention: self.y = TOP du joueur.
        Résout collisions avec plateformes (statiques/mobiles) en gardant un léger "stick"
        sur plateformes mobiles et en transportant le joueur via last_dy.
        """
        HORIZONTAL_TOLERANCE = 2   # px, évite l'accrochage latéral
        VERTICAL_STICK_TOL   = 3   # px, tolérance de collage
        CONTACT_BUFFER_S     = 0.08

        # Rect du joueur (TOP-based)
        player_rect = pygame.Rect(
            PLAYER_X - PLAYER_W // 2,
            int(self.y),
            PLAYER_W,
            PLAYER_H
        )
        check_rect = player_rect.copy()
        check_rect.inflate_ip(-HORIZONTAL_TOLERANCE * 2, 0)

        new_grounded = False
        collision_occurred = False
        contacted_platform = None

        # 0) Si on était déjà posé sur une plateforme mobile, tenter de maintenir le contact
        if self._standing_on_platform is not None and self._platform_contact_buffer > 0.0:
            plat = self._standing_on_platform
            plat_rect = plat.rect

            # overlap horizontal ?
            if not (check_rect.right <= plat_rect.left or check_rect.left >= plat_rect.right):
                if self.grav_dir > 0:
                    # posé SUR (on veut player_rect.bottom ~ plat.top)
                    if abs(player_rect.bottom - plat_rect.top) <= VERTICAL_STICK_TOL:
                        # transporter avec la plateforme
                        self.y += getattr(plat, "last_dy", 0)
                        player_rect.top = int(self.y)
                        new_grounded = True
                        collision_occurred = True
                else:
                    # collé DESSOUS (on veut player_rect.top ~ plat.bottom)
                    if abs(player_rect.top - plat_rect.bottom) <= VERTICAL_STICK_TOL:
                        self.y += getattr(plat, "last_dy", 0)
                        player_rect.top = int(self.y)
                        new_grounded = True
                        collision_occurred = True

            if new_grounded and getattr(plat, "platform_type", "static") == "moving":
                # rafraîchir un peu le buffer si on a re-capturé le contact
                self._platform_contact_buffer = max(self._platform_contact_buffer, CONTACT_BUFFER_S / 2.0)

        # 1) Balayage collisions "fraîches"
        if not new_grounded:
            for plat in platforms:
                plat_rect = plat.rect

                # filtre horizontal
                if (check_rect.right <= plat_rect.left) or (check_rect.left >= plat_rect.right):
                    continue

                if self.grav_dir > 0:
                    # Atterrir SUR la plateforme (du dessus)
                    is_crossing_top = (
                        player_rect.bottom >= plat_rect.top and
                        player_rect.top    <  plat_rect.top and
                        self.vy >= 0.0
                    )
                    if is_crossing_top:
                        # Snap: top = plat.top - hauteur
                        self.y  = plat_rect.top - PLAYER_H
                        self.vy = 0.0
                        new_grounded = True
                        collision_occurred = True
                        contacted_platform = plat

                        if getattr(plat, "platform_type", "static") == "moving":
                            self.y += getattr(plat, "last_dy", 0)
                            self._standing_on_platform = plat
                            self._platform_contact_buffer = CONTACT_BUFFER_S
                        break

                else:
                    # Coller SOUS la plateforme (plafond)
                    is_crossing_bottom = (
                        player_rect.top    <= plat_rect.bottom and
                        player_rect.bottom >  plat_rect.bottom and
                        self.vy <= 0.0
                    )
                    if is_crossing_bottom:
                        # Snap: top = plat.bottom (car top-based)
                        self.y  = plat_rect.bottom
                        self.vy = 0.0
                        new_grounded = True
                        collision_occurred = True
                        contacted_platform = plat

                        if getattr(plat, "platform_type", "static") == "moving":
                            self.y += getattr(plat, "last_dy", 0)
                            self._standing_on_platform = plat
                            self._platform_contact_buffer = CONTACT_BUFFER_S
                        break

        # 2) Gestion des états/flags
        if new_grounded:
            if contacted_platform is not None:
                self._standing_on_platform = contacted_platform
                if getattr(contacted_platform, "platform_type", "static") == "moving":
                    self._platform_contact_buffer = CONTACT_BUFFER_S
                else:
                    self._platform_contact_buffer = 0.0
        else:
            # On conserve la plateforme en mémoire tant que le buffer > 0 (permet flip juste après)
            if self._platform_contact_buffer <= 0.0:
                self._standing_on_platform = None

        self.grounded = new_grounded

        # (Optionnel) si tu maintiens un self.rect ailleurs, tu peux le synchroniser ici :
        # self.rect.topleft = (PLAYER_X - PLAYER_W // 2, int(self.y))

        return new_grounded, collision_occurred


    def _resolve_horizontal_collisions(self, platforms: List[object]) -> bool:
        """
        Handle horizontal (side) collisions with platforms.
        When player hits the side of a platform, push them back and stop horizontal movement.
        
        Returns:
            True if a horizontal collision occurred
        """
        player_rect = self.rect
        collision_occurred = False
        
        for platform in platforms:
            platform_rect = platform.rect
            
            # Check if player is overlapping with platform
            if not player_rect.colliderect(platform_rect):
                continue
            
            # Calculate overlap amounts
            overlap_left = player_rect.right - platform_rect.left
            overlap_right = platform_rect.right - player_rect.left
            overlap_top = player_rect.bottom - platform_rect.top
            overlap_bottom = platform_rect.bottom - player_rect.top
            
            # Check if this is primarily a horizontal collision
            # (horizontal overlap is smaller than vertical overlap)
            min_horizontal_overlap = min(overlap_left, overlap_right)
            min_vertical_overlap = min(overlap_top, overlap_bottom)
            
            # Only treat as horizontal collision if horizontal overlap is smaller
            # and there's significant vertical overlap (player is beside the platform)
            if (min_horizontal_overlap < min_vertical_overlap and 
                min_vertical_overlap > PLAYER_H * 0.3):  # At least 30% vertical overlap
                
                # Determine which side we hit and push back
                if overlap_left < overlap_right:
                    # Hit the left side of platform - push player left
                    self.x = platform_rect.left - PLAYER_W - 1
                else:
                    # Hit the right side of platform - push player right  
                    self.x = platform_rect.right + 1
                
                # Stop any horizontal momentum (if the game had horizontal movement)
                # Since this is a runner game, we mainly just position the player
                collision_occurred = True
                
                # Optional: Add a small vertical "bounce" effect for better feel
                if abs(self.vy) < 50:  # Only if not moving fast vertically
                    bounce_strength = 30.0
                    if self.grav_dir > 0:
                        self.vy = -bounce_strength  # Bounce up slightly
                    else:
                        self.vy = bounce_strength   # Bounce down slightly
                
                break  # Handle one collision at a time
        
        return collision_occurred

    def _try_maintain_moving_platform_contact(self, platform: object) -> bool:
        """
        Try to maintain contact with a moving platform even if slightly separated.
        This handles the case where the platform moves faster than the physics update.
        """
        player_rect = self.rect
        platform_rect = platform.rect
        
        # Check if we're still close enough horizontally
        HORIZONTAL_TOLERANCE = 8  # pixels
        if (player_rect.right < platform_rect.left - HORIZONTAL_TOLERANCE or 
            player_rect.left > platform_rect.right + HORIZONTAL_TOLERANCE):
            return False
        
        # Check vertical proximity and adjust position
        VERTICAL_TOLERANCE = 12  # pixels
        
        if self.grav_dir > 0:  # Standing on top
            distance_to_top = abs(player_rect.bottom - platform_rect.top)
            if distance_to_top <= VERTICAL_TOLERANCE:
                # Snap back to platform top
                self.y = platform_rect.top - PLAYER_H
                self.vy = 0.0
                return True
        else:  # Hanging from bottom
            distance_to_bottom = abs(player_rect.top - platform_rect.bottom)
            if distance_to_bottom <= VERTICAL_TOLERANCE:
                # Snap back to platform bottom
                self.y = platform_rect.bottom
                self.vy = 0.0
                return True
        
        return False

    def resolve_collisions_swept(self, prev_y: float, platforms: List[object]) -> Tuple[bool, bool]:
        """
        Legacy method name kept for compatibility with existing code.
        Calls the new collision resolution system.
        """
        return self.resolve_collisions_with_platforms(platforms)
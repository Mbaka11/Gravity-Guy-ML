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
        return (self._flip_cooldown <= 0.0) and self.grounded

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
            self._platform_contact_buffer -= dt

    def resolve_collisions_with_platforms(self, platforms: List[object]) -> Tuple[bool, bool]:
        """
        Resolve collisions with platforms (both static and moving).
        
        Args:
            platforms: List of Platform objects with .rect, .platform_type, .rect.y attributes
            
        Returns:
            (grounded, collision_occurred)
        """
        player_rect = self.rect
        collision_occurred = False
        new_grounded = False
        contacted_platform = None
        
        # Horizontal tolerance for platform edges (prevents getting stuck on edges)
        HORIZONTAL_TOLERANCE = 2
        
        # Create a slightly smaller player rect for horizontal overlap check
        # This prevents side-snagging while scrolling
        check_rect = player_rect.copy()
        check_rect.inflate_ip(-HORIZONTAL_TOLERANCE * 2, 0)
        
        for platform in platforms:
            platform_rect = platform.rect
            
            # Only check platforms that horizontally overlap with player
            if (check_rect.right <= platform_rect.left or 
                check_rect.left >= platform_rect.right):
                continue
            
            # Check for vertical collision based on gravity direction
            if self.grav_dir > 0:  # Gravity pulls down - check floor collision
                if (player_rect.bottom >= platform_rect.top and 
                    player_rect.top < platform_rect.top and
                    self.vy >= 0):  # Only if moving down or stationary
                    
                    # Land on top of platform
                    self.y = platform_rect.top - PLAYER_H
                    self.vy = 0.0
                    new_grounded = True
                    collision_occurred = True
                    contacted_platform = platform
                    
                    # If it's a moving platform, match its vertical movement
                    if hasattr(platform, 'platform_type') and platform.platform_type == "moving":
                        self._standing_on_platform = platform
                        self._platform_contact_buffer = 0.1  # 100ms buffer
                    
                    break
                    
            else:  # Gravity pulls up - check ceiling collision
                if (player_rect.top <= platform_rect.bottom and 
                    player_rect.bottom > platform_rect.bottom and
                    self.vy <= 0):  # Only if moving up or stationary
                    
                    # Stick to bottom of platform
                    self.y = platform_rect.bottom
                    self.vy = 0.0
                    new_grounded = True
                    collision_occurred = True
                    contacted_platform = platform
                    
                    # If it's a moving platform, match its vertical movement
                    if hasattr(platform, 'platform_type') and platform.platform_type == "moving":
                        self._standing_on_platform = platform
                        self._platform_contact_buffer = 0.1  # 100ms buffer
                    
                    break
        
        # Handle moving platform tracking
        if contacted_platform != self._standing_on_platform:
            self._standing_on_platform = contacted_platform
        
        # If we didn't contact any platform, check if we should maintain contact with moving platform
        if not collision_occurred and self._standing_on_platform and self._platform_contact_buffer > 0:
            moving_platform = self._standing_on_platform
            if hasattr(moving_platform, 'platform_type') and moving_platform.platform_type == "moving":
                # Try to maintain contact with the moving platform
                if self._try_maintain_moving_platform_contact(moving_platform):
                    new_grounded = True
                    collision_occurred = True
                else:
                    # Lost contact with moving platform
                    self._standing_on_platform = None
                    self._platform_contact_buffer = 0.0
        
        # Clear platform tracking if not grounded
        if not new_grounded:
            self._standing_on_platform = None
            self._platform_contact_buffer = 0.0
        
        self.grounded = new_grounded
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
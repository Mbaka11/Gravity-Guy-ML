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
    MIN_STATIC_BETWEEN_MOVING,
    SPIKE_CHANCE, SPIKE_MIN_PER_PLATFORM, SPIKE_MAX_PER_PLATFORM,
    SPIKE_HEIGHT, SPIKE_BASE, SPIKE_MARGIN_X, SPIKE_MIN_SPACING, COLOR_SPIKE, SPIKE_MIN_SPACING, SPIKE_MARGIN_X
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
            
@dataclass
class Spike:
    """Un pic triangulaire attaché à une plateforme."""
    platform: Platform        # référence directe à la plateforme
    lane: str                 # "top" ou "bot"
    local_x: int              # offset local depuis left de la plateforme
    height: int
    base: int                 # largeur de la base

    def world_points(self) -> Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
        """Retourne (A,B,C) en coords monde pour le triangle."""
        rect = self.platform.rect
        cx = rect.left + self.local_x
        half = self.base // 2

        if self.lane == "bot":
            y_base = rect.top
            A = (cx - half, y_base)
            B = (cx + half, y_base)
            C = (cx, y_base - self.height)   # pointe vers le haut (intérieur)
        else:  # "top"
            y_base = rect.bottom
            A = (cx - half, y_base)
            B = (cx + half, y_base)
            C = (cx, y_base + self.height)   # pointe vers le bas (intérieur)
        return A, B, C

    def aabb(self) -> pygame.Rect:
        A, B, C = self.world_points()
        xs = (A[0], B[0], C[0])
        ys = (A[1], B[1], C[1])
        return pygame.Rect(min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))


def _point_in_triangle(p, a, b, c) -> bool:
    """Test barycentrique: p dans triangle abc."""
    (px, py) = p
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    v0x, v0y = cx-ax, cy-ay
    v1x, v1y = bx-ax, by-ay
    v2x, v2y = px-ax, py-ay
    dot00 = v0x*v0x + v0y*v0y
    dot01 = v0x*v1x + v0y*v1y
    dot02 = v0x*v2x + v0y*v2y
    dot11 = v1x*v1x + v1y*v1y
    dot12 = v1x*v2x + v1y*v2y
    inv_denom = 1.0 / max(1e-8, (dot00 * dot11 - dot01 * dot01))
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)

def rect_intersects_triangle(r: pygame.Rect, tri_pts) -> bool:
    """Rect ↔ triangle: coins dans tri OU sommet tri dans rect OU arêtes qui se croisent."""
    A, B, C = tri_pts
    # 1) coins du rect à l'intérieur du triangle
    corners = [(r.left, r.top), (r.right, r.top), (r.right, r.bottom), (r.left, r.bottom)]
    if any(_point_in_triangle(p, A, B, C) for p in corners):
        return True
    # 2) sommet du triangle dans le rect
    if r.collidepoint(A) or r.collidepoint(B) or r.collidepoint(C):
        return True
    # 3) intersection d'arêtes (triangle edges vs rect edges)
    def segs_intersect(p1, p2, p3, p4):
        def cross(u, v): return u[0]*v[1]-u[1]*v[0]
        def sub(u, v): return (u[0]-v[0], u[1]-v[1])
        r = sub(p2, p1); s = sub(p4, p3)
        denom = cross(r, s)
        if abs(denom) < 1e-8: return False
        t = cross(sub(p3, p1), s) / denom
        upar = cross(sub(p3, p1), r) / denom
        return 0 <= t <= 1 and 0 <= upar <= 1
    tri_edges = [(A, B), (B, C), (C, A)]
    rect_edges = [
        ((r.left, r.top), (r.right, r.top)),
        ((r.right, r.top), (r.right, r.bottom)),
        ((r.right, r.bottom), (r.left, r.bottom)),
        ((r.left, r.bottom), (r.left, r.top)),
    ]
    for e1 in tri_edges:
        for e2 in rect_edges:
            if segs_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False

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
        self.spikes: List[Spike] = []
        self.consecutive_moving = 0  # Track consecutive moving platforms
        self.last_safe_x = 0  # X position of last guaranteed safe platform pair
        self._init_start()

    def _init_start(self):
        x = 0
        while x < WIDTH * 1.5:
            w = WIDTH // 3
            top = self._create_platform(x, "top", w, force_static=True)
            bot = self._create_platform(x, "bot", w, force_static=True)
            self.platforms.append(top)
            self._on_platform_created(top, safe=True)
            self.platforms.append(bot)
            self._on_platform_created(bot, safe=True)
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
        
    def _maybe_add_spikes_for_platform(self, plat: Platform):
        """25% de chance d'ajouter 1–3 pics à cette plateforme (hors départ),
        en garantissant un espacement minimal entre pics et une marge aux bords."""
        # Éviter la zone de départ (plateformes encore visibles au spawn)
        if plat.rect.right <= WIDTH * 1.0:
            return
        if self.rng.random() > SPIKE_CHANCE:
            return

        half_base = SPIKE_BASE // 2

        # On impose que la BASE du triangle reste à l'intérieur + marge
        usable_left  = plat.rect.left  + max(SPIKE_MARGIN_X, half_base)
        usable_right = plat.rect.right - max(SPIKE_MARGIN_X, half_base)
        usable_w = max(0, usable_right - usable_left)
        if usable_w < SPIKE_BASE:
            return

        # Espacement minimal entre pics mesuré entre leurs BORDS :
        # |c1 - c2| - BASE >= SPIKE_MIN_SPACING  ->  |c1 - c2| >= BASE + SPIKE_MIN_SPACING
        min_center_dist = SPIKE_BASE + SPIKE_MIN_SPACING

        # Upper bound du nb de pics possibles avec cette largeur et cet espacement
        max_possible = 1 + ((usable_w - SPIKE_BASE) // min_center_dist) if usable_w >= SPIKE_BASE else 0
        if max_possible <= 0:
            return

        desired = self.rng.randint(SPIKE_MIN_PER_PLATFORM, SPIKE_MAX_PER_PLATFORM)
        count = int(min(desired, max_possible))

        positions = []
        attempts = 0
        while len(positions) < count and attempts < 50:
            attempts += 1
            # on choisit un CENTRE de base dans la zone utilisable
            cx_world = self.rng.randint(usable_left, usable_right)
            if all(abs(cx_world - p) >= min_center_dist for p in positions):
                positions.append(cx_world)

        for cx_world in positions:
            self.spikes.append(Spike(
                platform=plat,
                lane=plat.lane,
                local_x=int(cx_world - plat.rect.left),
                height=SPIKE_HEIGHT,
                base=SPIKE_BASE
            ))


    def _on_platform_created(self, plat: Platform, safe: bool = False):
        """À appeler chaque fois qu'une plateforme est ajoutée à self.platforms."""
        if not safe:
            self._maybe_add_spikes_for_platform(plat)


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

        # Clean spikes attached to removed platforms (use identity, not hashing)
        valid_ids = {id(p) for p in self.platforms}
        self.spikes = [
            s for s in self.spikes
            if (id(s.platform) in valid_ids and s.platform.rect.right > -200)
]

        # Generate new platforms as needed
        right_edge = max((p.rect.right for p in self.platforms), default=0)
        target_x = max(right_edge, 0)
        while target_x < WIDTH * 1.5:
            # Decide between segment or gap
            if self.rng.random() < 0.65:  # 65% chance for platform segment
                new_platforms, width = self._generate_segment_pair(target_x)
                self.platforms.extend(new_platforms)
                for i in range(len(self.platforms) - len(new_platforms), len(self.platforms)):
                    self._on_platform_created(self.platforms[i], safe=False)
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
                
        for sp in self.spikes:
            A, B, C = sp.world_points()
            pygame.draw.polygon(surf, COLOR_SPIKE, (A, B, C))
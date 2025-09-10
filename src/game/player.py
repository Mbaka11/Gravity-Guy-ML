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

    # --- anti-décrochage plateformes mobiles ---
    _stick_time: float = 0.0          # durée restante de "stick" (s)
    _support_y: float | None = None   # y de la surface support (top pour sol, bottom pour plafond)
    _support_face: int = 0            # +1 = sol ; -1 = plafond ; 0 = aucun
    _last_dt: float = 0.0             # dt de la dernière update_physics

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), PLAYER_W, PLAYER_H)
    
    def _inner_rect(self, rect: pygame.Rect) -> pygame.Rect:
        r = rect.copy()
        # Skin latéral pour éviter les accrochages avec les plateformes qui défilent
        r.inflate_ip(-4, 0)  # 2 px de chaque côté (assez pour ignorer les side-snags)
        return r

    def can_flip(self) -> bool:
        return (self._flip_cooldown <= 0.0) and self.grounded

    def try_flip(self) -> bool:
        """Flip gravity only if grounded and cooldown elapsed. Returns True if performed."""
        if self.can_flip():
            self.grav_dir *= -1
            self._flip_cooldown = JUMP_COOLDOWN_S
            self.vy = 0.0
            self.grounded = False
            # en changeant de face, on arrête le stick courant
            self._stick_time = 0.0
            self._support_y = None
            self._support_face = 0
            return True
        return False

    def update_physics(self, dt: float):
        """Integrate vertical motion under signed gravity, clamp velocity."""
        self._last_dt = float(dt)

        if self._flip_cooldown > 0:
            self._flip_cooldown -= dt
            if self._flip_cooldown < 0.0:
                self._flip_cooldown = 0.0

        # Appliquer la gravité
        self.vy += self.grav_dir * G_ABS * dt

        # Clamp vertical speed
        if self.vy > MAX_VY: self.vy = MAX_VY
        if self.vy < -MAX_VY: self.vy = -MAX_VY

        # Intégrer la position
        self.y += self.vy * dt

        # Décroissance du "stick" (on le réduit après usage dans resolve, mais on assure une décroissance)
        if self._stick_time > 0.0:
            self._stick_time -= dt
            if self._stick_time < 0.0:
                self._stick_time = 0.0

    def _try_stick_to_support(self, me_now: pygame.Rect, platforms: List[pygame.Rect]) -> bool:
        """
        Si on a un support récent (_stick_time > 0), essayer de "reposer" dessus même si
        le chevauchement horizontal est limite à cause du scroll.
        """
        if self._stick_time <= 0.0 or self._support_face == 0 or self._support_y is None:
            return False

        X_STICK = 8  # tolérance horizontale (px) pour suivre la plateforme qui bouge
        EPS = 1

        # Calcul du y cible en fonction de la face support
        if self._support_face == +1:       # sol: on pose le bas du joueur sur top de la plateforme
            y_target = self._support_y - PLAYER_H
        else:                               # plafond: on pose le haut du joueur sur bottom de la plateforme
            y_target = self._support_y

        # Chercher une plateforme à la bonne altitude (top/bottom) encore proche en X
        for pr in platforms:
            pr_in = self._inner_rect(pr)
            if self._support_face == +1 and pr_in.top == int(self._support_y):
                if me_now.right >= pr_in.left - X_STICK and me_now.left <= pr_in.right + X_STICK:
                    # snap doux sur la surface
                    self.y = y_target
                    self.vy = 0.0
                    self.grounded = (self.grav_dir > 0)
                    return True
            elif self._support_face == -1 and pr_in.bottom == int(self._support_y):
                if me_now.right >= pr_in.left - X_STICK and me_now.left <= pr_in.right + X_STICK:
                    self.y = y_target
                    self.vy = 0.0
                    self.grounded = (self.grav_dir < 0)
                    return True
        return False

    def resolve_collisions_swept(self, prev_y: float, platforms: List[pygame.Rect]) -> Tuple[bool, bool]:
        """
        Résolution **verticale uniquement** (runner à X fixe) avec:
          - skin latéral (ignore les contacts côté),
          - snapping directionnel (sol si on descend, plafond si on monte),
          - "stick" temporaire pour suivre une plateforme qui bouge.
        """
        EPS = 2  # petite tolérance d'arrondi
        me_before = pygame.Rect(int(self.x), int(prev_y), PLAYER_W, PLAYER_H)
        me_now = self.rect
        grounded = False

        moving_down = (self.vy > 0.0) or (abs(self.vy) < 1e-6 and self.grav_dir > 0)
        moving_up   = (self.vy < 0.0) or (abs(self.vy) < 1e-6 and self.grav_dir < 0)

        # 0) Tentative de "stick" si on en a un actif (utile sur plateformes mobiles)
        if self._try_stick_to_support(me_now, platforms):
            return True, False  # déjà reposé proprement

        # 1) Résolution verticale prioritaire, directionnelle
        landed = False
        if moving_down:
            for pr in platforms:
                pr_in = self._inner_rect(pr)
                # besoin d'un chevauchement horizontal (avec skin) pour un contact vertical
                if me_now.right <= pr_in.left or me_now.left >= pr_in.right:
                    continue
                if me_before.bottom <= pr_in.top and me_now.bottom >= pr_in.top - EPS:
                    # poser sur le sol
                    self.y = pr_in.top - PLAYER_H
                    self.vy = 0.0
                    grounded = (self.grav_dir > 0)
                    me_now = self.rect
                    landed = True
                    # activer le "stick" pour suivre cette surface qui bouge
                    self._stick_time = 0.08   # ~80 ms
                    self._support_y = float(pr_in.top)
                    self._support_face = +1
                    break

        elif moving_up:
            for pr in platforms:
                pr_in = self._inner_rect(pr)
                if me_now.right <= pr_in.left or me_now.left >= pr_in.right:
                    continue
                if me_before.top >= pr_in.bottom and me_now.top <= pr_in.bottom + EPS:
                    # coller au plafond
                    self.y = pr_in.bottom
                    self.vy = 0.0
                    grounded = (self.grav_dir < 0)
                    me_now = self.rect
                    landed = True
                    self._stick_time = 0.08
                    self._support_y = float(pr_in.bottom)
                    self._support_face = -1
                    break

        # 2) Si pas d'atterrissage/contact, on est en l'air → on coupe le stick
        if not landed:
            grounded = False
            self._support_y = None
            self._support_face = 0
            # (la décroissance de _stick_time se fait dans update_physics)

        self.grounded = grounded
        return grounded, False

# src/game/game.py
import sys
import pygame
from pygame import K_SPACE, K_ESCAPE, K_r
from .config import (
    WIDTH, HEIGHT, FPS,
    COLOR_BG, COLOR_FG, COLOR_ACCENT, COLOR_PLAT, COLOR_DANGER,
    PLAYER_X, PLAYER_W, PLAYER_H, SCROLL_PX_PER_S
)
from .level import LevelGen
from .player import Player

def run():
    pygame.init()
    pygame.display.set_caption("Gravity Guy — playable baseline")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    def reset_world(seed=12345):
        level = LevelGen(seed=seed)
        player = Player(x=float(PLAYER_X), y=HEIGHT/2 - PLAYER_H/2, vy=0.0, grav_dir=1, grounded=False)
        return level, player, 0.0, True  # distance_px, alive

    level, player, distance_px, alive = reset_world()
    font = pygame.font.SysFont("jetbrainsmono", 18)

    # Simple button rect for restart
    btn_w, btn_h = 180, 44
    restart_rect = pygame.Rect((WIDTH - btn_w)//2, (HEIGHT - btn_h)//2, btn_w, btn_h)

    while True:
        dt = clock.tick(FPS) / 1000.0
        # Clamp dt to avoid huge physics steps on occasional stalls
        if dt > 1.0 / 30.0:
            dt = 1.0 / 30.0

        # --- Input / quit ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == K_SPACE and alive:
                    player.try_flip()
                if (event.key == K_r) and (not alive):
                    level, player, distance_px, alive = reset_world()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and (not alive):
                if restart_rect.collidepoint(event.pos):
                    level, player, distance_px, alive = reset_world()

        # --- Update world ---
        if alive:
            level.update_and_generate(dt)
            prev_y = player.y
            player.update_physics(dt)
            plat_rects = [p.rect for p in level.platforms]
            _ = player.resolve_side_collisions(plat_rects)
            _, crushed = player.resolve_collisions_swept(prev_y, plat_rects)

            distance_px += dt * SCROLL_PX_PER_S

            out_of_bounds = (player.y < -80) or (player.y > HEIGHT + 80)
            if crushed or out_of_bounds:
                alive = False

        # --- Render ---
        screen.fill(COLOR_BG)

        # Platforms
        level.draw(screen, COLOR_PLAT)

        # Player
        color_player = COLOR_ACCENT if alive else COLOR_DANGER
        pygame.draw.rect(screen, color_player, player.rect)

        # HUD
        g_txt = "↓" if player.grav_dir > 0 else "↑"
        hud = f"Dist: {int(distance_px)} px   Grav: {g_txt}   {'ALIVE' if alive else 'DEAD'}"
        text_surf = font.render(hud, True, COLOR_FG)
        screen.blit(text_surf, (12, 10))

        help_surf = font.render("SPACE = flip gravity | ESC = quit", True, (160, 180, 210))
        screen.blit(help_surf, (12, 32))

        # Restart UI when dead
        if not alive:
            # Button
            pygame.draw.rect(screen, (40, 60, 90), restart_rect, border_radius=10)
            pygame.draw.rect(screen, (90, 130, 180), restart_rect, width=2, border_radius=10)
            btn_txt = font.render("Restart (R)", True, (220, 235, 255))
            screen.blit(btn_txt, (restart_rect.centerx - btn_txt.get_width()//2,
                                  restart_rect.centery - btn_txt.get_height()//2))

        pygame.display.flip()

if __name__ == "__main__":
    run()

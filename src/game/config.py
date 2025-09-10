# --- Display ---
WIDTH = 960
HEIGHT = 540
FPS = 60

# --- World / Physics ---
SCROLL_PX_PER_S = 250.0     # map scroll speed (px/s)
G_ABS = 1800.0              # gravity magnitude (px/s^2), sign = gravity direction
JUMP_COOLDOWN_S = 0.18      # anti “ping-pong” between flips (sec)

# --- Player ---
PLAYER_X = 220              # player’s fixed x (world scrolls left)
PLAYER_W = 32
PLAYER_H = 32
MAX_VY = 1200.0              # clamp vertical velocity

# --- Level generation ---
PLATFORM_THICKNESS = 24
LANE_TOP_Y = 120            # y line of top platforms
LANE_BOT_Y = HEIGHT - 120   # y line of bottom platforms
SEGMENT_MIN_W = 160
SEGMENT_MAX_W = 360
GAP_MIN_W = 120
GAP_MAX_W = 260
SEED_DEFAULT = 12345

# --- Moving Platform Parameters ---
MOVING_PLATFORM_SPEED = 1.5    # radians per second (slower oscillation)
MOVING_PLATFORM_RANGE = 60.0   # pixels up/down from center (smaller range)
MOVING_PLATFORM_CHANCE = 0.3   # probability of a platform being moving
MIN_STATIC_BETWEEN_MOVING = 2  # minimum static platforms between moving ones

# --- Colors (RGB) ---
COLOR_BG = (9, 14, 28)
COLOR_FG = (220, 232, 255)
COLOR_ACCENT = (120, 200, 255)
COLOR_PLAT = (33, 46, 68)
COLOR_DANGER = (255, 86, 110)
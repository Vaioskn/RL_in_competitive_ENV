import math

CELLS = [(col, row) for row in range(5) for col in range(7) if not (row % 2 == 1 and col == 6)]

GOAL_CELLS, CORNER_CELLS = [(0, 2), (6, 2)], [(0, 0), (0, 4), (6, 0), (6, 4)]

# GLOBAL TRAINING SETTINGS
INFINITE_EPISODES = False
EPISODES_FOR_QUITING_PROGRAM = 5000

# Q LEARNING
EPISODES_FOR_TRAINING_Q = 10000
MIN_EPSILON_Q = 0.1
MAX_EPSILON_Q = 0.1

# MINIMAX_Q_LEARNING
MIN_ALPHA = 0.1
MAX_ALPHA = 0.1

EPISODES_FOR_TRAINING_MIN_MAX = 10000
MIN_EPSILON_MIN_MAX = 0.05
MAX_EPSILON_MIN_MAX = 0.05


# BELIEF_Q_LEARNING
EPISODES_FOR_TRAINING_BELIEF = 10000
MIN_EPSILON_BELIEF = 0.1
MAX_EPSILON_BELIEF = 0.1


# # TRAINING SETTINGS
# ENABLE_Q_LEARNING_TRAINING = False
# ENABLE_MINIMAX_Q_LEARNING_TRAINING = False
# ENABLE_BELIEF_Q_LEARNING_TRAINING = True


# Game window settings
window_width = 900
window_height = 700

# Hexagon settings
hex_size = 40
hex_width = math.sqrt(3) * hex_size
hex_height = 2 * hex_size

# Grid settings
grid_width = 11 * hex_width
grid_height = 11 * hex_size
startx = (window_width - grid_width) / 2
starty = (window_height - grid_height) / 2

# Scoreboard settings
scoreboard_x = window_width // 2
scoreboard_y = starty // 2

# Colors
white = (255, 255, 255)

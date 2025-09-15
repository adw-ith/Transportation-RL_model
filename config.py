# --- file: config.py ---

# Training settings
EPISODES = 5000
SAVE_PATH = "logistics_model_v3.weights.h5"

# Agent hyperparameters
STATE_SIZE = 145  # This should match your environment's state size
ACTION_SIZE = 21   # This should match your environment's action space size

GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 100

# Epsilon-greedy exploration settings
EPSILON_START = 1.0
EPSILON_END = 0.01
# Calculate decay rate to reach EPSILON_END in approx. 80% of episodes
EPSILON_DECAY_FRAMES = 0.8 * EPISODES
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / EPSILON_DECAY_FRAMES)

# Prioritized Experience Replay (PER) settings
PER_ALPHA = 0.6  # How much prioritization to use (0=uniform, 1=full)
PER_BETA_START = 0.4  # Initial importance sampling weight
PER_BETA_FRAMES = EPISODES
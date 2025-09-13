import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# --- Data Classes for Environment Objects ---

@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    pickup_location: tuple[int, int]
    delivery_location: tuple[int, int]
    weight: float
    # Status: 0 = waiting, 1 = assigned/in-transit, 2 = delivered
    status: int = 0

@dataclass
class Vehicle:
    """Represents a vehicle with its properties."""
    id: int
    capacity: float
    current_location: tuple[int, int]
    speed: float
    cost_per_km: float
    # Time when the vehicle becomes available for its next task
    available_at_time: int = 0

# --- The Logistics Environment ---

class LogisticsEnvironment:
    """Simulates the logistics environment for the RL agent."""
    def __init__(self, grid_size=20, n_packages=10, n_vehicles=3):
        self.grid_size = grid_size
        self.n_packages = n_packages
        self.n_vehicles = n_vehicles
        self.max_time = 1000

        self.packages = []
        self.vehicles = []
        self.current_time = 0
        self.packages_delivered = 0

        self.action_space_size = self.n_packages
        self.state_size = (self.n_vehicles * 3) + (self.n_packages * 5) + 1
        # Maximum possible distance in the grid to help with reward normalization
        self.max_dist = grid_size * 2

    def _generate_packages(self):
        """Generates a new set of random packages."""
        return [
            Package(
                id=i,
                pickup_location=(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)),
                delivery_location=(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)),
                weight=np.random.uniform(1, 10)
            ) for i in range(self.n_packages)
        ]

    def _generate_vehicles(self):
        """Generates a new set of random vehicles."""
        return [
            Vehicle(
                id=i,
                capacity=np.random.uniform(20, 50),
                current_location=(np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)),
                speed=1.0,
                cost_per_km=1.0
            ) for i in range(self.n_vehicles)
        ]

    def reset(self):
        """Resets the environment to an initial state."""
        self.packages = self._generate_packages()
        self.vehicles = self._generate_vehicles()
        self.current_time = 0
        self.packages_delivered = 0
        return self._get_state()

    def _get_state(self):
        """Constructs the normalized state vector."""
        state = []
        for vehicle in self.vehicles:
            state.extend([
                vehicle.current_location[0] / self.grid_size,
                vehicle.current_location[1] / self.grid_size,
                min(vehicle.available_at_time / self.max_time, 1.0)
            ])
        for package in self.packages:
            state.extend([
                package.status / 2.0,
                package.pickup_location[0] / self.grid_size,
                package.pickup_location[1] / self.grid_size,
                package.delivery_location[0] / self.grid_size,
                package.delivery_location[1] / self.grid_size,
            ])
        state.append(self.current_time / self.max_time)
        return np.array(state, dtype=np.float32)

    def _calculate_distance(self, loc1, loc2):
        """Calculates Manhattan distance."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def step(self, action_package_id):
        """Executes one time step in the environment with normalized rewards."""
        available_vehicle = min(self.vehicles, key=lambda v: v.available_at_time)
        self.current_time = available_vehicle.available_at_time
        
        # --- Invalid Action Penalties ---
        if not (0 <= action_package_id < len(self.packages)):
            available_vehicle.available_at_time += 1 # Small penalty for waiting
            return self._get_state(), -1.0, self.packages_delivered == self.n_packages, {}

        package_to_assign = self.packages[action_package_id]

        if package_to_assign.status != 0 or package_to_assign.weight > available_vehicle.capacity:
            available_vehicle.available_at_time += 1
            return self._get_state(), -1.0, self.packages_delivered == self.n_packages or self.current_time >= self.max_time, {}
        
        # --- Execute Valid Action ---
        package_to_assign.status = 1
        dist_to_pickup = self._calculate_distance(available_vehicle.current_location, package_to_assign.pickup_location)
        dist_to_delivery = self._calculate_distance(package_to_assign.pickup_location, package_to_assign.delivery_location)
        total_dist = dist_to_pickup + dist_to_delivery
        travel_time = total_dist / available_vehicle.speed

        delivery_completion_time = self.current_time + travel_time
        available_vehicle.current_location = package_to_assign.delivery_location
        available_vehicle.available_at_time = delivery_completion_time
        package_to_assign.status = 2
        self.packages_delivered += 1

        # --- Normalized Reward Calculation ---
        # Reward for successful delivery + penalty for travel cost
        reward = 1.0 - (total_dist / (self.max_dist * 2))

        done = False
        if self.packages_delivered == self.n_packages:
            done = True
            reward += 10.0  # Large bonus for finishing
        elif delivery_completion_time >= self.max_time:
            done = True
            reward -= 10.0 # Large penalty for timing out

        return self._get_state(), reward, done, {}

# --- Prioritized Experience Replay Buffer ---

class SumTree:
    """A SumTree data structure for efficient priority-based sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """A replay buffer that uses a SumTree to prioritize experiences."""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01
        self.max_priority = 1.0

    def add(self, experience):
        self.tree.add(self.max_priority, experience)

    def sample(self, n):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idxs, errors):
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
            
    def __len__(self):
        return self.tree.n_entries

# --- The DQN Agent ---

class DQNAgent:
    """A Dueling Deep Q-Network agent with PER and Double DQN."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(50000) # Increased memory
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

    def _build_model(self):
        """Builds a deeper and wider Dueling DQN architecture."""
        input_layer = keras.layers.Input(shape=(self.state_size,))
        
        # Common hidden layers (deeper and wider)
        common = keras.layers.Dense(256, activation='relu')(input_layer)
        common = keras.layers.Dense(256, activation='relu')(common)
        
        # Dueling Streams
        # Value Stream
        value_stream = keras.layers.Dense(128, activation='relu')(common)
        value = keras.layers.Dense(1, activation='linear')(value_stream)
        
        # Advantage Stream
        advantage_stream = keras.layers.Dense(128, activation='relu')(common)
        advantage = keras.layers.Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine streams using Keras layers
        mean_advantage = keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        advantage_sub = keras.layers.Subtract()([advantage, mean_advantage])
        q_values = keras.layers.Add()([value, advantage_sub])

        model = keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch, idxs, is_weights = self.memory.sample(self.batch_size)
        
        states = np.array([e[0] for e in minibatch], dtype=np.float32)
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch], dtype=np.float32)
        dones = np.array([e[4] for e in minibatch])

        q_values_current = self.q_network.predict(states, verbose=0)
        q_values_next_target = self.target_network.predict(next_states, verbose=0)
        q_values_next_main = self.q_network.predict(next_states, verbose=0)
        
        errors = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            old_q_value = q_values_current[i][actions[i]]
            if dones[i]:
                target = rewards[i]
            else:
                best_action = np.argmax(q_values_next_main[i])
                target_q_value = q_values_next_target[i][best_action]
                target = rewards[i] + self.gamma * target_q_value
            
            q_values_current[i][actions[i]] = target
            errors[i] = old_q_value - target

        self.memory.update(idxs, errors)
        self.q_network.fit(states, q_values_current, batch_size=self.batch_size, epochs=1, verbose=0, sample_weight=is_weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        print(f"Saving trained model weights to {file_path}")
        self.q_network.save_weights(file_path)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            print(f"Loading model weights from {file_path}")
            self.q_network.load_weights(file_path)
            self.update_target_network()
        else:
            print(f"Model file not found at {file_path}. Starting with a new model.")

# --- Main Training and Execution Logic ---

def train_optimizer(episodes=2000):
    env = LogisticsEnvironment()
    agent = DQNAgent(env.state_size, env.action_space_size)
    training_history = []
    print("Starting training...")
    update_target_network_freq = 10 

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                if episode % update_target_network_freq == 0:
                    agent.update_target_network()
                break
        agent.replay()
        training_history.append({'episode': episode, 'total_reward': total_reward, 'delivered': env.packages_delivered, 'epsilon': agent.epsilon})

        if episode % 25 == 0:
            avg_reward = np.mean([h['total_reward'] for h in training_history[-50:]])
            avg_delivered = np.mean([h['delivered'] for h in training_history[-50:]])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Delivered: {avg_delivered:.2f}, Epsilon: {agent.epsilon:.3f}")

    print("Training completed!")
    agent.save_model("logistics_model.weights.h5")
    
    # Visualization
    episodes_list = [h['episode'] for h in training_history]
    rewards_list = [h['total_reward'] for h in training_history]
    delivered_list = [h['delivered'] for h in training_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(episodes_list, rewards_list, color='b', alpha=0.6, label='Raw Reward')
    if len(rewards_list) >= 50:
        running_avg = np.convolve(rewards_list, np.ones(50)/50, mode='valid')
        ax1.plot(episodes_list[49:], running_avg, color='r', label='50-ep Moving Avg')
    ax1.set_title('Training Rewards over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(episodes_list, delivered_list, color='g')
    ax2.set_title('Packages Delivered per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Packages Delivered')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_optimizer(episodes=2000)


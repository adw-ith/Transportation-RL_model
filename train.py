import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

# --- NEW: Data Class for a Route ---

@dataclass
class Route:
    """Represents a direct route with a specific distance between two named locations."""
    start_location: str
    end_location: str
    distance: float

# --- UPDATED: Data Classes for Environment Objects ---

@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    pickup_location: str  # UPDATED: Location is now a string name
    delivery_location: str # UPDATED: Location is now a string name
    weight: float
    status: int = 0  # 0: waiting, 1: assigned, 2: delivered

@dataclass
class Vehicle:
    """Represents a vehicle with its properties."""
    id: int
    capacity: float
    current_location: str # UPDATED: Location is now a string name
    speed: float
    cost_per_km: float
    available_at_time: int = 0

# --- UPDATED: The Logistics Environment with a Defined Route Network ---

class LogisticsEnvironment:
    """
    Simulates the logistics environment using a pre-defined route network
    to calculate distances between named locations.
    """
    def __init__(self, locations, routes, n_packages=10, n_vehicles=3):
        self.n_packages = n_packages
        self.n_vehicles = n_vehicles
        self.max_time = 1000

        # UPDATED: Environment is defined by named locations and routes
        self.all_locations = sorted(locations)
        self.routes = routes
        self.num_locations = len(self.all_locations)
        self.location_to_idx = {loc: i for i, loc in enumerate(self.all_locations)}
        self.distance_matrix = None # Will be computed once in init
        self._create_distance_matrix() # Create the all-pairs shortest path matrix

        self.packages = []
        self.vehicles = []
        self.current_time = 0
        self.packages_delivered = 0

        self.action_space_size = self.n_packages
        # UPDATED: State size calculation based on the new representation
        self.state_size = (self.n_vehicles * 3) + (self.n_packages * 4) + 1

    def _generate_packages(self):
        """Generates packages with random named locations."""
        packages = []
        for i in range(self.n_packages):
            pickup = random.choice(self.all_locations)
            delivery = random.choice([loc for loc in self.all_locations if loc != pickup])
            packages.append(Package(
                id=i,
                pickup_location=pickup,
                delivery_location=delivery,
                weight=np.random.uniform(1, 10)
            ))
        return packages

    def _generate_vehicles(self):
        """Generates vehicles starting at random named locations."""
        return [
            Vehicle(
                id=i,
                capacity=np.random.uniform(20, 50),
                current_location=random.choice(self.all_locations),
                speed=1.0,
                cost_per_km=1.0
            ) for i in range(self.n_vehicles)
        ]

    def _create_distance_matrix(self):
        """
        Creates an all-pairs shortest path distance matrix using the
        Floyd-Warshall algorithm from the defined routes. This allows finding
        the distance between any two locations in the network.
        """
        num_locs = self.num_locations
        dist_matrix = np.full((num_locs, num_locs), np.inf)
        np.fill_diagonal(dist_matrix, 0)

        # Populate the matrix with distances from direct routes
        for route in self.routes:
            idx1 = self.location_to_idx.get(route.start_location)
            idx2 = self.location_to_idx.get(route.end_location)
            if idx1 is not None and idx2 is not None:
                dist_matrix[idx1, idx2] = route.distance
                dist_matrix[idx2, idx1] = route.distance # Assume routes are bidirectional

        # Floyd-Warshall algorithm to find all-pairs shortest paths
        for k in range(num_locs):
            for i in range(num_locs):
                for j in range(num_locs):
                    dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
        
        self.distance_matrix = dist_matrix
        if np.any(np.isinf(self.distance_matrix)):
            print("Warning: The route network graph is not fully connected.")

    def load_scenario(self, packages, vehicles):
        """
        Loads a specific scenario for testing. The number of packages and vehicles
        must match the numbers the environment was initialized with to ensure
        the state size is correct for the neural network.
        """
        if len(packages) != self.n_packages or len(vehicles) != self.n_vehicles:
            raise ValueError(
                f"Scenario size mismatch. Environment initialized for "
                f"{self.n_packages} packages and {self.n_vehicles} vehicles, but "
                f"scenario has {len(packages)} packages and {len(vehicles)} vehicles."
            )
        self.packages = packages
        self.vehicles = vehicles
        self.current_time = 0
        self.packages_delivered = 0
        # Reset vehicle availability and status for a clean test run
        for v in self.vehicles:
            v.available_at_time = 0
        for p in self.packages:
            p.status = 0
        print("Custom scenario loaded successfully.")
        return self._get_state()

    def reset(self):
        """Resets the environment for a new episode."""
        self.packages = self._generate_packages()
        self.vehicles = self._generate_vehicles()
        self.current_time = 0
        self.packages_delivered = 0
        return self._get_state()

    def _get_distance(self, loc1_name, loc2_name):
        """Looks up the shortest distance between two named locations from the matrix."""
        idx1 = self.location_to_idx[loc1_name]
        idx2 = self.location_to_idx[loc2_name]
        return self.distance_matrix[idx1, idx2]

    def _get_state(self):
        """
        Constructs the state vector for the RL agent.
        Locations are converted to their normalized index.
        """
        state = []
        max_cap = 50.0  # Used for normalization
        max_weight = 10.0 # Used for normalization
        
        for vehicle in self.vehicles:
            loc_idx = self.location_to_idx[vehicle.current_location]
            state.extend([
                loc_idx / self.num_locations,
                vehicle.capacity / max_cap,
                min(vehicle.available_at_time / self.max_time, 1.0)
            ])
            
        for package in self.packages:
            pickup_idx = self.location_to_idx[package.pickup_location]
            delivery_idx = self.location_to_idx[package.delivery_location]
            state.extend([
                package.status / 2.0,
                pickup_idx / self.num_locations,
                delivery_idx / self.num_locations,
                package.weight / max_weight
            ])
            
        state.append(self.current_time / self.max_time)
        return np.array(state, dtype=np.float32)

    def step(self, action_package_id):
        """Executes one time step based on the chosen action."""
        available_vehicle = min(self.vehicles, key=lambda v: v.available_at_time)
        self.current_time = available_vehicle.available_at_time
        
        # Handle invalid action
        if not (0 <= action_package_id < len(self.packages)):
            available_vehicle.available_at_time += 1 # Penalize by waiting
            return self._get_state(), -1.0, self.packages_delivered == self.n_packages, {}

        package_to_assign = self.packages[action_package_id]

        # Handle invalid assignment (package already assigned or too heavy)
        if package_to_assign.status != 0 or package_to_assign.weight > available_vehicle.capacity:
            available_vehicle.available_at_time += 1 # Penalize by waiting
            return self._get_state(), -1.0, self.packages_delivered == self.n_packages or self.current_time >= self.max_time, {}
        
        package_to_assign.status = 1
        
        # UPDATED: Use the distance matrix for all travel calculations
        dist_to_pickup = self._get_distance(available_vehicle.current_location, package_to_assign.pickup_location)
        dist_to_delivery = self._get_distance(package_to_assign.pickup_location, package_to_assign.delivery_location)
        total_dist = dist_to_pickup + dist_to_delivery

        # Check for unreachable locations
        if np.isinf(total_dist):
             available_vehicle.available_at_time += 1 # Penalize for trying an impossible route
             package_to_assign.status = 0 # Revert status
             return self._get_state(), -10.0, self.current_time >= self.max_time, {}

        travel_time = total_dist / available_vehicle.speed
        delivery_completion_time = self.current_time + travel_time

        # Update vehicle and package states
        available_vehicle.current_location = package_to_assign.delivery_location
        available_vehicle.available_at_time = delivery_completion_time
        package_to_assign.status = 2
        self.packages_delivered += 1

        # UPDATED: Reward is now based on the route network distance
        max_possible_dist = 2 * np.max(self.distance_matrix[np.isfinite(self.distance_matrix)])
        reward = 1.0 - (total_dist / max_possible_dist)

        done = False
        if self.packages_delivered == self.n_packages:
            done = True
            reward += 10.0
        elif delivery_completion_time >= self.max_time:
            done = True
            reward -= 10.0

        return self._get_state(), reward, done, {}

# --- Prioritized Experience Replay Buffer (UNCHANGED) ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree): return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree, self.alpha, self.beta, self.beta_increment = SumTree(capacity), alpha, beta, beta_increment_per_sampling
        self.epsilon, self.max_priority = 0.01, 1.0
    def add(self, experience): self.tree.add(self.max_priority, experience)
    def sample(self, n):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment])
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            priorities.append(p); batch.append(data); idxs.append(idx)
        probs = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * probs, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight
    def update(self, idxs, errors):
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    def __len__(self): return self.tree.n_entries

# --- The DQN Agent (UNCHANGED) ---

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.01, 0.9988
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

    def _build_model(self):
        input_layer = keras.layers.Input(shape=(self.state_size,))
        common = keras.layers.Dense(256, activation='relu')(input_layer)
        common = keras.layers.Dense(256, activation='relu')(common)
        value_stream = keras.layers.Dense(128, activation='relu')(common)
        value = keras.layers.Dense(1, activation='linear')(value_stream)
        advantage_stream = keras.layers.Dense(128, activation='relu')(common)
        advantage = keras.layers.Dense(self.action_size, activation='linear')(advantage_stream)
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
        return np.argmax(self.q_network.predict(state.reshape(1, -1), verbose=0)[0])

    def replay(self):
        if len(self.memory) < self.batch_size: return
        minibatch, idxs, is_weights = self.memory.sample(self.batch_size)
        states = np.array([e[0] for e in minibatch], dtype=np.float32)
        actions, rewards = np.array([e[1] for e in minibatch]), np.array([e[2] for e in minibatch])
        next_states, dones = np.array([e[3] for e in minibatch], dtype=np.float32), np.array([e[4] for e in minibatch])
        q_values_current = self.q_network.predict(states, verbose=0)
        q_values_next_target = self.target_network.predict(next_states, verbose=0)
        q_values_next_main = self.q_network.predict(next_states, verbose=0)
        errors = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            old_q, target = q_values_current[i][actions[i]], rewards[i]
            if not dones[i]:
                best_action = np.argmax(q_values_next_main[i])
                target += self.gamma * q_values_next_target[i][best_action]
            q_values_current[i][actions[i]], errors[i] = target, old_q - target
        self.memory.update(idxs, errors)
        self.q_network.fit(states, q_values_current, batch_size=self.batch_size, epochs=1, verbose=0, sample_weight=is_weights)
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        print(f"Saving trained model weights to {file_path}")
        self.q_network.save_weights(file_path)

# --- Main Training and Execution Logic ---

def train_optimizer(episodes=2000, model_path="logistics_model.weights.h5"):
    # UPDATED: Define the logistics network
    LOCATIONS = ["Warehouse A", "Hub B", "City Center C", "Suburb D", "Industrial E"]
    ROUTES = [
        Route("Warehouse A", "Hub B", 15),
        Route("Warehouse A", "Suburb D", 20),
        Route("Hub B", "City Center C", 10),
        Route("City Center C", "Suburb D", 12),
        Route("City Center C", "Industrial E", 8),
        Route("Suburb D", "Industrial E", 25)
    ]
    
    # UPDATED: Instantiate the environment with the new network definition
    env = LogisticsEnvironment(locations=LOCATIONS, routes=ROUTES, n_packages=10, n_vehicles=3)
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
        training_history.append({'total_reward': total_reward, 'delivered': env.packages_delivered})

        if episode % 25 == 0 or episode == episodes - 1:
            avg_reward = np.mean([h['total_reward'] for h in training_history[-50:]])
            avg_delivered = np.mean([h['delivered'] for h in training_history[-50:]])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Delivered: {avg_delivered:.2f}, Epsilon: {agent.epsilon:.3f}")

    print("Training completed!")
    agent.save_model(model_path)
    # Visualization can be added here as before
    
if __name__ == "__main__":
    train_optimizer(episodes=5000)
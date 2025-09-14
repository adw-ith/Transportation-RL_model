import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any
import pickle

# ========================= DATA CLASSES =========================

@dataclass
class Route:
    """Represents a route between two locations"""
    start_location: str
    end_location: str
    distance: float
    traffic_factor: float = 1.0  # For future: traffic conditions

@dataclass
class Package:
    """Represents a package to be delivered"""
    id: int
    pickup_location: str
    delivery_location: str
    weight: float
    priority: int = 1  # 1=normal, 2=express, 3=urgent
    status: int = 0  # 0: waiting, 1: in-transit, 2: delivered
    time_window: Tuple[float, float] = (0, float('inf'))  # pickup time window

@dataclass
class Vehicle:
    """Represents a delivery vehicle"""
    id: int
    capacity: float
    current_location: str
    speed: float
    cost_per_km: float
    available_at_time: float = 0
    inventory: list = field(default_factory=list)
    current_capacity: float = 0.0
    total_distance_traveled: float = 0.0

# ========================= IMPROVED ENVIRONMENT =========================

class ImprovedLogisticsEnvironment:
    """Enhanced environment with better state representation and reward structure"""
    
    def __init__(self, max_locations=20, max_packages=50, max_vehicles=10):
        # Maximum sizes for normalization
        self.max_locations = max_locations
        self.max_packages = max_packages
        self.max_vehicles = max_vehicles
        self.max_time = 10000
        
        # Will be set when scenario is loaded
        self.locations = []
        self.routes = []
        self.packages = []
        self.vehicles = []
        self.distance_matrix = None
        
        # State tracking
        self.current_time = 0
        self.packages_delivered = 0
        self.total_distance = 0
        self.total_cost = 0
        
        # Enhanced state representation
        self.state_size = self._calculate_state_size()
        self.action_space_size = max_locations + 1  # +1 for "wait" action
        
    def _calculate_state_size(self):
        """Calculate the fixed state vector size"""
        # Global features: time, delivery progress, efficiency metrics
        global_features = 5
        
        # Vehicle features (for active vehicle)
        vehicle_features = 10
        
        # Package features (top N most relevant packages)
        max_packages_in_state = 20
        package_features = max_packages_in_state * 6
        
        # Location features (proximity info)
        location_features = 10
        
        return global_features + vehicle_features + package_features + location_features
    
    def load_scenario(self, locations: List[str], routes: List[Route], 
                     packages: List[Package], vehicles: List[Vehicle]):
        """Load a specific scenario"""
        self.locations = sorted(locations)
        self.routes = routes
        self.packages = packages
        self.vehicles = vehicles
        
        self.num_locations = len(self.locations)
        self.location_to_idx = {loc: i for i, loc in enumerate(self.locations)}
        
        # Build distance matrix
        self._create_distance_matrix()
        
        # Reset scenario
        self._reset_scenario()
        
        return self._get_state()
    
    def _create_distance_matrix(self):
        """Create distance matrix using Floyd-Warshall algorithm"""
        n = len(self.locations)
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        
        for route in self.routes:
            i = self.location_to_idx.get(route.start_location)
            j = self.location_to_idx.get(route.end_location)
            if i is not None and j is not None:
                dist[i][j] = min(dist[i][j], route.distance * route.traffic_factor)
                dist[j][i] = min(dist[j][i], route.distance * route.traffic_factor)
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        self.distance_matrix = dist
    
    def _reset_scenario(self):
        """Reset the scenario to initial state"""
        self.current_time = 0
        self.packages_delivered = 0
        self.total_distance = 0
        self.total_cost = 0
        
        for v in self.vehicles:
            v.available_at_time = 0
            v.inventory = []
            v.current_capacity = v.capacity
            v.total_distance_traveled = 0
        
        for p in self.packages:
            p.status = 0
    
    def _get_state(self):
        """Get enhanced state representation"""
        state = []
        
        # 1. Global features
        delivery_progress = self.packages_delivered / max(len(self.packages), 1)
        time_progress = min(self.current_time / self.max_time, 1.0)
        
        # Calculate efficiency metrics
        avg_vehicle_utilization = np.mean([
            (v.capacity - v.current_capacity) / v.capacity 
            for v in self.vehicles
        ]) if self.vehicles else 0
        
        waiting_packages = sum(1 for p in self.packages if p.status == 0)
        in_transit_packages = sum(1 for p in self.packages if p.status == 1)
        
        state.extend([
            delivery_progress,
            time_progress,
            avg_vehicle_utilization,
            waiting_packages / max(len(self.packages), 1),
            in_transit_packages / max(len(self.packages), 1)
        ])
        
        # 2. Active vehicle features
        vehicle = self._get_active_vehicle()
        if vehicle:
            loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
            
            # Calculate nearest package distances
            nearest_pickup_dist = self._get_nearest_package_distance(vehicle, pickup=True)
            nearest_delivery_dist = self._get_nearest_package_distance(vehicle, pickup=False)
            
            state.extend([
                loc_idx / max(self.num_locations, 1),
                vehicle.capacity / 100.0,  # Normalized assuming max capacity of 100
                (vehicle.capacity - vehicle.current_capacity) / vehicle.capacity,
                vehicle.speed / 2.0,  # Normalized assuming max speed of 2
                vehicle.cost_per_km / 5.0,  # Normalized
                len(vehicle.inventory) / 10.0,  # Normalized
                vehicle.available_at_time / self.max_time,
                vehicle.total_distance_traveled / 1000.0,  # Normalized
                nearest_pickup_dist / 100.0,  # Normalized
                nearest_delivery_dist / 100.0  # Normalized
            ])
        else:
            state.extend([0] * 10)
        
        # 3. Package features (top 20 most relevant)
        relevant_packages = self._get_relevant_packages(vehicle, max_count=20)
        package_features = []
        
        for p in relevant_packages:
            pickup_idx = self.location_to_idx.get(p.pickup_location, 0)
            delivery_idx = self.location_to_idx.get(p.delivery_location, 0)
            
            # Distance from vehicle to package
            if vehicle:
                vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
                dist_to_pickup = self.distance_matrix[vehicle_loc_idx][pickup_idx]
            else:
                dist_to_pickup = 0
            
            package_features.extend([
                p.status / 2.0,
                pickup_idx / max(self.num_locations, 1),
                delivery_idx / max(self.num_locations, 1),
                p.weight / 20.0,  # Normalized
                p.priority / 3.0,
                dist_to_pickup / 100.0  # Normalized
            ])
        
        # Pad if fewer packages
        while len(package_features) < 20 * 6:
            package_features.extend([0] * 6)
        
        state.extend(package_features[:20 * 6])
        
        # 4. Location density features
        location_features = self._get_location_density_features(vehicle)
        state.extend(location_features)
        
        return np.array(state[:self.state_size], dtype=np.float32)
    
    def _get_active_vehicle(self):
        """Get the vehicle that should act next"""
        if not self.vehicles:
            return None
        return min(self.vehicles, key=lambda v: v.available_at_time)
    
    def _get_nearest_package_distance(self, vehicle, pickup=True):
        """Get distance to nearest package pickup or delivery"""
        if not vehicle:
            return 0
        
        vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
        min_dist = float('inf')
        
        if pickup:
            for p in self.packages:
                if p.status == 0:
                    pkg_loc_idx = self.location_to_idx.get(p.pickup_location, 0)
                    dist = self.distance_matrix[vehicle_loc_idx][pkg_loc_idx]
                    min_dist = min(min_dist, dist)
        else:
            for p in vehicle.inventory:
                pkg_loc_idx = self.location_to_idx.get(p.delivery_location, 0)
                dist = self.distance_matrix[vehicle_loc_idx][pkg_loc_idx]
                min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 0
    
    def _get_relevant_packages(self, vehicle, max_count=20):
        """Get most relevant packages for state representation"""
        if not vehicle:
            return self.packages[:max_count]
        
        # Combine packages in inventory and waiting packages
        relevant = list(vehicle.inventory)
        
        # Add waiting packages sorted by distance and priority
        vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
        waiting = [(p, self.distance_matrix[vehicle_loc_idx][self.location_to_idx.get(p.pickup_location, 0)])
                   for p in self.packages if p.status == 0]
        waiting.sort(key=lambda x: (x[1], -x[0].priority))
        
        for p, _ in waiting:
            if len(relevant) >= max_count:
                break
            relevant.append(p)
        
        return relevant[:max_count]
    
    def _get_location_density_features(self, vehicle):
        """Get features about package density at nearby locations"""
        features = []
        
        if not vehicle:
            return [0] * 10
        
        vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
        
        # Count packages at each location
        pickup_density = {}
        delivery_density = {}
        
        for p in self.packages:
            if p.status == 0:
                pickup_density[p.pickup_location] = pickup_density.get(p.pickup_location, 0) + 1
            if p.status == 1:
                delivery_density[p.delivery_location] = delivery_density.get(p.delivery_location, 0) + 1
        
        # Get top 5 nearest locations with packages
        location_scores = []
        for loc in self.locations:
            if loc == vehicle.current_location:
                continue
            loc_idx = self.location_to_idx[loc]
            dist = self.distance_matrix[vehicle_loc_idx][loc_idx]
            pickups = pickup_density.get(loc, 0)
            deliveries = delivery_density.get(loc, 0)
            if pickups + deliveries > 0:
                location_scores.append((dist, pickups + deliveries * 2))  # Prioritize deliveries
        
        location_scores.sort(key=lambda x: x[0])
        
        for i in range(5):
            if i < len(location_scores):
                features.extend([location_scores[i][0] / 100.0, location_scores[i][1] / 10.0])
            else:
                features.extend([0, 0])
        
        return features
    
    def get_valid_actions_mask(self):
        """Get mask for valid actions"""
        vehicle = self._get_active_vehicle()
        if not vehicle:
            return np.zeros(self.action_space_size)
        
        mask = np.zeros(self.action_space_size)
        vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
        
        # Valid pickup locations
        for p in self.packages:
            if p.status == 0 and p.weight <= vehicle.current_capacity:
                loc_idx = self.location_to_idx.get(p.pickup_location, 0)
                if loc_idx != vehicle_loc_idx and loc_idx < self.action_space_size:
                    mask[loc_idx] = 1
        
        # Valid delivery locations
        for p in vehicle.inventory:
            loc_idx = self.location_to_idx.get(p.delivery_location, 0)
            if loc_idx != vehicle_loc_idx and loc_idx < self.action_space_size:
                mask[loc_idx] = 1
        
        # Wait action (last index) - always valid
        mask[-1] = 1
        
        return mask
    
    def step(self, action):
        """Execute action and return new state"""
        vehicle = self._get_active_vehicle()
        if not vehicle:
            return self._get_state(), -100, True, {}
        
        self.current_time = vehicle.available_at_time
        
        # Handle wait action
        if action == self.action_space_size - 1:
            vehicle.available_at_time += 10  # Wait for 10 time units
            return self._get_state(), -5, False, {}  # Small penalty for waiting
        
        # Handle movement to location
        if action >= len(self.locations):
            return self._get_state(), -50, False, {}  # Invalid action
        
        destination = self.locations[action]
        vehicle_loc_idx = self.location_to_idx.get(vehicle.current_location, 0)
        dest_loc_idx = action
        
        distance = self.distance_matrix[vehicle_loc_idx][dest_loc_idx]
        if np.isinf(distance):
            return self._get_state(), -100, False, {}  # Impossible move
        
        # Calculate travel time and cost
        travel_time = distance / vehicle.speed
        travel_cost = distance * vehicle.cost_per_km
        
        # Update vehicle state
        vehicle.current_location = destination
        vehicle.available_at_time += travel_time
        vehicle.total_distance_traveled += distance
        self.total_distance += distance
        self.total_cost += travel_cost
        
        # Initialize reward
        reward = -travel_cost * 0.1  # Base travel cost
        
        # Process deliveries
        delivered = []
        for p in list(vehicle.inventory):
            if p.delivery_location == destination:
                vehicle.inventory.remove(p)
                vehicle.current_capacity += p.weight
                p.status = 2
                self.packages_delivered += 1
                delivered.append(p)
                
                # Reward based on priority
                reward += 50 * p.priority
        
        # Process pickups
        picked_up = []
        available_packages = [p for p in self.packages 
                            if p.status == 0 and p.pickup_location == destination]
        
        # Sort by priority and weight
        available_packages.sort(key=lambda p: (-p.priority, p.weight))
        
        for p in available_packages:
            if p.weight <= vehicle.current_capacity:
                # Check time window
                if self.current_time <= p.time_window[1]:
                    vehicle.inventory.append(p)
                    vehicle.current_capacity -= p.weight
                    p.status = 1
                    picked_up.append(p)
                    
                    # Reward for pickup
                    reward += 10 * p.priority
                    
                    # Bonus for picking up within time window
                    if self.current_time >= p.time_window[0]:
                        reward += 5
        
        # Penalties and bonuses
        if len(delivered) == 0 and len(picked_up) == 0:
            reward -= 30  # Penalty for useless trip
        
        # Efficiency bonus
        if vehicle.current_capacity < vehicle.capacity * 0.2:  # Vehicle is well utilized
            reward += 5
        
        # Check termination
        done = (self.packages_delivered == len(self.packages) or 
                self.current_time >= self.max_time)
        
        if done and self.packages_delivered == len(self.packages):
            # Completion bonus
            reward += 500
            # Efficiency bonus
            reward += max(0, (self.max_time - self.current_time) / 10)
        
        info = {
            'delivered': len(delivered),
            'picked_up': len(picked_up),
            'total_delivered': self.packages_delivered,
            'current_time': self.current_time,
            'total_distance': self.total_distance,
            'total_cost': self.total_cost
        }
        
        return self._get_state(), reward, done, info

# ========================= IMPROVED DQN AGENT =========================

class DuelingDQNNetwork(keras.Model):
    """Dueling DQN architecture for better value estimation"""
    
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # Shared layers
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.2)
        
        # Value stream
        self.value_dense = layers.Dense(128, activation='relu')
        self.value_out = layers.Dense(1)
        
        # Advantage stream
        self.advantage_dense = layers.Dense(128, activation='relu')
        self.advantage_out = layers.Dense(action_size)
        
    def call(self, state, training=False):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        # Value stream
        value = self.value_dense(x)
        value = self.value_out(value)
        
        # Advantage stream
        advantage = self.advantage_dense(x)
        advantage = self.advantage_out(advantage)
        
        # Combine streams
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        return q_values

class ImprovedDQNAgent:
    """Enhanced DQN agent with Double DQN and Dueling architecture"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.update_target_freq = 100
        self.training_step = 0
        
        # Networks
        self.q_network = DuelingDQNNetwork(state_size, action_size)
        self.target_network = DuelingDQNNetwork(state_size, action_size)
        
        # Build networks
        dummy_state = tf.zeros((1, state_size))
        self.q_network(dummy_state)
        self.target_network(dummy_state)
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Initialize target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done, mask, next_mask):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done, mask, next_mask))
    
    def act(self, state, valid_actions_mask):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions_mask == 1)[0]
            if len(valid_indices) > 0:
                return np.random.choice(valid_indices)
            return np.random.randint(self.action_size)
        
        # Greedy action
        state_tensor = tf.expand_dims(state, 0)
        q_values = self.q_network(state_tensor, training=False).numpy()[0]
        
        # Apply mask
        masked_q_values = q_values + (1 - valid_actions_mask) * -1e9
        return np.argmax(masked_q_values)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        next_masks = np.array([e[6] for e in batch])
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        # Train step
        with tf.GradientTape() as tape:
            # Current Q values
            current_q_values = self.q_network(states_tensor, training=True)
            current_q = tf.gather_nd(current_q_values, 
                                    tf.stack([tf.range(self.batch_size), actions], axis=1))
            
            # Double DQN: use main network to select action, target network to evaluate
            next_q_values = self.q_network(next_states_tensor, training=False)
            masked_next_q = next_q_values + (1 - next_masks) * -1e9
            next_actions = tf.argmax(masked_next_q, axis=1)
            
            target_next_q_values = self.target_network(next_states_tensor, training=False)
            next_actions = tf.cast(next_actions, dtype=tf.int32)
            next_q = tf.gather_nd(target_next_q_values,
                                 tf.stack([tf.range(self.batch_size), next_actions], axis=1))
            
            # Calculate targets
            targets = rewards + self.gamma * next_q * (1 - dones)
            
            # Loss
            loss = tf.reduce_mean(tf.square(targets - current_q))
        
        # Backpropagation
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model weights"""
        self.q_network.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights"""
        self.q_network.load_weights(filepath)
        self.update_target_network()

# ========================= TRAINING FUNCTION =========================

def train_model(episodes=3000, save_path="improved_logistics_model"):
    """Train the improved logistics model"""
    
    # Create environment
    env = ImprovedLogisticsEnvironment()
    
    # Create agent
    agent = ImprovedDQNAgent(env.state_size, env.action_space_size)
    
    # Training history
    history = {
        'episode': [],
        'total_reward': [],
        'packages_delivered': [],
        'completion_time': [],
        'total_distance': [],
        'epsilon': []
    }
    
    print("Starting training...")
    print(f"State size: {env.state_size}, Action size: {env.action_space_size}")
    
    for episode in range(episodes):
        # Generate random scenario for training
        locations, routes, packages, vehicles = generate_random_scenario()
        
        # Load scenario
        state = env.load_scenario(locations, routes, packages, vehicles)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            # Get valid actions
            mask = env.get_valid_actions_mask()
            
            # Choose action
            action = agent.act(state, mask)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            next_mask = env.get_valid_actions_mask()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done, mask, next_mask)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        # Record history
        history['episode'].append(episode)
        history['total_reward'].append(total_reward)
        history['packages_delivered'].append(env.packages_delivered)
        history['completion_time'].append(env.current_time)
        history['total_distance'].append(env.total_distance)
        history['epsilon'].append(agent.epsilon)
        
        
        avg_reward = np.mean(history['total_reward'][-100:]) if len(history['total_reward']) >= 100 else total_reward
        avg_delivered = np.mean(history['packages_delivered'][-100:]) if len(history['packages_delivered']) >= 100 else env.packages_delivered
        print(f"Episode {episode}/{episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Packages Delivered: {avg_delivered:.2f}/{len(packages)}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        print()
    
    # Save model
    agent.save(save_path)
    
    # Save training history
    with open(f"{save_path}_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print("Training completed!")
    return agent, history

def generate_random_scenario():
    """Generate a random scenario for training"""
    
    # Random locations
    num_locations = random.randint(5, 15)
    locations = [f"Location_{i}" for i in range(num_locations)]
    
    # Random routes (ensure connectivity)
    routes = []
    for i in range(num_locations):
        for j in range(i + 1, min(i + 4, num_locations)):
            if random.random() < 0.7:  # 70% chance of connection
                distance = random.uniform(5, 50)
                routes.append(Route(locations[i], locations[j], distance))
    
    # Ensure full connectivity
    for i in range(num_locations - 1):
        route_exists = any(
            (r.start_location == locations[i] and r.end_location == locations[i + 1]) or
            (r.start_location == locations[i + 1] and r.end_location == locations[i])
            for r in routes
        )
        if not route_exists:
            routes.append(Route(locations[i], locations[i + 1], random.uniform(10, 30)))
    
    # Random packages
    num_packages = random.randint(5, 20)
    packages = []
    for i in range(num_packages):
        pickup = random.choice(locations)
        delivery = random.choice([loc for loc in locations if loc != pickup])
        packages.append(Package(
            id=i,
            pickup_location=pickup,
            delivery_location=delivery,
            weight=random.uniform(1, 15),
            priority=random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        ))
    
    # Random vehicles
    num_vehicles = random.randint(2, 5)
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(Vehicle(
            id=i,
            capacity=random.uniform(30, 60),
            current_location=random.choice(locations),
            speed=random.uniform(0.8, 1.5),
            cost_per_km=random.uniform(0.5, 2.0)
        ))
        vehicles[-1].current_capacity = vehicles[-1].capacity
    
    return locations, routes, packages, vehicles

# ========================= INFERENCE CLASS =========================

class LogisticsOptimizer:
    """Main class for using the trained model"""
    
    def __init__(self, model_path="improved_logistics_model"):
        self.env = ImprovedLogisticsEnvironment()
        self.agent = ImprovedDQNAgent(self.env.state_size, self.env.action_space_size)
        self.agent.load(model_path)
        self.agent.epsilon = 0  # No exploration during inference
    
    def optimize_routes(self, scenario_dict):
        """
        Optimize routes for given scenario
        
        Input format:
        {
            "locations": ["Location_A", "Location_B", ...],
            "routes": [
                {"start": "Location_A", "end": "Location_B", "distance": 10.5, "traffic_factor": 1.0},
                ...
            ],
            "packages": [
                {
                    "id": 1,
                    "pickup": "Location_A",
                    "delivery": "Location_C",
                    "weight": 5.2,
                    "priority": 2,  # 1=normal, 2=express, 3=urgent
                    "time_window": [0, 1000]  # Optional
                },
                ...
            ],
            "vehicles": [
                {
                    "id": 1,
                    "capacity": 40.0,
                    "location": "Location_A",
                    "speed": 1.0,
                    "cost_per_km": 1.5
                },
                ...
            ]
        }
        
        Output format:
        {
            "success": bool,
            "execution_plan": [
                {
                    "time": 0,
                    "vehicle_id": 1,
                    "action": "move_to",
                    "destination": "Location_B",
                    "pickups": [package_ids],
                    "deliveries": [package_ids]
                },
                ...
            ],
            "metrics": {
                "total_time": float,
                "total_distance": float,
                "total_cost": float,
                "packages_delivered": int,
                "delivery_rate": float
            },
            "vehicle_routes": {
                vehicle_id: ["Location_A", "Location_B", ...]
            }
        }
        """
        
        # Parse input
        locations = scenario_dict["locations"]
        
        routes = [
            Route(
                start_location=r["start"],
                end_location=r["end"],
                distance=r["distance"],
                traffic_factor=r.get("traffic_factor", 1.0)
            )
            for r in scenario_dict["routes"]
        ]
        
        packages = [
            Package(
                id=p["id"],
                pickup_location=p["pickup"],
                delivery_location=p["delivery"],
                weight=p["weight"],
                priority=p.get("priority", 1),
                time_window=tuple(p.get("time_window", [0, float('inf')]))
            )
            for p in scenario_dict["packages"]
        ]
        
        vehicles = [
            Vehicle(
                id=v["id"],
                capacity=v["capacity"],
                current_location=v["location"],
                speed=v.get("speed", 1.0),
                cost_per_km=v.get("cost_per_km", 1.0),
                current_capacity=v["capacity"]
            )
            for v in scenario_dict["vehicles"]
        ]
        
        # Load scenario
        state = self.env.load_scenario(locations, routes, packages, vehicles)
        
        # Execute optimization
        execution_plan = []
        vehicle_routes = {v.id: [v.current_location] for v in vehicles}
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Get current vehicle
            active_vehicle = self.env._get_active_vehicle()
            if not active_vehicle:
                break
            
            # Get valid actions
            mask = self.env.get_valid_actions_mask()
            
            # Choose best action
            action = self.agent.act(state, mask)
            
            # Record state before action
            prev_location = active_vehicle.current_location
            prev_time = self.env.current_time
            prev_inventory = list(active_vehicle.inventory)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Record action details
            if action < len(self.env.locations):
                destination = self.env.locations[action]
                
                # Determine what was picked up and delivered
                pickups = []
                deliveries = []
                
                if destination != prev_location:
                    vehicle_routes[active_vehicle.id].append(destination)
                    
                    # Check deliveries
                    for p in prev_inventory:
                        if p not in active_vehicle.inventory and p.status == 2:
                            deliveries.append(p.id)
                    
                    # Check pickups
                    for p in active_vehicle.inventory:
                        if p not in prev_inventory:
                            pickups.append(p.id)
                    
                    execution_plan.append({
                        "time": prev_time,
                        "vehicle_id": active_vehicle.id,
                        "action": "move_to",
                        "destination": destination,
                        "pickups": pickups,
                        "deliveries": deliveries,
                        "distance": info.get("distance", 0),
                        "cost": info.get("cost", 0)
                    })
            elif action == self.env.action_space_size - 1:
                # Wait action
                execution_plan.append({
                    "time": prev_time,
                    "vehicle_id": active_vehicle.id,
                    "action": "wait",
                    "duration": 10
                })
            
            state = next_state
            steps += 1
        
        # Compile results
        result = {
            "success": self.env.packages_delivered == len(packages),
            "execution_plan": execution_plan,
            "metrics": {
                "total_time": self.env.current_time,
                "total_distance": self.env.total_distance,
                "total_cost": self.env.total_cost,
                "packages_delivered": self.env.packages_delivered,
                "total_packages": len(packages),
                "delivery_rate": self.env.packages_delivered / len(packages) if packages else 0,
                "vehicles_used": len(set(ep["vehicle_id"] for ep in execution_plan if ep["action"] == "move_to"))
            },
            "vehicle_routes": vehicle_routes,
            "undelivered_packages": [p.id for p in packages if p.status != 2]
        }
        
        return result
    
    def evaluate_scenario(self, scenario_dict, num_runs=5):
        """
        Evaluate a scenario multiple times to get average performance
        """
        results = []
        
        for _ in range(num_runs):
            result = self.optimize_routes(scenario_dict)
            results.append(result)
        
        # Calculate average metrics
        avg_metrics = {
            "avg_time": np.mean([r["metrics"]["total_time"] for r in results]),
            "avg_distance": np.mean([r["metrics"]["total_distance"] for r in results]),
            "avg_cost": np.mean([r["metrics"]["total_cost"] for r in results]),
            "avg_delivery_rate": np.mean([r["metrics"]["delivery_rate"] for r in results]),
            "std_time": np.std([r["metrics"]["total_time"] for r in results]),
            "std_cost": np.std([r["metrics"]["total_cost"] for r in results])
        }
        
        return avg_metrics, results

# ========================= USAGE EXAMPLES =========================

def example_usage():
    """Example of how to use the model"""
    
    # Example scenario
    scenario = {
        "locations": ["Warehouse", "Store_A", "Store_B", "Store_C", "Hub"],
        "routes": [
            {"start": "Warehouse", "end": "Hub", "distance": 20},
            {"start": "Hub", "end": "Store_A", "distance": 15},
            {"start": "Hub", "end": "Store_B", "distance": 12},
            {"start": "Store_A", "end": "Store_B", "distance": 8},
            {"start": "Store_B", "end": "Store_C", "distance": 10},
            {"start": "Store_C", "end": "Warehouse", "distance": 25}
        ],
        "packages": [
            {"id": 1, "pickup": "Warehouse", "delivery": "Store_A", "weight": 5, "priority": 2},
            {"id": 2, "pickup": "Warehouse", "delivery": "Store_B", "weight": 8, "priority": 1},
            {"id": 3, "pickup": "Store_A", "delivery": "Store_C", "weight": 3, "priority": 3},
            {"id": 4, "pickup": "Hub", "delivery": "Store_B", "weight": 6, "priority": 1},
            {"id": 5, "pickup": "Store_B", "delivery": "Warehouse", "weight": 4, "priority": 2}
        ],
        "vehicles": [
            {"id": 1, "capacity": 20, "location": "Warehouse", "speed": 1.0, "cost_per_km": 1.5},
            {"id": 2, "capacity": 15, "location": "Hub", "speed": 1.2, "cost_per_km": 1.2}
        ]
    }
    
    # Initialize optimizer
    optimizer = LogisticsOptimizer("improved_logistics_model")
    
    # Optimize routes
    result = optimizer.optimize_routes(scenario)
    
    # Print results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"\nMetrics:")
    print(f"  Total Time: {result['metrics']['total_time']:.2f}")
    print(f"  Total Distance: {result['metrics']['total_distance']:.2f}")
    print(f"  Total Cost: ${result['metrics']['total_cost']:.2f}")
    print(f"  Delivery Rate: {result['metrics']['delivery_rate']*100:.1f}%")
    
    print(f"\nExecution Plan:")
    for step in result['execution_plan'][:10]:  # Show first 10 steps
        if step['action'] == 'move_to':
            print(f"  Time {step['time']:.1f}: Vehicle {step['vehicle_id']} -> {step['destination']}")
            if step['pickups']:
                print(f"    Picked up packages: {step['pickups']}")
            if step['deliveries']:
                print(f"    Delivered packages: {step['deliveries']}")
    
    print(f"\nVehicle Routes:")
    for vehicle_id, route in result['vehicle_routes'].items():
        print(f"  Vehicle {vehicle_id}: {' -> '.join(route)}")
    
    if result['undelivered_packages']:
        print(f"\nUndelivered Packages: {result['undelivered_packages']}")
    
    return result

# ========================= MAIN EXECUTION =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train the model
        print("Training new model...")
        agent, history = train_model(episodes=2000)
        print("Model saved to 'improved_logistics_model'")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with example
        print("Testing with example scenario...")
        result = example_usage()
        
    else:
        print("Usage:")
        print("  python script.py train    # Train new model")
        print("  python script.py test     # Test with example scenario")
        print("\nFor custom usage, import LogisticsOptimizer class")
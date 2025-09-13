import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import json

# --- Configuration Model was Trained On ---
# This MUST match the configuration used in the training script.
TRAINED_N_PACKAGES = 10
TRAINED_N_VEHICLES = 3

# --- Core Classes (Copied from the training script) ---

@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    pickup_location: tuple[int, int]
    delivery_location: tuple[int, int]
    weight: float
    status: int = 0

@dataclass
class Vehicle:
    """Represents a vehicle with its properties."""
    id: int
    capacity: float
    current_location: tuple[int, int]
    speed: float
    cost_per_km: float
    available_at_time: int = 0

class LogisticsEnvironment:
    """Simulates the logistics environment for the RL agent."""
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.packages = []
        self.vehicles = []
        self.current_time = 0
        self.packages_delivered_count = 0
        self.max_time = 1000
        self.max_dist = grid_size * 2

        # The state size is now fixed based on the training configuration
        self.state_size = (TRAINED_N_VEHICLES * 3) + (TRAINED_N_PACKAGES * 5) + 1

    def load_scenario(self, packages, vehicles):
        """Loads a specific scenario and validates it."""
        if len(packages) > TRAINED_N_PACKAGES:
            raise ValueError(f"Scenario has {len(packages)} packages, but model was trained on {TRAINED_N_PACKAGES}.")
        if len(vehicles) > TRAINED_N_VEHICLES:
            raise ValueError(f"Scenario has {len(vehicles)} vehicles, but model was trained on {TRAINED_N_VEHICLES}.")
            
        self.packages = packages
        self.vehicles = vehicles
        self.total_packages_in_scenario = len(packages)
        return self._get_state()

    def _get_state(self):
        """Constructs the state vector, padding it to the required training size."""
        state = []
        
        # Add vehicle data and pad if necessary
        for i in range(TRAINED_N_VEHICLES):
            if i < len(self.vehicles):
                vehicle = self.vehicles[i]
                state.extend([
                    vehicle.current_location[0] / self.grid_size,
                    vehicle.current_location[1] / self.grid_size,
                    min(vehicle.available_at_time / self.max_time, 1.0)
                ])
            else:
                # Pad with placeholder data for non-existent vehicles
                state.extend([0.0, 0.0, 1.0]) # Location (0,0), Available now

        # Add package data and pad if necessary
        for i in range(TRAINED_N_PACKAGES):
            if i < len(self.packages):
                package = self.packages[i]
                state.extend([
                    package.status / 2.0,
                    package.pickup_location[0] / self.grid_size,
                    package.pickup_location[1] / self.grid_size,
                    package.delivery_location[0] / self.grid_size,
                    package.delivery_location[1] / self.grid_size,
                ])
            else:
                # Pad with placeholder data (status=delivered)
                state.extend([1.0, 0.0, 0.0, 0.0, 0.0])

        state.append(self.current_time / self.max_time)
        return np.array(state, dtype=np.float32)

    def _calculate_distance(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def step(self, action_package_id):
        """Executes a step in the environment."""
        available_vehicle = min(self.vehicles, key=lambda v: v.available_at_time)
        self.current_time = available_vehicle.available_at_time
        
        # Check if the chosen action is for a package that actually exists in the scenario
        if not (0 <= action_package_id < len(self.packages)):
            available_vehicle.available_at_time += 1 # Penalize by waiting
            return self._get_state(), -1.0, self.packages_delivered_count == self.total_packages_in_scenario, {}

        package_to_assign = self.packages[action_package_id]

        if package_to_assign.status != 0 or package_to_assign.weight > available_vehicle.capacity:
            available_vehicle.available_at_time += 1
            return self._get_state(), -1.0, self.packages_delivered_count == self.total_packages_in_scenario, {}
        
        package_to_assign.status = 1
        dist_to_pickup = self._calculate_distance(available_vehicle.current_location, package_to_assign.pickup_location)
        dist_to_delivery = self._calculate_distance(package_to_assign.pickup_location, package_to_assign.delivery_location)
        total_dist = dist_to_pickup + dist_to_delivery
        travel_time = total_dist / available_vehicle.speed

        delivery_completion_time = self.current_time + travel_time
        available_vehicle.current_location = package_to_assign.delivery_location
        available_vehicle.available_at_time = delivery_completion_time
        package_to_assign.status = 2
        self.packages_delivered_count += 1
        
        done = self.packages_delivered_count == self.total_packages_in_scenario or delivery_completion_time >= self.max_time
        return self._get_state(), 1.0, done, {}

class DQNAgent:
    """Inference-only version of the DQNAgent."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0001
        self.q_network = self._build_model()

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

    def load_model(self, file_path):
        if os.path.exists(file_path):
            print(f"Loading model weights from {file_path}")
            self.q_network.load_weights(file_path)
        else:
            raise FileNotFoundError(f"Model file not found at {file_path}. Please ensure the trained model exists.")

    def act(self, state):
        """Act greedily based on the learned policy."""
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

# --- Main Test and Visualization Logic ---

def load_scenario_from_json(filepath):
    """Loads package and vehicle data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    packages = [Package(**p) for p in data['packages']]
    vehicles = [Vehicle(**v) for v in data['vehicles']]
    
    return packages, vehicles

def visualize_solution(env, vehicle_routes, initial_vehicle_locs):
    """Plots the routes taken by vehicles for a given solution."""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    packages_info = {p.id: {'pickup': p.pickup_location, 'delivery': p.delivery_location} for p in env.packages}
    
    for pid, locs in packages_info.items():
        plt.scatter(locs['pickup'][0], locs['pickup'][1], c='green', marker='s', s=100, label='Pickup' if pid == 0 else "")
        plt.scatter(locs['delivery'][0], locs['delivery'][1], c='red', marker='X', s=100, label='Delivery' if pid == 0 else "")
        plt.text(locs['pickup'][0], locs['pickup'][1] + 0.5, f'P{pid}')
        plt.text(locs['delivery'][0], locs['delivery'][1] + 0.5, f'D{pid}')

    colors = ['blue', 'orange', 'purple', 'cyan', 'magenta']
    for v_id, route in vehicle_routes.items():
        if not route or len(route) == 1: continue
        color = colors[v_id % len(colors)]
        plt.scatter(initial_vehicle_locs[v_id][0], initial_vehicle_locs[v_id][1], c=color, marker='*', s=200, label=f'Vehicle {v_id} Start')
        route_x, route_y = zip(*route)
        plt.plot(route_x, route_y, color=color, linestyle='--', marker='o', markersize=5, label=f'Vehicle {v_id} Route')

    plt.title('Optimized Vehicle Routes from Custom Scenario')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.xlim(-1, env.grid_size + 1)
    plt.ylim(-1, env.grid_size + 1)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

if __name__ == "__main__":
    MODEL_FILE = "logistics_model.weights.h5"
    SCENARIO_FILE = "custom_scenario.json"

    try:
        # 1. Load the custom scenario
        packages, vehicles = load_scenario_from_json(SCENARIO_FILE)
        
        # 2. Initialize the environment
        env = LogisticsEnvironment(grid_size=20)
        state = env.load_scenario(packages, vehicles)
        
        # 3. Initialize agent with the *fixed training configuration*
        agent = DQNAgent(env.state_size, TRAINED_N_PACKAGES)
        agent.load_model(MODEL_FILE)
        
        # 4. Run the simulation
        initial_vehicle_locs = {v.id: v.current_location for v in env.vehicles}
        vehicle_routes = {v.id: [v.current_location] for v in env.vehicles}
        done = False
        
        print("\nRunning simulation with the trained model...")
        while not done:
            available_vehicle = min(env.vehicles, key=lambda v: v.available_at_time)
            action = agent.act(state)
            
            # Log the decision only if it's a valid action for the current scenario
            if 0 <= action < len(env.packages) and env.packages[action].status == 0:
                 package = env.packages[action]
                 print(f"Time: {env.current_time:.2f} - Vehicle {available_vehicle.id} chose to handle Package {package.id}")
                 vehicle_routes[available_vehicle.id].append(package.pickup_location)
                 vehicle_routes[available_vehicle.id].append(package.delivery_location)

            state, _, done, _ = env.step(action)
            
        print("\nSimulation finished.")
        print(f"All packages delivered by time: {env.current_time:.2f}")

        # 5. Visualize the final solution
        visualize_solution(env, vehicle_routes, initial_vehicle_locs)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


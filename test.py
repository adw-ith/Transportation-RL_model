import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx
from train import LogisticsEnvironment, DQNAgent, Package, Vehicle
from custom_scenario import LOCATIONS, ROUTES, CUSTOM_PACKAGES, CUSTOM_VEHICLES

# Define the dimensions the model was originally trained with
TRAINING_N_PACKAGES = 10
TRAINING_N_VEHICLES = 3

def run_agent_simulation(env, agent):
    """Runs the simulation using the trained RL agent and returns the results."""
    state = env._get_state()
    total_reward = 0
    vehicle_paths = {v.id: [] for v in env.vehicles if v.capacity > 0}
    num_real_packages = sum(1 for p in env.packages if p.status != 2)

    while True:
        available_vehicle = min(env.vehicles, key=lambda v: v.available_at_time)
        current_time = available_vehicle.available_at_time
        action = agent.act(state)
        
        package = env.packages[action]
        # Record the path taken by the vehicle for this action
        if package.status == 0 and package.weight <= available_vehicle.capacity:
            start_loc = available_vehicle.current_location
            pickup_loc = package.pickup_location
            delivery_loc = package.delivery_location
            vehicle_paths[available_vehicle.id].append((start_loc, pickup_loc))
            vehicle_paths[available_vehicle.id].append((pickup_loc, delivery_loc))

        next_state, reward, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        real_packages_delivered = sum(1 for p in env.packages[:num_real_packages] if p.status == 2)
        if real_packages_delivered == num_real_packages or current_time >= env.max_time:
            break
            
    return {"time": env.current_time, "reward": total_reward, "paths": vehicle_paths}

def run_baseline_simulation(env):
    """Runs a baseline First-Come, First-Served simulation and returns the results."""
    vehicle_paths = {v.id: [] for v in env.vehicles if v.capacity > 0}
    num_real_packages = sum(1 for p in env.packages if p.status != 2)
    
    # Process packages in their defined order
    for package_id in range(num_real_packages):
        available_vehicle = min(env.vehicles, key=lambda v: v.available_at_time)
        
        package = env.packages[package_id]
        if package.status == 0 and package.weight <= available_vehicle.capacity:
            start_loc = available_vehicle.current_location
            pickup_loc = package.pickup_location
            delivery_loc = package.delivery_location
            vehicle_paths[available_vehicle.id].append((start_loc, pickup_loc))
            vehicle_paths[available_vehicle.id].append((pickup_loc, delivery_loc))
        
        env.step(package_id)

    return {"time": env.current_time, "paths": vehicle_paths}

def visualize_routes(locations, routes, agent_paths, baseline_paths, agent_time, baseline_time):
    """Creates a side-by-side plot of the agent's and baseline's routes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Route Visualization: RL Agent vs. Baseline', fontsize=16)

    G = nx.Graph()
    for loc in locations:
        G.add_node(loc)
    for route in routes:
        G.add_edge(route.start_location, route.end_location, weight=route.distance)

    pos = nx.spring_layout(G, seed=42)
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_paths)))

    # Plot Agent Routes
    ax1.set_title(f'RL Agent Routes (Total Time: {agent_time:.2f})')
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    for (vehicle_id, path_list), color in zip(agent_paths.items(), colors):
        for start, end in path_list:
            if start != end:
                ax1.annotate("", xy=pos[end], xycoords='data', xytext=pos[start], textcoords='data',
                             arrowprops=dict(arrowstyle="->", color=color, shrinkA=15, shrinkB=15,
                                             patchA=None, patchB=None, connectionstyle="arc3,rad=0.1"))

    # Plot Baseline Routes
    ax2.set_title(f'Baseline FCFS Routes (Total Time: {baseline_time:.2f})')
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
    for (vehicle_id, path_list), color in zip(baseline_paths.items(), colors):
        for start, end in path_list:
            if start != end:
                ax2.annotate("", xy=pos[end], xycoords='data', xytext=pos[start], textcoords='data',
                             arrowprops=dict(arrowstyle="->", color=color, shrinkA=15, shrinkB=15,
                                             patchA=None, patchB=None, connectionstyle="arc3,rad=0.1"))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def plot_efficiency_comparison(agent_time, baseline_time):
    """Creates a bar chart comparing the total time taken."""
    plt.figure(figsize=(8, 6))
    strategies = ['RL Agent', 'Baseline (FCFS)']
    times = [agent_time, baseline_time]
    
    bars = plt.bar(strategies, times, color=['#4CAF50', '#F44336'])
    plt.ylabel('Total Time to Complete Deliveries')
    plt.title('Efficiency Comparison')
    plt.ylim(0, max(times) * 1.2)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}', ha='center', va='bottom')

def setup_environment(locations, routes, packages, vehicles):
    """Pads a scenario and initializes the environment."""
    num_real_packages = len(packages)
    num_real_vehicles = len(vehicles)
    padded_packages = list(packages)
    padded_vehicles = list(vehicles)

    while len(padded_packages) < TRAINING_N_PACKAGES:
        padded_packages.append(Package(id=len(padded_packages), pickup_location=locations[0], delivery_location=locations[0], weight=0, status=2))
    
    while len(padded_vehicles) < TRAINING_N_VEHICLES:
        padded_vehicles.append(Vehicle(id=len(padded_vehicles), capacity=0, current_location=locations[0], speed=1.0, cost_per_km=1.0, available_at_time=float('inf')))

    env = LogisticsEnvironment(locations, routes, n_packages=TRAINING_N_PACKAGES, n_vehicles=TRAINING_N_VEHICLES)
    env.packages = padded_packages
    env.vehicles = padded_vehicles
    return env


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = "logistics_model.weights.h5"
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model file not found at '{MODEL_WEIGHTS_PATH}'")
    else:
        # --- Run Agent Simulation ---
        print("\n--- Running RL Agent Simulation ---")
        agent_env = setup_environment(LOCATIONS, ROUTES, CUSTOM_PACKAGES, CUSTOM_VEHICLES)
        agent = DQNAgent(agent_env.state_size, agent_env.action_space_size)
        agent.q_network.load_weights(MODEL_WEIGHTS_PATH)
        agent.epsilon = 0.0
        agent_results = run_agent_simulation(agent_env, agent)
        print(f"Agent finished in {agent_results['time']:.2f} time units.")

        # --- Run Baseline Simulation ---
        print("\n--- Running Baseline (FCFS) Simulation ---")
        baseline_env = setup_environment(LOCATIONS, ROUTES, CUSTOM_PACKAGES, CUSTOM_VEHICLES)
        baseline_results = run_baseline_simulation(baseline_env)
        print(f"Baseline finished in {baseline_results['time']:.2f} time units.")

        # --- Generate Visualizations ---
        print("\nGenerating visualizations...")
        visualize_routes(
            LOCATIONS, ROUTES,
            agent_results['paths'], baseline_results['paths'],
            agent_results['time'], baseline_results['time']
        )
        plot_efficiency_comparison(agent_results['time'], baseline_results['time'])
        
        plt.show()


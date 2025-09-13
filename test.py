import numpy as np
import os
import tensorflow as tf # Required for the agent
from train import LogisticsEnvironment, DQNAgent # Import classes from your main script

# Import the custom scenario data
from custom_scenario import LOCATIONS, ROUTES, CUSTOM_PACKAGES, CUSTOM_VEHICLES

def run_simulation(locations, routes, packages, vehicles, model_path):
    """
    Runs a single simulation using a trained agent on a specific scenario.
    """
    print("--- Setting up the test environment ---")
    
    # 1. Initialize the environment with the custom scenario data
    # We pass the number of packages and vehicles based on our custom input
    n_packages = len(packages)
    n_vehicles = len(vehicles)
    env = LogisticsEnvironment(locations, routes, n_packages=n_packages, n_vehicles=n_vehicles)

    # Manually set the environment's packages and vehicles
    env.packages = packages
    env.vehicles = vehicles
    
    print(f"Scenario loaded: {n_packages} packages, {n_vehicles} vehicles.")
    print("-" * 35)

    # 2. Initialize the agent and load the trained weights
    agent = DQNAgent(env.state_size, env.action_space_size)
    print(f"Loading trained model from: {model_path}")
    agent.q_network.load_weights(model_path)
    
    # IMPORTANT: Set epsilon to 0 for evaluation to ensure the agent uses its learned policy
    agent.epsilon = 0.0

    # 3. Run the simulation loop
    state = env._get_state() # Get the initial state from our custom setup
    done = False
    total_reward = 0
    
    print("\n--- Starting Simulation ---")
    while not done:
        # Get the next available vehicle to understand who is making the decision
        available_vehicle = min(env.vehicles, key=lambda v: v.available_at_time)
        current_time = available_vehicle.available_at_time

        # Choose the best action based on the learned policy
        action = agent.act(state)
        
        # Get package details for logging
        package = env.packages[action]

        print(f"Time: {current_time:.2f} | Vehicle {available_vehicle.id} at '{available_vehicle.current_location}' chose to pickup Package {package.id}.")
        print(f"  > Route: '{package.pickup_location}' -> '{package.delivery_location}' (Weight: {package.weight})")

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Update state and total reward
        state = next_state
        total_reward += reward

        # Log the result of the action
        updated_vehicle = env.vehicles[available_vehicle.id]
        print(f"  > Action Complete. Vehicle {updated_vehicle.id} is now at '{updated_vehicle.current_location}', available at time {updated_vehicle.available_at_time:.2f}.\n")


    # 4. Print the final results
    print("--- Simulation Finished ---")
    print(f"All {env.packages_delivered} packages delivered.")
    print(f"Total elapsed time: {env.current_time:.2f}")
    print(f"Total reward accumulated: {total_reward:.2f}")
    print("-" * 27)


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = "logistics_model.weights.h5"

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model file not found at '{MODEL_WEIGHTS_PATH}'")
        print("Please run the training script first to generate the model weights.")
    else:
        run_simulation(
            locations=LOCATIONS,
            routes=ROUTES,
            packages=CUSTOM_PACKAGES,
            vehicles=CUSTOM_VEHICLES,
            model_path=MODEL_WEIGHTS_PATH
        )

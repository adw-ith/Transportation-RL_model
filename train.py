# --- file: train.py ---

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

# You would import your environment from a separate file
# from environment import ImprovedLogisticsEnvironment, generate_random_scenario
from agent import DQNAgent 
import config

# Placeholder for your environment - replace with your actual import
from m import ImprovedLogisticsEnvironment, generate_random_scenario

def main():
    env = ImprovedLogisticsEnvironment(max_locations=20, max_packages=50)
    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    
    # Setup TensorBoard logging
    log_dir = f"logs/{int(time.time())}"
    summary_writer = tf.summary.create_file_writer(log_dir)

    beta = config.PER_BETA_START
    beta_increment = (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES

    print("Starting training...")
    
    for episode in tqdm(range(config.EPISODES)):
        locations, routes, packages, vehicles = generate_random_scenario()
        state = env.load_scenario(locations, routes, packages, vehicles)
        
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while not done and steps < 1000:
            mask = env.get_valid_actions_mask()
            action = agent.act(state, mask)
            next_state, reward, done, _ = env.step(action)
            next_mask = env.get_valid_actions_mask()
            
            agent.remember(state, action, reward, next_state, done, mask, next_mask)
            
            loss = agent.replay(beta)
            total_loss += loss
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        beta = min(1.0, beta + beta_increment)
        
        # Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('total_reward', total_reward, step=episode)
            tf.summary.scalar('avg_loss', total_loss / steps if steps > 0 else 0, step=episode)
            tf.summary.scalar('epsilon', agent.epsilon, step=episode)
            tf.summary.scalar('packages_delivered', env.packages_delivered, step=episode)

        if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

    agent.save(config.SAVE_PATH)
    print(f"\nTraining finished. Model saved to {config.SAVE_PATH}")

if __name__ == '__main__':
    main()
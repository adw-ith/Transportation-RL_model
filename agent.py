# --- file: agent.py ---

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from replay_buffer import PrioritizedReplayBuffer
import config

class DuelingDQNNetwork(keras.Model):
    def __init__(self, action_size):
        super().__init__()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.value_dense = layers.Dense(128, activation='relu')
        self.value_out = layers.Dense(1)
        self.advantage_dense = layers.Dense(128, activation='relu')
        self.advantage_out = layers.Dense(action_size)

    def call(self, state, training=False):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        value = self.value_dense(x)
        value = self.value_out(value)
        advantage = self.advantage_dense(x)
        advantage = self.advantage_out(advantage)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE, alpha=config.PER_ALPHA)
        self.epsilon = config.EPSILON_START
        
        self.q_network = DuelingDQNNetwork(action_size)
        self.target_network = DuelingDQNNetwork(action_size)
        
        self.q_network.build((None, state_size))
        self.target_network.build((None, state_size))
        
        self.optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done, mask, next_mask):
        self.memory.add(state, action, reward, next_state, done, mask, next_mask)

    def act(self, state, valid_actions_mask):
        if np.random.rand() <= self.epsilon:
            valid_indices = np.where(valid_actions_mask == 1)[0]
            return np.random.choice(valid_indices) if len(valid_indices) > 0 else self.action_size - 1
            
        q_values = self.q_network(tf.expand_dims(state, 0), training=False)[0]
        masked_q_values = q_values + (1 - valid_actions_mask) * -1e9
        return tf.argmax(masked_q_values).numpy()

    def replay(self, beta):
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0

        samples, indices, weights = self.memory.sample(config.BATCH_SIZE, beta)
        
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        next_masks = np.array([s[6] for s in samples])

        with tf.GradientTape() as tape:
            # Double DQN logic
            next_q_main = self.q_network(next_states)
            masked_next_q = next_q_main + (1 - next_masks) * -1e9
            next_actions = tf.argmax(masked_next_q, axis=1, output_type=tf.int32)
            
            next_q_target = self.target_network(next_states)
            next_action_indices = tf.stack([tf.range(config.BATCH_SIZE, dtype=tf.int32), next_actions], axis=1)
            next_q_values = tf.gather_nd(next_q_target, next_action_indices)

            targets = rewards + config.GAMMA * next_q_values * (1 - dones)
            
            # Get current Q values for chosen actions
            current_q_values = self.q_network(states)
            action_indices = tf.stack([tf.range(config.BATCH_SIZE, dtype=tf.int32), actions], axis=1)
            current_q = tf.gather_nd(current_q_values, action_indices)
            
            # Calculate TD errors for PER
            td_errors = tf.abs(targets - current_q)
            
            # Compute loss with importance sampling weights
            loss = tf.reduce_mean(weights * tf.square(td_errors)) # Or use Huber Loss

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors.numpy())

        return loss.numpy()

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    def save(self, filepath):
        self.q_network.save_weights(filepath)

    def load(self, filepath):
        self.q_network.load_weights(filepath)
        self.update_target_network()
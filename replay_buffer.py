# --- file: replay_buffer.py ---

import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, mask, next_mask):
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        experience = (state, action, reward, next_state, done, mask, next_mask)

        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None, None, None

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance Sampling Weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + 1e-5  # Add small epsilon for stability

    def __len__(self):
        return self.size
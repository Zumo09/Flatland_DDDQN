import random
from collections import namedtuple, deque

import numpy as np

from components.model import bootstrapped_dueling_q_network, load_dueling_dqn


class BDDDQNPolicy:
    """Dueling Double DQN policy"""

    def __init__(self, state_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode
        self.num_heads = parameters.num_heads

        self.action_size = 4

        # Q-Network
        if evaluation_mode:
            self.local_networks = None
        else:
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size
            self.p_head = parameters.p_head

            self.local_networks = bootstrapped_dueling_q_network(state_size, self.action_size, parameters)

            self.target_networks = bootstrapped_dueling_q_network(state_size, self.action_size, parameters)
            self._soft_update()
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

            self.t_step = 0

    def _random_mask(self):
        return np.random.rand(self.num_heads) < self.p_head

    def act(self, state, head=None):
        state = np.expand_dims(state, 0)
        if head is not None:    # single head selected
            action_values = self.local_networks[head](state)
            action = np.argmax(action_values)
        else:   # ensemble method
            votes = np.sum([net(state) for net in self.local_networks], axis=0)
            return np.argmax(votes)

        return action + 1

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        mask = self._random_mask()
        # Save experience in replay memory
        self.memory.add(state, action - 1, reward, next_state, done, mask)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        states, actions, rewards, next_states, dones, masks = self.memory.sample()

        for head in range(self.num_heads):
            target = self.local_networks[head].predict(states)
            target_next = self.local_networks[head].predict(next_states)
            target_val = self.target_networks[head].predict(next_states)

            train_states = []
            train_targets = []
            size = 0

            for i in range(self.memory.batch_size):
                if masks[i][head]:
                    size += 1
                    # like Q Learning, get maximum Q value at s'
                    # But from target model
                    if dones[i]:
                        target[i][actions[i]] = rewards[i]
                    else:
                        # the key point of Double DQN
                        # selection of action is from model
                        # update is from target model
                        a = np.argmax(target_next[i])
                        target[i][actions[i]] = rewards[i] + self.gamma * (target_val[i][a])
                    train_states.append(states[i])
                    train_targets.append(target[i])

            self.local_networks[head].fit(np.array(train_states), np.array(train_targets),
                                          batch_size=size, epochs=1, verbose=0)

        # Update target network
        self._soft_update(self.tau)

    def _soft_update(self, tau=1.0):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (Keras model): weights will be copied from
            target_model (Keras model): weights will be copied to
            tau (float): interpolation parameter
        """

        for head in range(self.num_heads):
            local_weights = self.local_networks[head].get_weights()
            target_weights = self.target_networks[head].get_weights()

            new_weights = []

            for loc, tar in zip(local_weights, target_weights):
                nw = tau * loc + (1.0 - tau) * tar
                new_weights.append(nw)

            self.target_networks[head].set_weights(new_weights)

    def save(self, filename):
        for head in range(self.num_heads):
            self.local_networks[head].save(filename + '/local/' + str(head))
            self.target_networks[head].save(filename + '/target/' + str(head))

    def load(self, filename):
        self.local_networks = [None] * self.num_heads
        self.target_networks = [None] * self.num_heads
        for head in range(self.num_heads):
            self.local_networks[head] = load_dueling_dqn(filename + '/local/' + str(head))
            self.target_networks[head] = load_dueling_dqn(filename + '/target/' + str(head))


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "mask"])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done, mask):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done, mask)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)
        masks = np.array([e.mask for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones, masks

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

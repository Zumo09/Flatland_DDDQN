import random

import numpy as np

from dueling_double_dqn.model import dueling_dqn, load_dueling_dqn
from dueling_double_dqn.replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99

LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 1000  # how often to update the network
TRAIN_EVERY = 10

EPS_END = 0.005
EPS_DECAY = 0.997


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, action_size=5, observation_shape=None, path=None):
        """Initialize an Agent object.
        """
        self.action_size = action_size

        if path is not None:
            self.qnetwork_local = load_dueling_dqn(path + '/local')
            self.qnetwork_target = load_dueling_dqn(path + '/target')
        elif observation_shape is not None:
            # Q-Network
            self.qnetwork_local = dueling_dqn(observation_shape, action_size, LR)
            self.qnetwork_target = dueling_dqn(observation_shape, action_size)
        else:
            print('[FATAL ERROR]: Unable to create models')

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.eps = 1.
        self._update_target()

    def act(self, state, train=False):
        # Epsilon-greedy action selection
        if train and random.random() < self.eps:
            return random.choice(np.arange(self.action_size))
        else:
            state = np.expand_dims(state, 0)
            action_values = self.qnetwork_local(state)
            return np.argmax(action_values)

    def add_experience(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def step(self):
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # If enough samples are available in memory, get random subset and learn
        if self.t_step % TRAIN_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample()

            target = self.qnetwork_local.predict(states)
            target_next = self.qnetwork_local.predict(next_states)
            target_val = self.qnetwork_target.predict(next_states)

            for i in range(self.memory.batch_size):
                # like Q Learning, get maximum Q value at s'
                # But from target model
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    a = np.argmax(target_next[i])
                    target[i][actions[i]] = rewards[i] + GAMMA * (
                        target_val[i][a])

            # make minibatch which includes target q value and predicted q value
            # and do the model fit!
            self.qnetwork_local.fit(states, target, batch_size=BATCH_SIZE,
                                    epochs=1, verbose=0)

        # Decrease Eps
        if self.t_step == 0:
            self.eps = max(EPS_END, EPS_DECAY * self.eps)
            # Update target
            self._update_target()

    def save(self, filename):
        self.qnetwork_local.save(filename + '/local')
        self.qnetwork_target.save(filename + '/target')

    def _update_target(self):
        self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

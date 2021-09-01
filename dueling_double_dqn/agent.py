import copy
import os
import random

import numpy as np

from dueling_double_dqn.replay_buffer import ReplayBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from dueling_double_dqn.model import dueling_dqn

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99

LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 1000  # how often to update the network
TRAIN_EVERY = 10

EPS_END = 0.005
EPS_DECAY = 0.998


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, observation_shape, action_size, double_dqn=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.action_size = action_size
        self.double_dqn = double_dqn
        # Q-Network
        self.qnetwork_local = dueling_dqn(observation_shape, action_size, LR)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.eps = 1.

    def act(self, state, train=False):
        # Epsilon-greedy action selection
        if train and random.random() < self.eps:
            return random.choice(np.arange(self.action_size))
        else:
            state = tf.expand_dims(state, 0)
            action_values = self.qnetwork_local(state)
            return np.argmax(action_values)

    def add_experience(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def step(self, end_episode=False):
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if end_episode:
            # Decrease Epsilon
            self.eps = max(EPS_END, EPS_DECAY * self.eps)
            self.t_step = 0

        # If enough samples are available in memory, get random subset and learn
        if self.t_step % TRAIN_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample()

            # Double DQN
            q_best_action = np.expand_dims(np.argmax(self.qnetwork_local(next_states), axis=1), -1)
            Q_targets_next = np.take_along_axis(self.qnetwork_target(next_states),
                                                indices=q_best_action, axis=1)

            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

            self.qnetwork_local.fit(states, Q_targets, verbose=0)

        if self.t_step == 0:
            # ------------------- update target network ------------------- #
            self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

    def save(self, filename):
        # TODO: error, it does not work
        self.qnetwork_local.save_weights(filename + "_local.h5")
        self.qnetwork_target.save_weights(filename + "._target.h5")

    def load(self, filename):
        # TODO: error, it does not work
        if os.path.exists(filename + "_local.h5"):
            self.qnetwork_local.load_weights(filename + "_local.h5")
        else:
            print(f'The path "{filename}_local.h5" does not exits, creating a new local network')
        if os.path.exists(filename + "_target.h5"):
            self.qnetwork_target.load_weights(filename + "_target.h5")
        else:
            print(f'The path "{filename}_target.h5" does not exits, creating a new target network')

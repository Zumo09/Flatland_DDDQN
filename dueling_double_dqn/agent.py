import copy
import os
import random

import numpy as np

from dueling_double_dqn.replay_buffer import ReplayBuffer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from marl_tutorial.model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 1000  # how often to update the network

EPS_END = 0.005
EPS_DECAY = 0.998


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, action_size, double_dqn=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.action_size = action_size
        self.double_dqn = double_dqn
        # Q-Network
        self.qnetwork_local = QNetwork(action_size)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

        self.qnetwork_local.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                                    loss=tf.keras.losses.MeanSquaredError())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.eps = 1.

    def save(self, filename):
        self.qnetwork_local.save(filename + "_local.h5", save_format='tf')
        self.qnetwork_target.save(filename + "._target.h5", save_format='tf')

    def load(self, filename):
        if os.path.exists(filename + "_local.h5"):
            self.qnetwork_local = tf.keras.models.load_model(filename + "_local.h5")
        else:
            print(f'The path "{filename}_local.h5" does not exits, creating a new local network')
        if os.path.exists(filename + "_target.h5"):
            self.qnetwork_target = tf.keras.models.load_model(filename + "_target.h5")
        else:
            print(f'The path "{filename}_target.h5" does not exits, creating a new target network')

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
        if len(self.memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample()

            if self.double_dqn:
                # Double DQN
                q_best_action = np.expand_dims(np.argmax(self.qnetwork_local.predict(next_states), axis=1), -1)
                Q_targets_next = np.take_along_axis(self.qnetwork_target.predict(next_states),
                                                    indices=q_best_action, axis=1)
            else:
                # DQN
                Q_targets_next = self.qnetwork_target.predict(next_states)

            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

            self.qnetwork_local.fit(states, Q_targets, verbose=0)

        if self.t_step == 0:
            # ------------------- update target network ------------------- #
            self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())
            print('updated-------------------------------------------')

    def act(self, state, train=False):
        # Epsilon-greedy action selection
        if train and random.random() > self.eps:
            state = tf.expand_dims(state, 0)
            action_values = self.qnetwork_local.predict(state)
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

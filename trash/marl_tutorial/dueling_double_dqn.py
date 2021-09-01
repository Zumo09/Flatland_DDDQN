import copy
import os
import random
from collections import namedtuple, deque, Iterable

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf

from trash.marl_tutorial.model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, double_dqn=True):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
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

    def save(self, filename):
        self.qnetwork_local.save(filename + ".local")
        self.qnetwork_target.save(filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local = tf.keras.models.load_model(filename + ".local")
        else:
            self.qnetwork_local = tf.keras.models.load_model(filename)
        if os.path.exists(filename + ".target"):
            self.qnetwork_target = tf.keras.models.load_model(filename + ".target")
        else:
            self.soft_update(1.0)

    def step(self, state, action, reward, next_state, done, train=True):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                if train:
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = tf.expand_dims(state, 0)
            action_values = self.qnetwork_local.predict(state)
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.double_dqn:
            # Double DQN
            q_best_action = np.expand_dims(np.argmax(self.qnetwork_local.predict(next_states), axis=1), -1)
            Q_targets_next = np.take_along_axis(self.qnetwork_target.predict(next_states),
                                                indices=q_best_action, axis=1)
        else:
            # DQN
            Q_targets_next = self.qnetwork_target.predict(next_states)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        self.qnetwork_local.fit(states, Q_targets, verbose=0)

        # ------------------- update target network ------------------- #
        self.soft_update(TAU)

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (Keras model): weights will be copied from
            target_model (Keras model): weights will be copied to
            tau (float): interpolation parameter
        """
        local_weights = self.qnetwork_local.get_weights()
        target_weights = self.qnetwork_target.get_weights()

        new_weights = []

        for loc, tar in zip(local_weights, target_weights):
            nw = tau * loc + (1.0 - tau) * tar
            new_weights.append(nw)

        self.qnetwork_target.set_weights(new_weights)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.__v_stack_impr([e.state for e in experiences if e is not None])
        actions = self.__v_stack_impr([e.action for e in experiences if e is not None])
        rewards = self.__v_stack_impr([e.reward for e in experiences if e is not None])
        next_states = self.__v_stack_impr([e.next_state for e in experiences if e is not None])
        dones = self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    @staticmethod
    def __v_stack_impr(states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states

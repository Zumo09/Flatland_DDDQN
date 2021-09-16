import pickle
import random
from collections import namedtuple, deque, Iterable

import numpy as np

from components.model import DuelingQNetwork, load_dueling_dqn
from components.policy import Policy


class DDDQNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        # self.state_size = state_size
        # self.action_size = action_size
        # self._action_diff = 5 - self.action_size

        self.action_size = 4

        if not evaluation_mode:
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, self.action_size, parameters)

        if not evaluation_mode:
            self.qnetwork_target = DuelingQNetwork(state_size, self.action_size, parameters)
            self._soft_update()
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

            self.t_step = 0
            self.loss = 0.0

    def act(self, state, eps=0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = np.expand_dims(state, 0)
            action_values = self.qnetwork_local(state)
            action = np.argmax(action_values)
        else:
            action = random.choice(np.arange(self.action_size))

        return action + 1

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action - 1, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
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
                target[i][actions[i]] = rewards[i] + self.gamma * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.qnetwork_local.fit(states, target, batch_size=self.batch_size,
                                epochs=1, verbose=0)

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
        local_weights = self.qnetwork_local.get_weights()
        target_weights = self.qnetwork_target.get_weights()

        new_weights = []

        for loc, tar in zip(local_weights, target_weights):
            nw = tau * loc + (1.0 - tau) * tar
            new_weights.append(nw)

        self.qnetwork_target.set_weights(new_weights)

    def save(self, filename):
        self.qnetwork_local.save(filename + '/local')
        self.qnetwork_target.save(filename + '/target')

    def load(self, filename):
        self.qnetwork_local = load_dueling_dqn(filename + '/local')
        self.qnetwork_target = load_dueling_dqn(filename + '/target')

    # def save_replay_buffer(self, filename):
    #     memory = self.memory.memory
    #     with open(filename, 'wb') as f:
    #         pickle.dump(list(memory)[-500000:], f)
    #
    # def load_replay_buffer(self, filename):
    #     with open(filename, 'rb') as f:
    #         self.memory.memory = pickle.load(f)

    # def test(self):
    #     self.act(np.array([[0] * self.state_size]))
    #     self._learn()


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


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

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.__v_stack([e.state for e in experiences if e is not None])
        actions = self.__v_stack([e.action for e in experiences if e is not None])
        rewards = self.__v_stack([e.reward for e in experiences if e is not None])
        next_states = self.__v_stack([e.next_state for e in experiences if e is not None])
        dones = self.__v_stack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    @staticmethod
    def __v_stack_impr(states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states

    @staticmethod
    def __v_stack(states):
        if isinstance(states[0], Iterable):
            sub_dim = len(states[0][0])
            np_states = np.reshape(np.array(states), (len(states), sub_dim))
        else:
            np_states = np.array(states)
        return np_states

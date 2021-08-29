from collections import namedtuple, deque, Iterable

import numpy as np
import random


def dict_to_list(done):
    ret = []
    a = 0
    while True:
        if a in done.keys():
            ret.append(done[a])
            a += 1
        else:
            break
    return ret


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        :param buffer_size: (int) maximum size of buffer
        :param batch_size: (int) size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "observations", "actions", "rewards", "next_obs", "dones"])

    def add(self, state, obs, actions, rewards, next_obs, dones):
        """Add a new experience to memory."""
        e = self.experience(state, obs, actions, rewards, next_obs, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e.state for e in experiences if e is not None])
        obs = np.array([e.observations for e in experiences if e is not None])
        actions = np.array([e.actions for e in experiences if e is not None])
        rewards = np.array([e.rewards for e in experiences if e is not None])
        next_obs = np.array([e.next_obs for e in experiences if e is not None])
        dones = np.array([dict_to_list(e.dones) for e in experiences if e is not None]).astype(np.uint8)

        return states, obs, actions, rewards, next_obs, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

import copy

import numpy as np
import tensorflow as tf

from trash.qmix_prova_veloce import ReplayBuffer
from trash.qmix_prova_veloce import DuelingAgent
from utils.observation_utils import normalize_observation

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network


class AgentsController:
    def __init__(self, n_agents, tree_depth=2, action_size=5, rnn_hidden_dim=64, double_dqn=True, obs_last_action=True):
        self.double_dqn = double_dqn
        self.obs_last_action = obs_last_action
        self.n_agents = n_agents
        self.tree_depth = tree_depth
        self.action_size = action_size
        self.rng = np.random.default_rng()

        self.action_dict = dict()
        self.update_values = [False] * n_agents
        self.action_prob = np.zeros(action_size)

        self.agent_obs = [None] * n_agents
        self.agent_obs_buffer = [None] * n_agents
        self.agent_next_obs = [None] * n_agents
        self.agent_obs_buffer = [None] * n_agents
        self.agent_action_buffer = [2] * n_agents
        self.cummulated_reward = np.zeros(n_agents)
        self.action_values = np.zeros((self.n_agents, self.action_size))

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # self.agent = RnnAgent(rnn_hidden_dim=rnn_hidden_dim, n_actions=action_size)
        self.agent = DuelingAgent(action_size=action_size)
        self.agent_target = copy.deepcopy(self.agent)

    def act(self, obs, info, eps=0.0, training=False):
        """
        Get the actions for all the agents, given the observations
        """

        # Build agent specific observations
        self.build_inputs(obs)

        self.action_values = np.zeros((self.n_agents, self.action_size))

        for a in range(self.n_agents):
            if info['action_required'][a]:
                # If an action is require, we want to store the obs a that step as well as the action
                self.update_values[a] = True

                state = tf.expand_dims(self.agent_obs[a], 0)
                self.action_values[a] = self.agent.predict(state)

                if not training or self.rng.random() > eps:
                    action = np.argmax(self.action_values[a])
                else:
                    action = self.rng.integers(0, self.action_size)

                self.action_prob[action] += 1
            else:
                self.update_values[a] = False
                action = 0
            self.action_dict.update({a: action})

        return self.action_dict

    def build_inputs(self, obs):
        for a in range(self.n_agents):
            if obs[a]:
                agent_obs = normalize_observation(obs[a], self.tree_depth, observation_radius=10)

                if self.obs_last_action:
                    agent_la = np.zeros(self.action_size)
                    agent_la[self.agent_action_buffer[a]] = 1
                    agent_obs = np.append(agent_obs, agent_la)

                # Add an extra info to the input to differentiate the agents
                agent_id = np.zeros(self.n_agents)
                agent_id[a] = 1
                self.agent_obs[a] = np.append(agent_obs, agent_id)

                self.agent_obs_buffer[a] = self.agent_obs[a].copy()

    def step(self, all_rewards, done, train=True):
        score = 0
        # Update replay buffer and train agent
        for a in range(self.n_agents):
            score += all_rewards[a]
            # Only update the values when we are done or when an action was taken and thus relevant information
            # is present
            if self.update_values[a] or done[a]:
                self.agent_step(self.agent_obs_buffer[a], self.agent_action_buffer[a], all_rewards[a],
                                self.agent_obs[a], done[a], train)

                self.cummulated_reward[a] = 0.

                self.agent_obs_buffer[a] = self.agent_obs[a].copy()
                self.agent_action_buffer[a] = self.action_dict[a]

        return score / self.n_agents
            
    def agent_step(self, state, action, reward, next_state, done, train):
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

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences : tuple of (s, a, r, s', done) tuples
                    gamma : discount factor
                """
        states, actions, rewards, next_states, dones = experiences

        if self.double_dqn:
            # Double DQN
            q_best_action = np.expand_dims(np.argmax(self.agent.predict(next_states), axis=1), -1)
            Q_targets_next = np.take_along_axis(self.agent_target.predict(next_states),
                                                indices=q_best_action, axis=1)
        else:
            # DQN
            Q_targets_next = self.agent_target.predict(next_states)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        self.agent.fit(states, Q_targets, verbose=0)

        # ------------------- update target network ------------------- #
        self.soft_update(TAU)

    def save(self, filename):
        self.agent.save(filename)

    def load(self, filename):
        self.agent = tf.keras.models.load_model(filename)

    def action_prob(self, reset=False):
        if reset:
            self.action_prob = np.zeros(self.action_size)
        else:
            return self.action_prob / np.sum(self.action_prob)

    def soft_update(self, tau=1.0):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (Keras model): weights will be copied from
            target_model (Keras model): weights will be copied to
            tau (float): interpolation parameter
        """
        local_weights = self.agent.get_weights()
        target_weights = self.agent_target.get_weights()

        new_weights = []

        for loc, tar in zip(local_weights, target_weights):
            nw = tau * loc + (1.0 - tau) * tar
            new_weights.append(nw)

        self.agent_target.set_weights(new_weights)

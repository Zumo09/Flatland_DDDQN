import copy

import numpy as np
import tensorflow as tf

from qmix_flatland_orribile.model import QMixer
from qmix_flatland_orribile.replay_buffer import ReplayBuffer
from utils import normalize_observation

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 4  # minibatch size
# BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
LR = 0.5e-4  # learning rate 0.5e-4 works
TRAIN_EVERY = 2  # how often to fit the local network
UPDATE_EVERY = 10  # how often to update the target network


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
        self.action_prob = np.ones(action_size)

        self.agent_obs = [None] * n_agents
        self.agent_obs_buffer = [None] * n_agents
        self.agent_next_obs = [None] * n_agents
        self.agent_action_buffer = [2] * n_agents

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step
        self.t_step = 0

        self.agent = QMixer(n_agents=n_agents, rnn_hidden_dim=rnn_hidden_dim, n_actions=action_size)

        self.agent.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LR),
                           loss=tf.keras.losses.MeanSquaredError())

        self.agent_target = copy.deepcopy(self.agent)

        print('Controller Ready!')

    def act(self, obs, info, eps=0.0):
        """
        Get the actions for all the agents, given the observations
        """

        # Build agent specific observations
        self._build_inputs(obs)

        for a in range(self.n_agents):
            if info['action_required'][a]:
                self.update_values[a] = True

                if self.rng.random() > eps:
                    state = tf.expand_dims(self.agent_obs[a], 0)
                    action_values = self.agent(state)
                    action = np.argmax(action_values)
                else:
                    action = self.rng.integers(0, self.action_size)

                self.action_prob[action] += 1
            else:
                self.update_values[a] = False
                action = 0
            self.action_dict.update({a: action})

        return self.action_dict

    def _build_inputs(self, obs):
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
                agent_obs = np.append(agent_obs, agent_id)

                self.agent_obs[a] = agent_obs

                self.agent_obs_buffer[a] = self.agent_obs[a].copy()

    def step(self, state, all_rewards, done):
        self.t_step += 1

        score = 0
        # Update replay buffer and train agent
        for a in range(self.n_agents):
            score += all_rewards[a]
            # Only update the values when we are done or when an action was taken and thus relevant information
            # is present
            if self.update_values[a] or done[a]:
                self.agent_obs_buffer[a] = self.agent_obs[a].copy()
                self.agent_action_buffer[a] = self.action_dict[a]

        score /= self.n_agents
        # Save experience in replay memory
        self.memory.add(state, self.agent_obs_buffer, self.agent_action_buffer,
                        # all_rewards,
                        score,
                        self.agent_obs, done['__all__'])

        # Train every TRAIN_EVERY time steps.
        if self.t_step % TRAIN_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self._train(experiences, GAMMA)

        # Update target every UPDATE_EVERY time steps.
        if self.t_step % UPDATE_EVERY == 0:
            self.agent_target.set_weights(self.agent.get_weights())

        return score

    def _train(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences : tuple of (s, o, a, r, o', done) tuples
                    gamma : discount factor
                """
        states, obs, actions, rewards, next_obs, dones = experiences

        for i in range(BATCH_SIZE):
            self.agent.update_state(states[i])
            self.agent_target.update_state(states[i])

            Q_target_next = self.agent_target(next_obs[i], training=True)

            # Calculate 1-step Q-Learning targets
            Q_target = rewards[i] + gamma * (1 - dones[i]) * Q_target_next

            print(Q_target_next, Q_target)

            self.agent.fit(np.expand_dims(obs[i], 0), np.expand_dims(Q_target, 0), verbose=0)

    def action_probability(self, reset=False):
        if reset:
            self.action_prob = np.ones(self.action_size)
        else:
            return self.action_prob / np.sum(self.action_prob)

    def save(self, filename):
        self.agent.save(filename)

    def load(self, filename):
        self.agent = tf.keras.models.load_model(filename)

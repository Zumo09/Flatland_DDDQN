import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
# from flatland.envs.malfunction_generators import MalfunctionParameters, malfunction_from_params
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from dueling_double_dqn.agent import Agent
from dueling_double_dqn.observations import CustomObservation


class FlatlandController:
    def __init__(self, grid_shape, n_agents, seed=1, load_from=None):

        self.rng = np.random.default_rng(seed=seed)

        # Parameters for the Environment
        x_dim, y_dim = grid_shape
        self.n_agents = n_agents

        # The action space of flatland is 5 discrete actions
        self.action_size = 5

        # And the max number of steps we want to take per episode
        self.max_steps = int(4 * 2 * (20 + x_dim + y_dim))

        # Use a the malfunction generator to break agents from time to time
        # stochastic_data = MalfunctionParameters(malfunction_rate=1. / 10000,  # Rate of malfunction occurrence
        #                                        min_duration=15,  # Minimal duration of malfunction
        #                                        max_duration=50  # Max duration of malfunction
        #                                        )

        # Custom observation builder
        observations = CustomObservation()

        # Different agent types (trains) with different speeds.
        speed_ration_map = {1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

        self.env = RailEnv(width=x_dim, height=y_dim,
                           rail_generator=sparse_rail_generator(max_num_cities=3,
                                                                # Number of cities in map (where train stations are)
                                                                seed=seed,  # Random seed
                                                                grid_mode=False,
                                                                max_rails_between_cities=2,
                                                                max_rails_in_city=3),
                           schedule_generator=sparse_schedule_generator(speed_ration_map),
                           number_of_agents=n_agents,
                           # TODO: add malfunctions
                           # malfunction_generator=malfunction_from_params(stochastic_data),
                           obs_builder_object=observations)

        # Reset env
        self.env.reset(True, True)
        # After training we want to render the results so we also load a renderer
        self.env_renderer = RenderTool(self.env)

        # Now we load a Double dueling DQN agent
        self.agent = Agent(self.action_size)

        if load_from:
            self.agent.load(load_from)

    def run_episode(self, train=True, render=False):
        # Some variables to keep track of the progress
        action_dict = dict()
        agent_action_buffer = [2] * self.n_agents
        update_values = [False] * self.n_agents

        # Reset environment
        agent_obs, info = self.env.reset(True, True)
        agent_obs_buffer = agent_obs.copy()

        if render:
            self.env_renderer.reset()

        # Reset score
        score = 0
        step = 0

        # Run episode
        while step < self.max_steps:
            # Action
            for a in range(self.n_agents):
                if info['action_required'][a]:
                    # If an action is require, we want to store the obs a that step as well as the action
                    update_values[a] = True

                    action = self.agent.act(agent_obs[a], train=train)
                else:
                    update_values[a] = False
                    action = 0
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, info = self.env.step(action_dict)
            # Update replay buffer and train agent
            for a in range(self.n_agents):
                # Only update the values when we are done or when an action was taken and thus relevant information
                # is present
                if train and (update_values[a] or done[a]):
                    self.agent.add_experience(agent_obs_buffer[a], agent_action_buffer[a],
                                              all_rewards[a], agent_obs[a], done[a])

                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]

                if next_obs[a] is not None:
                    agent_obs[a] = next_obs[a]

                score += all_rewards[a] / self.n_agents

            if train:
                self.agent.step()

            if render:
                self.env_renderer.render_env()

            step += 1
            # Copy observation
            if done['__all__']:
                break

        self.agent.step(end_episode=True)

        tasks_finished = 0
        for current_agent in self.env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1

        return score, tasks_finished

    def save(self, path):
        self.agent.save(path)

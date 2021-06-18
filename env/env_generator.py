import numpy as np

from flatland.envs.observations import TreeObsForRailEnv, LocalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool


def transform_tree_obs(next_obs):
    return next_obs


class FlatlandEnv:
    def __init__(self, n_agents, render=False, observation=None, seed=1, observation_transformer=None):
        self.transformer = transform_tree_obs
        if observation is None:
            observation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
        elif observation_transformer is not None:
            self.transformer = observation_transformer
        else:
            print('Observation Transformer not provided, using the default for TreeObsForRailEnv')

        self.env = RailEnv(width=20, height=20,
                           rail_generator=complex_rail_generator(nr_start_goal=10,
                                                                 nr_extra=2,
                                                                 min_dist=8,
                                                                 max_dist=99999,
                                                                 seed=seed),
                           schedule_generator=complex_schedule_generator(),
                           number_of_agents=n_agents,
                           obs_builder_object=observation)

        self.renderer = None
        if render:
            self.renderer = RenderTool(self.env)

        self.info = {
            'n_actions': 5,
            'n_agents': n_agents,
        }

    def reset(self):
        obs, info = self.env.reset()

        obs = self.transformer(obs)
        state = self._get_state()

        return obs, state

    def step(self, actions):
        action_dict = {}
        for a, action in enumerate(actions):
            action_dict[a] = action

        next_obs, all_rewards, done, _ = self.env.step(action_dict)

        if self.renderer is not None:
            self.renderer.render_env(show=True, show_observations=True, show_predictions=False)

        obs = self.transformer(next_obs)
        state = self._get_state()
        reward = sum([r for r in all_rewards.values()])
        terminated = done['__all__']

        return obs, state, reward, terminated

    def get_env_info(self):
        return self.info

    def _get_state(self):
        return self.env

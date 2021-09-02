import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool
from dueling_double_dqn.env.tree_obs_utils import normalize_observation

depth = 3


def tree_transformer(observations):
    new_obs = {}
    for a in observations.keys():
        if observations[a] is None:
            new_obs[a] = None
        else:
            new_obs[a] = normalize_observation(observations[a], tree_depth=depth)
    return new_obs


class FlatlandEnv:
    def __init__(self, n_agents, episode_limit, render=False, seed=1):
        self.transformer = tree_transformer
        observation = TreeObsForRailEnv(max_depth=depth, predictor=ShortestPathPredictorForRailEnv())

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

        self.episode_limit = episode_limit
        self.episode_t = 0

        self.not_finished_penalty = 100

        self.info = {
            'n_actions': 5,
            'n_agents': n_agents,
            'episode_limit': episode_limit
        }

        self.observations = None
        self.state = None

    def reset(self):
        obs, info = self.env.reset()

        obs = self.transformer(obs)
        state = self.get_state()

        self.episode_t = 0

        return obs, state

    def step(self, actions):
        info = {}
        action_dict = {}
        for a, action in enumerate(actions):
            action_dict[a] = action

        next_obs, all_rewards, done, _ = self.env.step(action_dict)

        if self.renderer is not None:
            self.renderer.render_env(show=True, show_observations=True, show_predictions=False)

        self.observations = self.transformer(next_obs)
        self.state = self._get_state_from_env()
        reward = sum([r for r in all_rewards.values()])
        terminated = done['__all__']

        if self.episode_t < self.episode_limit:
            self.episode_t += 1
        else:
            info['episode_limit'] = True
            terminated = True
            reward += self.not_finished_penalty * (np.count_nonzero(done.values()) - 1)

        return reward, terminated, info

    def get_env_info(self):
        return self.info

    def get_state(self):
        return self.state

    def get_obs(self):
        return self.observations

    def _get_state_from_env(self):
        return self.env.agent_positions

    def save_replay(self):
        pass

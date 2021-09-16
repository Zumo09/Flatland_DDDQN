import numpy as np
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from components.custom_normalized_tree import TreeObsNormalized
from components.observation_2 import CustomObservation


def train_agent():
    # Setup the environments
    n_agents = 1
    x_dim = 35
    y_dim = 35
    n_cities = 2
    max_rails_between_cities = 2
    max_rails_in_city = 3
    seed = 0

    # Observation parameters
    observation_tree_depth = 3
    observation_max_path_depth = 10

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    # observation = CustomObservation(max_depth=observation_tree_depth, predictor=predictor)
    observation = TreeObsNormalized(max_depth=observation_tree_depth, predictor=predictor)

    # Only fast trains in Round 1
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=1 / 20.,
        min_duration=20,
        max_duration=50
    )

    train_env = RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(speed_profiles),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(malfunction_parameters),
        obs_builder_object=observation,
        random_seed=seed
    )

    # Reset environment
    obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)
    prev = obs[0].shape
    action_dict = {}
    # Run episode
    for step in range(100):
        for agent in train_env.get_agent_handles():
            if info['action_required'][agent]:
                action = np.random.choice(range(6))
            else:
                action = 0
            action_dict.update({agent: action})

        # Environment step
        next_obs, all_rewards, done, info = train_env.step(action_dict)

        for o in next_obs:
            # print('agent', o, 'step', step)
            print(next_obs[o].shape)
            # print(next_obs[o])

        if prev != next_obs[o].shape:
            print('male')
            break

        prev = next_obs[o].shape


if __name__ == "__main__":
    train_agent()

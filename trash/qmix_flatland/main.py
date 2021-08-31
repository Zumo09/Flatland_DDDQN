import getopt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

from trash.qmix_flatland import AgentsController

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.agent_utils import RailAgentStatus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# make sure the root path is in system path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


def main(argv):
    # We set the number of episodes we would like to train on
    n_trials = 1000
    # n_trials = 15000

    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print("Option error")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    np.random.seed(1)

    # Parameters for the Environment
    x_dim = 35
    y_dim = 35
    n_agents = 3
    tree_depth = 2

    # Use a the malfunction generator to break agents from time to time
    # stochastic_data = MalfunctionParameters(malfunction_rate=1. / 10000,  # Rate of malfunction occurrence
    #                                         min_duration=15,  # Minimal duration of malfunction
    #                                         max_duration=50  # Max duration of malfunction
    #                                         )

    # Custom observation builder
    TreeObservation = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv(30))

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  # TODO: aggiungere malfunzionamenti
                  # malfunction_generator=stochastic_data,
                  obs_builder_object=TreeObservation)

    # Reset env
    env.reset(True, True)
    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG")
    # Given the depth of the tree observation and the number of features per node we get the following state_size

    """ probably not useful """
    # num_features_per_node = env.obs_builder.observation_dim
    # tree_depth = 2
    # nr_nodes = 0
    # for i in range(tree_depth + 1):
    #     nr_nodes += np.power(4, i)
    # state_size = num_features_per_node * nr_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # And the max number of steps we want to take per episode
    max_steps = int(4 * 2 * (20 + env.height + env.width))

    # Define training parameters
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []

    # Now we load agent
    agent = AgentsController(n_agents)

    print(f'Start training for {n_trials} Episodes')
    for trials in range(1, n_trials + 1):
        # Reset environment
        obs, info = env.reset(True, True)
        env_renderer.reset()

        # Reset score and done
        score = 0

        # Run episode
        while True:
            # Action
            action_dict = agent.act(obs, info=info, eps=eps)
            # Environment step
            obs, all_rewards, done, info = env.step(action_dict)

            state = np.ones((1, 128))
            # Agent Step
            score += agent.step(state, all_rewards, done)

            if done['__all__']:
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        tasks_finished = 0
        for current_agent in env.agents:
            if current_agent.status == RailAgentStatus.DONE_REMOVED:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%'
            '\tEpsilon: {:.2f}\t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, agent.action_probability()), end=" ")

        if trials % 100 == 0:
            print()
            agent.save('./Nets/navigator_checkpoint' + str(trials) + '.h5')
            agent.action_probability(reset=True)

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

import getopt
import sys
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from dueling_double_dqn.controller import FlatlandController

TEST_EVERY = 100


def main(argv):
    n_trials = 1000
    n_agents = 2
    grid_shape = (35, 35)
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    path = None
    # path = './Nets/navigator_checkpoint' + str(1000)

    controller = FlatlandController(grid_shape=grid_shape,
                                    n_agents=n_agents,
                                    load_from=path)

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []

    print(f'Training for {n_trials} Episodes')
    for trials in range(1, n_trials + 1):
        score, tasks_finished = controller.run_episode(train=True, render=False)

        # Collection information about training
        done_window.append(100 * tasks_finished / n_agents)
        scores_window.append(score)  # save most recent score
        scores.append(np.mean(scores_window))

        print(f'\rEpisode {trials:5d}\tAverage Score: {np.mean(scores_window):5.3f}'
              f'\tDones: {np.mean(done_window):3.2f}%\tEpsilon: {controller.agent.eps:.2f}', end=" ")

        if trials % TEST_EVERY == 0:
            controller.save('./Nets/checkpoint' + str(trials))

            score, tasks_finished = controller.run_episode(train=False, render=False)

            print(f'\t|\tTest {trials // TEST_EVERY:3d}\tScore: {score:5.3f}'
                  f'\tDones: {100 * tasks_finished / n_agents:3.2f}%\t')

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

import getopt
import sys
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from dueling_double_dqn.trainer import FlatlandTrainer

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

    trainer = FlatlandTrainer(grid_shape=grid_shape,
                              n_agents=n_agents)

    # And the max number of steps we want to take per episode
    max_steps = int(4 * 2 * (20 + grid_shape[0] + grid_shape[1]))

    # And some variables to keep track of the progress
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []

    print(f'Training for {n_trials} Episodes')
    for trials in range(1, n_trials + 1):
        score, tasks_finished = trainer.run_episode(train=True, render=False)

        # Collection information about training

        done_window.append(tasks_finished / max(1, n_agents))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(f'\rEpisode {trials}\t Average Score: {np.mean(scores_window):.3f}'
              f'\tDones: {100 * np.mean(done_window):.2f}%\tEpsilon: {trainer.eps:.2f} '
              f'\t Action Probabilities: \t {trainer.action_probabilities()} ', end=" ")

        if trials % TEST_EVERY == 0:
            print()
            trainer.save('./Nets/navigator_checkpoint' + str(trials) + '_sec.pth')
            score, tasks_finished = trainer.run_episode(train=False, render=True)
            trainer.action_probabilities(reset=True)
            print(f'Test\t Score: {score:.3f} \t Dones: {tasks_finished / max(1, n_agents):.2f} '
                  f'\t Action Probabilities: \t {trainer.action_probabilities()}')
            trainer.action_probabilities(reset=True)

        # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

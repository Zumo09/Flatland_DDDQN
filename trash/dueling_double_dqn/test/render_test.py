from trash.dueling_double_dqn.env import FlatlandController


def main():
    n_agents = 2
    grid_shape = (35, 35)

    controller = FlatlandController(grid_shape=grid_shape,
                                    n_agents=n_agents,
                                    # load_from='./Nets/checkpoint1000'
                                    )

    score, tasks_finished = controller.run_episode(train=False, render=True)

    print(score, tasks_finished)


if __name__ == '__main__':
    main()

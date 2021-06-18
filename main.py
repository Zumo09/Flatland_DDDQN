from env.env_generator import FlatlandEnv
import numpy as np


def main():
    env = FlatlandEnv(n_agents=4, render=True)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        obs, state = env.reset()
        terminated = False
        episode_reward = 0
        t = 0

        while not terminated and t < 10:
            t += 1
            actions = []
            for agent_id in range(n_agents):
                action = np.random.choice(n_actions)
                actions.append(action)

            obs, state, reward, terminated = env.step(actions)
            episode_reward += reward
            print(t)

        if not terminated:
            episode_reward += 100

        print("Total reward in episode {} = {}".format(e, episode_reward))


if __name__ == '__main__':
    main()

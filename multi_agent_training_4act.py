import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from components.dddqn_policy import DDDQNPolicy
from components.env_config import get_env_config
from components.observations import CustomObservation
from utils.timer import Timer


def create_rail_env(env_params):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Observation parameters
    observation_tree_depth = env_params.observation_tree_depth
    observation_max_path_depth = env_params.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    observation = CustomObservation(max_depth=observation_tree_depth, predictor=predictor)

    # Only fast trains in Round 1
    speed_profiles = {
        1.: 1.0,  # Fast passenger train
        1. / 2.: 0.0,  # Fast freight train
        1. / 3.: 0.0,  # Slow commuter train
        1. / 4.: 0.0  # Slow freight train
    }

    # Break agents from time to time
    malfunction_parameters = MalfunctionParameters(
        malfunction_rate=env_params.malfunction_rate,
        min_duration=20,
        max_duration=50
    )

    env = RailEnv(
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

    env.invalid_action_penalty = env_params.invalid_action_penalty
    env.step_penalty = env_params.step_penalty
    env.global_reward = env_params.global_reward
    env.stop_penalty = env_params.stop_penalty  # penalty for stopping a moving agent
    env.start_penalty = env_params.start_penalty  # penalty for starting a stopped agent

    return env


def train_agent(config):
    training_id = config.training_id

    # Environment parameters
    n_agents = config.n_agents
    x_dim = config.x_dim
    y_dim = config.y_dim
    seed = config.seed

    # Training parameters
    eps_start = config.eps_start
    eps_end = config.eps_end
    eps_decay = config.eps_decay
    n_episodes = config.n_episodes
    checkpoint_interval = config.checkpoint_interval
    n_eval_episodes = config.n_evaluation_episodes

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Setup the environments
    train_env = create_rail_env(config)
    obs, _ = train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(config)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    state_size = obs[0].shape[0]
    print(f'state size : {state_size}')

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
    max_steps = train_env._max_episode_steps

    action_count = [0] * 5
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smooth_eval_norm_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    train_score_history = []
    train_score_avg_history = []
    test_scores_history = []

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, config)

    training_timer = Timer()
    training_timer.start()

    print(f"\n🚉 Training {train_env.get_num_agents()} trains on {x_dim}x{y_dim} grid for {n_episodes} episodes, "
          f"evaluating on {n_eval_episodes} episodes every {checkpoint_interval} episodes. "
          f"Training id '{training_id}'.\n")

    # main_timer = Timer()
    # step_timer = Timer()
    # learn_timer = Timer()
    # inference_timer = Timer()

    for episode_idx in range(n_episodes + 1):
        # main_timer.start()
        # Reset environment
        obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True)

        score = 0

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent] is not None:
                agent_obs[agent] = obs[agent]
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            # inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps_start)
                    if action == 0:
                        print('WTF?!')
                    action_count[action] += 1
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            # inference_timer.end()

            # Environment step
            # step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            # step_timer.end()

            # learn_timer.start()
            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                                agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent] is not None:
                    agent_obs[agent] = next_obs[agent]

                score += all_rewards[agent]
            # learn_timer.end()

            # nb_steps = step

            if done['__all__']:
                break

        # main_timer.end()

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [0] * 5

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        train_score_history.append(normalized_score)
        train_score_avg_history.append(smoothed_normalized_score)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            policy.save('./checkpoints/' + training_id + '/' + str(episode_idx))

        # time = main_timer.get()
        # inf_time = 100 * inference_timer.get() / time
        # step_time = 100 * step_timer.get() / time
        # learn_time = 100 * learn_timer.get() / time
        print(
            f'\r🚂 Episode {episode_idx:4d}\t🏆 Score: {normalized_score:.3f} (Avg: {smoothed_normalized_score:.3f})'
            f'\t💯 Done: {100 * completion:6.2f}% (Avg: {100 * smoothed_completion:6.2f}%)'
            f'\t🎲 Epsilon: {eps_start:.3f} \t🔀 Action Probs: {format_action_prob(action_probs)}'
            # f'\t⏱ Time: {time:6.2f} (inference: {inf_time:4.2f}%, step: {step_time:4.2f}%, learn: {learn_time:4.2f}%)'
            , end="")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0:
            scores, completions, nb_steps_eval = eval_policy(eval_env, policy, config)

            test_scores_history.append(scores)

            mean_score = np.mean(scores)
            mean_completions = np.mean(completions)

            smoothing = 0.9
            smooth_eval_norm_score = smooth_eval_norm_score * smoothing + mean_score * (1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * smoothing + mean_completions * (1.0 - smoothing)

            wandb.log({
                'test_mean_score': mean_score,
                'test_smoothed_score': smooth_eval_norm_score,
                'test_mean_completion': mean_completions,
                'test_smoothed_completion': smoothed_eval_completion
            }, commit=False)

            # main_timer.reset()
            # step_timer.reset()
            # learn_timer.reset()
            # inference_timer.reset()

        # Save log to WandB
        wandb.log({
            'training_normalized_score': normalized_score,
            'training_smoothed_normalized_score': smoothed_normalized_score,
            'training_completion': np.mean(completion),
            'training_smoothed_completion': np.mean(smoothed_completion),
            # 'epsilon': eps_start,
            # 'episode': episode_idx
        })

    print('Total time: ', training_timer.get_current())


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(env, policy, train_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if obs[agent] is not None:
                    agent_obs[agent] = obs[agent]

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print(f"\t✅ Eval: score {np.mean(scores):.3f} done {np.mean(completions) * 100.0:6.2f}%")

    return scores, completions, nb_steps


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(base_dir))

    parser = ArgumentParser()
    parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=2000, type=int)
    parser.add_argument("-t", "--env_config", help="training config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
    parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
    parser.add_argument("--eps_decay", help="exploration decay", default=0.998, type=float)

    parser.add_argument("--invalid_action_penalty", help="reward invalid action penalty", default=-1.0, type=float)
    parser.add_argument("--step_penalty", help="reward step penalty", default=-1.5, type=float)
    parser.add_argument("--global_reward", help="global reward", default=5.0, type=float)
    parser.add_argument("--stop_penalty", help="penalty for stopping a moving agent", default=0.0, type=float)
    parser.add_argument("--start_penalty", help="penalty for starting a stopped agent", default=0.0, type=float)

    parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e5), type=int)
    parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor", default=0.98, type=float)
    parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate", help="learning rate", default=0.1e-3, type=float)
    parser.add_argument("--hidden_size_1", help="hidden size 1st layer", default=256, type=int)
    parser.add_argument("--hidden_size_2", help="hidden size 2nd layer", default=64, type=int)
    parser.add_argument("--hidden_size_3", help="hidden size 3rd layer", default=32, type=int)
    parser.add_argument("--update_every", help="how often to update the network", default=16, type=int)
    parser.add_argument("--num_heads", help="number of heads of the bootstrapped q network", default=4, type=int)
    parser.add_argument("--p_head", help="probability of a head to be included in the mask", default=0.5, type=float)

    training_params = parser.parse_args()

    train_env_params = get_env_config(training_params.env_config)

    # Unique ID for this training
    now = datetime.now()
    train_id = now.strftime('%y%m%d%H%M%S')

    configuration = vars(training_params)
    configuration.update({'training_id': train_id})
    configuration.update(train_env_params)

    wandb.login(key='0f20b9f069a2312cc2b8e92e6f75310697d5fdfc')

    wandb.init(project='frl-eps-vs-boot', config=configuration)

    print('\nWeigh and Biases Configuration\n')
    for k, v in wandb.config.items():
        print(f'{k.rjust(30)}\t:\t{v}')
    print('\n')

    train_agent(wandb.config)

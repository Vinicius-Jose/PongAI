import gym
import random
import os
import pickle
import neat
import numpy as np


def initialize_atari_env_game(render_mode="rgb_array"):
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)
    return env
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    env.unwrapped.get_action_meanings()


def eval_network(net: neat.nn.FeedForwardNetwork, net_input):
    # net_input_array = np.zeros(500)
    # net_input_array[net_input] = 1
    net_input = [item.any() for item in net_input]
    val = net.activate(net_input)
    return np.argmax(val)


def train_ai(env: gym.Env, genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env.metadata["render_fps"] = 300
    state, info = env.reset()
    done = False
    env.observation_space
    score = 0
    reward, n_state, done = 0, 0, 0
    while not done:
        state = env.render()
        height, width, channels = env.observation_space.shape
        action = eval_network(net, state)
        # action = random.choice(list(range(0, 5)))
        state, reward, n_state, done, info = env.step(action)
        score += reward
        done = info.get("lives") == 0
    print(f"Score: {score}")
    genome.fitness += score
    env.close()
    pass


def eval_genomes(genomes: tuple, config):
    game = initialize_atari_env_game()
    for id, genome in genomes:
        genome.fitness = 0 if genome.fitness == None else genome.fitness
        train_ai(game, genome, config)
    return


def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner = p.run(eval_genomes, 20)
    with open("best_space_invaders.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    with open("best_space_invaders.pickle", "rb") as f:
        genome = pickle.load(f)
    env = initialize_atari_env_game(render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env.metadata["render_fps"] = 300
    state, info = env.reset()
    done = False
    score = 0
    reward, n_state, done = 0, 0, 0
    while not done:
        env.render()
        action = eval_network(net, state)

        # action = random.choice(list(range(0, 5)))
        state, reward, n_state, done, info = env.step(action)
        score += reward
        done = info.get("lives") == 0
    print(f"Score: {score}")
    env.close()
    pass


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_space_invaders.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    # run_neat(config)
    test_ai(config)

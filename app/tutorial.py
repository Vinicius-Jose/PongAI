import pygame
from pong.game import GameInformation
from pong import Game
import neat
import os
import pickle


class PongGame:
    def __init__(self, window, width, height) -> None:
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = net.activate(
                (
                    self.right_paddle.y,
                    self.ball.y,
                    abs(self.right_paddle.x - self.ball.x),
                )
            )
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome_1, genome_2, config):
        net_1 = neat.nn.FeedForwardNetwork.create(genome_1, config)
        net_2 = neat.nn.FeedForwardNetwork.create(genome_2, config)
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output_1 = net_1.activate(
                (self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x))
            )
            decision_1 = output_1.index(max(output_1))
            if decision_1 == 0:
                pass
            elif decision_1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            output_2 = net_2.activate(
                (
                    self.right_paddle.y,
                    self.ball.y,
                    abs(self.right_paddle.x - self.ball.x),
                )
            )
            decision_2 = output_2.index(max(output_2))

            if decision_2 == 0:
                pass
            elif decision_2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()
            if (
                game_info.left_score >= 1
                or game_info.right_score >= 1
                or game_info.left_hits > 50
            ):
                self.calculate_fitness(genome_1, genome_2, game_info)
                break

    def calculate_fitness(self, genome_1, genome_2, game_info: GameInformation):
        genome_1.fitness += game_info.left_hits
        genome_2.fitness += game_info.right_hits


def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    for i, (genome_id, genome_1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome_1.fitness = 0
        for genome_id2, genome_2 in genomes[i + 1 :]:
            genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome_1, genome_2, config)
    return


def run_neat(config):
    p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-20")
    # p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner = p.run(eval_genomes, 20)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    # run_neat(config)
    test_ai(config)

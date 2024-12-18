import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent


def initialize_game():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Tron Game")
    return screen


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def draw_game(screen, game_board, player1, player2):
    screen.fill((0, 0, 0))
    game_board.draw(screen)
    player1.draw(screen)
    player2.draw(screen)
    pygame.display.flip()


def run_game(ai_class1, ai_class2, model_file1=None, model_file2=None):
    screen = initialize_game()
    game_board = GameBoard(40, 30)

    ai1 = RLAgent(state_size=7 * 7 * 3, action_size=4, player_id=1,
                  model_file=model_file1) if ai_class1 == RLAgent else ai_class1()
    ai2 = RLAgent(state_size=7 * 7 * 3, action_size=4, player_id=2,
                  model_file=model_file2) if ai_class2 == RLAgent else ai_class2()

    player1 = Player(10, 15, (255, 0, 0), 1, ai1)
    player2 = Player(30, 15, (0, 0, 255), 2, ai2)

    # Set opponents for both players
    player1.set_opponent(player2)
    player2.set_opponent(player1)

    clock = pygame.time.Clock()

    running = True
    while running:
        running = handle_events()
        if running:
            player1.move(game_board)
            player2.move(game_board)
            if game_board.is_collision(player1.x, player1.y) or game_board.is_collision(player2.x, player2.y):
                running = False
            else:
                game_board.grid[player1.y][player1.x] = player1.player_id
                game_board.grid[player2.y][player2.x] = player2.player_id
        draw_game(screen, game_board, player1, player2)
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    run_game(RLAgent, RLAgent, "tron_model_player1.pth", "tron_model_player2.pth")
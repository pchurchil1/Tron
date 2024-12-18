import pygame

class PlayerBot:
    def __init__(self):
        self.directions = {
            pygame.K_UP: [0, -1],
            pygame.K_DOWN: [0, 1],
            pygame.K_LEFT: [-1, 0],
            pygame.K_RIGHT: [1, 0]
        }
        self.current_direction = [1, 0]

    def get_direction(self, game_board, player):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.directions:
                    new_direction = self.directions[event.key]
                    if new_direction[0] * self.current_direction[0] + new_direction[1] * self.current_direction[1] == 0:
                        self.current_direction = new_direction
        return self.current_direction
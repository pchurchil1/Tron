import pygame


class Player:
    def __init__(self, x, y, color, player_id, ai):
        """
        Initialize the player.
        :param x: Initial x-coordinate
        :param y: Initial y-coordinate
        :param color: Color of the player's trail
        :param player_id: ID of the player (1 or 2)
        :param ai: AI object that provides directions
        """
        self.x = x
        self.y = y
        self.color = color
        self.player_id = player_id
        self.direction = [1, 0] if player_id == 1 else [-1, 0]
        self.trail = [(x, y)]
        self.controller = ai
        self.opponent = None  # Will be set later

    def set_opponent(self, opponent):
        """
        Set the opponent for this player.
        :param opponent: The opponent Player object
        """
        self.opponent = opponent

    def move(self, game_board):
        """
        Move the player based on their current direction.
        """
        if self.opponent is None:
            raise ValueError("Opponent not set. Call set_opponent() before moving.")

        action = self.controller.get_direction(game_board, self, self.opponent)
        self.change_direction(action)

        new_x = self.x + self.direction[0]
        new_y = self.y + self.direction[1]

        if 0 <= new_x < game_board.width and 0 <= new_y < game_board.height:
            self.x = new_x
            self.y = new_y
            self.trail.append((self.x, self.y))
            return False  # No collision
        else:
            return True  # Collision occurred

    def change_direction(self, new_direction):
        """
        Change the player's direction, preventing 180-degree turns.
        :param new_direction: New direction as a list [dx, dy]
        """
        if new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]:
            self.direction = new_direction

    def draw(self, screen):
        """
        Draw the player and their trail on the screen.
        :param screen: Pygame screen object to draw on
        """
        for x, y in self.trail:
            pygame.draw.rect(screen, self.color,
                             (x * 20, y * 20, 20, 20))

    def reset(self, x, y):
        """
        Reset the player's position and trail.
        :param x: New x-coordinate
        :param y: New y-coordinate
        """
        self.x = x
        self.y = y
        self.direction = [1, 0] if self.player_id == 1 else [-1, 0]
        self.trail = [(x, y)]
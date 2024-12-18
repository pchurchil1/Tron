import pygame

class GameBoard:
    def __init__(self, width, height):
        """
        Initialize the game board.
        :param width: Width of the game board in grid cells
        :param height: Height of the game board in grid cells
        """
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.cell_size = 20

    def draw(self, screen):
        """
        Draw the game board on the screen.
        :param screen: Pygame screen object to draw on
        """
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                   self.cell_size, self.cell_size)
                if self.grid[y][x] == 0:
                    pygame.draw.rect(screen, (50, 50, 50), rect)
                elif self.grid[y][x] == 1:
                    pygame.draw.rect(screen, (200, 0, 0), rect)
                elif self.grid[y][x] == 2:
                    pygame.draw.rect(screen, (0, 0, 200), rect)

    def is_collision(self, x, y):
        """
        Check if the given coordinates collide with the board boundaries or a trail.
        :param x: X-coordinate to check
        :param y: Y-coordinate to check
        :return: True if collision, False otherwise
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            print(f"Collision detected: Out of bounds at ({x}, {y})")
            return True
        if self.grid[y][x] != 0:
            print(f"Collision detected: Trail or obstacle at ({x}, {y})")
            return True
        return False
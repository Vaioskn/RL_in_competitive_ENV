import pygame
import math
from settings import white, hex_size, hex_width, GOAL_CELLS, startx, starty, CELLS
from utilities import hex_center, get_hex_points

def draw_hex(screen, column, row, color=(0, 0, 0)):
    x, y = hex_center(column, row)
    points = get_hex_points(x, y)

    if (column, row) in GOAL_CELLS:
        highlight_hex(screen, x, y, (255, 0, 0, 128))

        font = pygame.font.Font(None, 36)
        text_surface = font.render("Goal", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    else:
        pygame.draw.polygon(screen, color, points, 2)

def highlight_hex(screen, x, y, color=(255, 255, 0, 128)):
    surface = pygame.Surface((hex_width, 2 * hex_size), pygame.SRCALPHA)
    hex_points = [
        (hex_width / 2 + hex_size * math.cos(math.radians(angle + 30)),
         hex_size + hex_size * math.sin(math.radians(angle + 30)))
        for angle in range(0, 360, 60)
    ]
    pygame.draw.polygon(surface, color, hex_points)
    screen.blit(surface, (x - hex_width / 2, y - hex_size))

def draw_field(screen):
    for col, row in CELLS:
        draw_hex(screen, col, row)

def redraw_game_state(player, agent, ball, scoreboard, screen, buttons):
    screen.fill(white)
    draw_field(screen)
    player.draw(screen, agent, ball)
    ball.draw(screen, player, agent)
    agent.draw(screen, player, ball)
    scoreboard.draw(screen)
    for button in buttons:
        button.draw(screen)

    pygame.display.flip()

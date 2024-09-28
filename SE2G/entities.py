import pygame
import random
from settings import GOAL_CELLS, CORNER_CELLS
from utilities import hex_center, neighbor_hex, highlight_hex, determine_direction, get_clicked_cell, find_cell_in_direction

class Player:
    def __init__(self, column, row):
        self.column = column
        self.row = row
        self.x, self.y = hex_center(self.column, self.row)
        self.radius = 20
        self.selected = False
        self.User_Controlled = True
        self.Random = False
        self.QLearning = False

    def draw(self, screen, agent, ball):
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 2)
        if self.selected:
            self.highlight_moves(screen, agent, ball)

    def highlight_moves(self, screen, agent, ball):
        for col, row in neighbor_hex(self.column, self.row):
            x, y = hex_center(col, row)
            if (col, row) != (agent.column, agent.row) and (col, row) != (ball.column, ball.row) and (col, row) not in GOAL_CELLS and (col, row) not in CORNER_CELLS:
                highlight_hex(screen, x, y, (173, 216, 230, 128))
            if (col, row) != (agent.column, agent.row) and (col, row) == (ball.column, ball.row) and (col, row) not in GOAL_CELLS and (col, row) not in CORNER_CELLS and not ball.selected:
                highlight_hex(screen, x, y, (0, 255, 0, 128))

    def move_to(self, column, row):
        if (column, row) not in GOAL_CELLS and (column, row) not in CORNER_CELLS:
            self.column = column
            self.row = row
            self.x, self.y = hex_center(self.column, self.row)
            self.selected = False 

    def is_clicked(self, pos):
        distance = ((self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2) ** 0.5
        return distance <= self.radius

    def get_possible_actions(self, agent, ball):
        player_position = (self.column, self.row)
        ball_position = (ball.column, ball.row)
        agent_position = (agent.column, agent.row)

        neighbors = neighbor_hex(*player_position)
        valid_moves = [
            cell for cell in neighbors
            if cell not in GOAL_CELLS and cell not in CORNER_CELLS and cell != ball_position and cell != agent_position
        ]

        actions = [
            f"move_{determine_direction(hex_center(*player_position), hex_center(*movable_cell))}"
            for movable_cell in valid_moves
        ]

        if player_position in neighbor_hex(*ball_position):
            ball_neighbors = neighbor_hex(*ball_position)
            kickable = [
                cell for cell in ball_neighbors
                if cell not in neighbors and cell != player_position and cell != agent_position
            ]
            actions.extend(
                f"kick_{determine_direction(hex_center(*ball_position), hex_center(*kickable_cell))}"
                for kickable_cell in kickable
            )

        return actions

    def perform_action(self, ball, action_string):
        parts = action_string.split('_', 1)  # Split only on the first underscore
        if len(parts) == 2:
            command, direction = parts[0], parts[1]

            if command == "move":
                target_cell = find_cell_in_direction(self, direction)
            elif command == "kick":
                target_cell = find_cell_in_direction(ball, direction)

            if target_cell:
                clicked_cell = get_clicked_cell(*hex_center(*target_cell))

                if command == 'move':
                    self.move_to(*clicked_cell)
                elif command == 'kick' and ball:
                    ball.move_to(*clicked_cell)

    def perform_random_action(self, agent, ball):
        actions = self.get_possible_actions(agent, ball)
        if actions:
            random_action = random.choice(actions)
            self.perform_action(ball, random_action)

    def get_position(self):
        return (self.column, self.row)


class Agent:
    def __init__(self, column, row):
        self.column = column
        self.row = row
        self.x, self.y = hex_center(self.column, self.row)
        self.radius = 20
        self.selected = False
        self.QLearning = True
        self.MinMaxQLearning = False
        self.TBRL = False

    def draw(self, screen, player, ball):
        start_line1 = (int(self.x - self.radius), int(self.y - self.radius))
        end_line1 = (int(self.x + self.radius), int(self.y + self.radius))
        start_line2 = (int(self.x + self.radius), int(self.y - self.radius))
        end_line2 = (int(self.x - self.radius), int(self.y + self.radius))
        pygame.draw.line(screen, (0, 0, 0), start_line1, end_line1, 2)
        pygame.draw.line(screen, (0, 0, 0), start_line2, end_line2, 2)

    def move_to(self, column, row):
        if (column, row) not in GOAL_CELLS and (column, row) not in CORNER_CELLS:
            self.column = column
            self.row = row
            self.x, self.y = hex_center(self.column, self.row)
            self.selected = False

    def get_possible_actions(self, player, ball):
        player_position = (player.column, player.row)
        ball_position = (ball.column, ball.row)
        agent_position = (self.column, self.row)

        neighbors = neighbor_hex(*agent_position)

        valid_moves = []
        for cell in neighbors:
            if cell not in GOAL_CELLS and cell not in CORNER_CELLS and cell != ball_position and cell != player_position:
                valid_moves.append(cell)
                direction = determine_direction(hex_center(*agent_position), hex_center(*cell))

        actions = [
            f"move_{determine_direction(hex_center(*agent_position), hex_center(*movable_cell))}"
            for movable_cell in valid_moves
        ]

        if agent_position in neighbor_hex(*ball_position):
            ball_neighbors = neighbor_hex(*ball_position)
            kickable = []
            for cell in ball_neighbors:
                if cell not in neighbors and cell != agent_position and cell != player_position:
                    kickable.append(cell)
                    direction = determine_direction(hex_center(*ball_position), hex_center(*cell))
                    actions.append(f"kick_{direction}")

        return actions

    def perform_action(self, ball, action_string):
        parts = action_string.split('_', 1)  # Split only on the first underscore
        if len(parts) == 2:
            command, direction = parts[0], parts[1]

            if command == "move":
                target_cell = find_cell_in_direction(self, direction)
            elif command == "kick":
                target_cell = find_cell_in_direction(ball, direction)

            if target_cell:
                clicked_cell = get_clicked_cell(*hex_center(*target_cell))

                if command == 'move':
                    self.move_to(*clicked_cell)
                elif command == 'kick' and ball:
                    ball.move_to(*clicked_cell)

    def perform_random_action(self, player, ball):
        actions = self.get_possible_actions(player, ball)
        if actions:
            random_action = random.choice(actions)
            self.perform_action(ball, random_action)

    def get_position(self):
        return (self.column, self.row)


class Ball:
    def __init__(self, column, row):
        self.column = column
        self.row = row
        self.x, self.y = hex_center(self.column, self.row)
        self.radius = 5
        self.selected = False

    def draw(self, screen, player, ball):
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius)
        if self.selected:
            self.highlight_moves(screen, player, ball)

    def highlight_moves(self, screen, player, ball):
        for col, row in neighbor_hex(self.column, self.row):
            if (col, row) not in neighbor_hex(player.column, player.row):
                x, y = hex_center(col, row)
                if (col, row) != (player.column, player.row) and (col, row) != (ball.column, ball.row):
                    highlight_hex(screen, x, y, (255, 255, 0, 128))

    def move_to(self, column, row):
        self.column = column
        self.row = row
        self.x, self.y = hex_center(self.column, self.row)
        self.selected = False

    def is_clicked(self, pos):
        distance = ((self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2) ** 0.5
        return distance <= self.radius

    def get_position(self):
        return (self.column, self.row)


class ScoreBoard:
    def __init__(self, x, y, font_size=36):
        self.x = x
        self.y = y
        self.left_score = 0
        self.right_score = 0
        self.font = pygame.font.Font(None, font_size)

    def draw(self, screen):
        score_text = f"{self.left_score} - {self.right_score}"
        text_surface = self.font.render(score_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(self.x, self.y))
        screen.blit(text_surface, text_rect)

    def increment_left(self):
        self.left_score += 1

    def increment_right(self):
        self.right_score += 1



class Button:
    def __init__(self, x, y, text, player, agent, selected=False, buttons=None):
        self.x = x
        self.y = y
        self.text = text
        self.font_size = 36
        self.font = pygame.font.Font(None, 36)
        self.text_color = (0, 0, 0)
        self.buttons = buttons

        text_surface = self.font.render(self.text, True, self.text_color)
        self.width = text_surface.get_width() + 20
        self.height = 50

        self.player = player
        self.agent = agent
        self.selected = selected
        self.color = (173, 216, 230)
        self.selected_color = (64, 156, 214)

    def draw(self, screen):
        border_radius = 10
        fill_color = self.selected_color if self.selected else self.color
        pygame.draw.rect(screen, fill_color, (self.x, self.y, self.width, self.height), border_radius=border_radius)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surface, text_rect)

    def check_click(self, position):
        x, y = position
        if self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height:
            if not self.selected:
                self.toggle_buttons()
                self.selected = True
                self.execute_selected_action()

    def toggle_buttons(self):
        relevant_buttons = [b for b in self.buttons if (b.x == 50) == (self.x == 50)]
        
        for button in relevant_buttons:
            if button != self:
                button.selected = False

    def execute_selected_action(self):
        if self.x == 50:  # Actions for agent
            self.agent.QLearning = (self.text == "Q-Learning")
            self.agent.MinMaxQLearning = (self.text == "MinMax Q-Learning")
            self.agent.TBRL = (self.text == "TBRL")
        else:  # Actions for player
            self.player.User_Controlled = (self.text == "User-Controlled")
            self.player.QLearning = (self.text == "Q-Learning")
            self.player.MinMaxQLearning = (self.text == "MinMax Q-Learning")
            self.player.TBRL = (self.text == "TBRL")
        

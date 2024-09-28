import os
import math
import pygame
import csv
import copy
from settings import CELLS, GOAL_CELLS, CORNER_CELLS, startx, starty, hex_size, hex_width

def hex_center(column, row):
    x = startx + column * hex_width + 0.5 * hex_width * (row % 2)
    y = starty + row * 1.5 * hex_size
    return x, y

def get_hex_points(x, y):
   points = [
        (x + hex_size * math.cos(math.radians(angle + 30)), y + hex_size * math.sin(math.radians(angle + 30)))
        for angle in range(0, 360, 60)
    ]
   return points

def neighbor_hex(column, row):
    neighbors = []

    # For even rows
    if row % 2 == 0:
        potential_neighbors = [
            (column - 1, row),    # Left
            (column + 1, row),    # Right
            (column - 1, row - 1),# Top-left
            (column, row - 1),    # Top-right
            (column - 1, row + 1),# Bottom-left
            (column, row + 1)     # Bottom-right
        ]
    # For odd rows
    else:
        potential_neighbors = [
            (column - 1, row),
            (column + 1, row),
            (column, row - 1),
            (column + 1, row - 1),
            (column, row + 1),
            (column + 1, row + 1)
        ]

    for col, rw in potential_neighbors:
        if ((col, rw) in CELLS):
            neighbors.append((col, rw))
    
    return neighbors

def get_clicked_cell(click_x, click_y):
    closest_distance = float('inf')
    closest_cell = None
    for col, row in CELLS:
        hex_x, hex_y = hex_center(col, row)
        distance = math.sqrt((hex_x - click_x) ** 2 + (hex_y - click_y) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_cell = (col, row)

    if closest_distance <= hex_size:  # Assuming a click within the hex_size is valid
        return closest_cell
    return None

def highlight_hex(screen, x, y, color=(0, 0, 0, 128)):
    surface = pygame.Surface((hex_width, 2 * hex_size), pygame.SRCALPHA)

    hex_points = [
        (hex_width / 2 + hex_size * math.cos(math.radians(angle + 30)), 
         hex_size + hex_size * math.sin(math.radians(angle + 30)))
        for angle in range(0, 360, 60)
    ]

    pygame.draw.polygon(surface, color, hex_points)

    screen.blit(surface, (x - hex_width / 2, y - hex_size))

def reset_game(player, agent, ball):
    player.move_to(5,2)
    agent.move_to(1,2)
    ball.move_to(3,2)

def check_for_goal_or_corners(player, agent, ball, scoreboard):
    if (ball.column, ball.row) == (0,2):
        scoreboard.increment_right()
    if (ball.column, ball.row) == (6,2):
        scoreboard.increment_left()
    if (ball.column, ball.row) in CORNER_CELLS or (ball.column, ball.row) in GOAL_CELLS:
        reset_game(player, agent, ball)

def determine_direction(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = -(to_pos[1] - from_pos[1])  # Negate dy to adjust for coordinate system

    angle = math.degrees(math.atan2(dy, dx))
    if -30 <= angle < 30:
        return 'right'
    elif 30 <= angle < 90:
        return 'up_right'
    elif 90 <= angle < 150:
        return 'up_left'
    elif 150 <= angle <= 180 or -180 <= angle < -150:
        return 'left'
    elif -150 <= angle < -90:
        return 'down_left'
    elif -90 <= angle < -30:
        return 'down_right'
    return 'undefined'

def find_cell_in_direction(entity, desired_direction):
    current_pos = (entity.column, entity.row)
    neighbors = neighbor_hex(*current_pos)
    for neighbor in neighbors:
        neighbor_center = hex_center(*neighbor)
        current_center = hex_center(*current_pos)
        direction = determine_direction(current_center, neighbor_center)
        if direction == desired_direction:
            return neighbor
    return None

def check_if_selected(buttons, name):
    return any(button.selected and button.text == name for button in buttons)

def log_to_csv(entity, state, reward, action, filename='states.csv'):
    fieldnames = ['entity', 'ball_position', 'player_position', 
                  'agent_position', 'score', 'reward', 'action']

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({
            'entity': entity,
            'ball_position': state['ball_position'],
            'player_position': state['player_position'],
            'agent_position': state['agent_position'],
            'score': state['score'],
            'reward': reward,
            'action': action
        })

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_possible_actions_for_agent(player_position, agent_position, ball_position):
    neighbors = neighbor_hex(*agent_position)

    valid_moves = []
    for cell in neighbors:
        if cell not in GOAL_CELLS and cell != ball_position and cell != player_position and cell not in CORNER_CELLS:
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

def get_possible_actions_for_player(player_position, agent_position, ball_position):
    neighbors = neighbor_hex(*player_position)
    valid_moves = [
        cell for cell in neighbors
        if cell not in GOAL_CELLS and cell != ball_position and cell != agent_position and cell not in CORNER_CELLS
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

# def simulate_execute_action(state, action, agent_type):
#     # Extract initial positions from the state dictionary
#     agent_pos = state['agent_position']
#     player_pos = state['player_position']
#     ball_pos = state['ball_position']

#     new_state = {
#         'agent_position': agent_pos,
#         'ball_position': ball_pos,
#         'player_position': player_pos
#     }

#     # Determine the subject of the action (agent/player or ball) and action direction
#     action_type, direction = action.split('_', 1)
    
#     # Decide who is acting based on agent_type
#     if agent_type == 'agent':
#         actor_pos = agent_pos
#     elif agent_type == 'player':
#         actor_pos = player_pos

#     # Determine the new position based on the direction of the action
#     if action_type == 'move':
#         new_pos = find_cell_in_direction_position(actor_pos, direction)
#         # Update the state with the new position of the agent
#         if agent_type == 'agent':
#             new_state['agent_position'] = new_pos
#         elif agent_type == 'player':
#             new_state['player_position'] = new_pos
#     elif action_type == 'kick':
#         new_pos = find_cell_in_direction_position(ball_pos, direction)
#         # Update the state with the new position of the ball
#         new_state['ball_position'] = new_pos

#     return new_state

def simulate_execute_action(state, action, agent_type):
    # Extract initial positions from the state dictionary
    agent_pos = state.get('agent_position')
    player_pos = state.get('player_position')
    ball_pos = state.get('ball_position')
    
    # Log initial state
    # print(f"Simulating action: {action} by {agent_type}")
    # print(f"Initial positions - Agent: {agent_pos}, Player: {player_pos}, Ball: {ball_pos}")

    new_state = {
        'agent_position': agent_pos,
        'ball_position': ball_pos,
        'player_position': player_pos
    }

    try:
        # Determine the subject of the action (agent/player or ball) and action direction
        action_type, direction = action.split('_', 1)
    except ValueError:
        # print(f"Error: Invalid action format '{action}'. Expected format 'action_direction'.")
        return new_state  # Return the state unchanged in case of invalid action

    # Decide who is acting based on agent_type
    if agent_type == 'agent':
        actor_pos = agent_pos
    elif agent_type == 'player':
        actor_pos = player_pos
    else:
        # print(f"Error: Unknown agent_type '{agent_type}'. Expected 'agent' or 'player'.")
        return new_state  # Return the state unchanged in case of invalid agent_type

    # Determine the new position based on the direction of the action
    new_pos = find_cell_in_direction_position(actor_pos, direction)
    if new_pos is None:
        # print(f"Warning: Movement from {actor_pos} to direction '{direction}' is invalid. Position remains unchanged.")
        return new_state  # Return the state unchanged if the movement is invalid

    # Check if the new position is within game boundaries and not occupied
    if not (new_pos in CELLS and new_pos not in [agent_pos, player_pos, ball_pos]):
        # print(f"Warning: New position {new_pos} is invalid, out of bounds, or occupied.")
        return new_state  # Return the state unchanged if the new position is invalid

    # Update positions based on the action type
    if action_type == 'move':
        if agent_type == 'agent':
            new_state['agent_position'] = new_pos
        elif agent_type == 'player':
            new_state['player_position'] = new_pos
    elif action_type == 'kick':
        new_state['ball_position'] = new_pos
    else:
        # print(f"Error: Unknown action type '{action_type}'. Expected 'move' or 'kick'.")
        return new_state  # Return the state unchanged in case of invalid action type

    # Log the updated state
    # print(f"Updated positions - Agent: {new_state['agent_position']}, Player: {new_state['player_position']}, Ball: {new_state['ball_position']}")

    return new_state


def find_cell_in_direction_position(position, desired_direction):
    current_pos = position
    neighbors = neighbor_hex(*current_pos)
    for neighbor in neighbors:
        neighbor_center = hex_center(*neighbor)
        current_center = hex_center(*current_pos)
        direction = determine_direction(current_center, neighbor_center)
        if direction == desired_direction:
            return neighbor
    return None

def state_to_tuple(state):
    return tuple(sorted(state.items()))

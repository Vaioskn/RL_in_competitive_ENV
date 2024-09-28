import pygame
import os
from settings import window_width, window_height, white, MAX_EPSILON_Q, MAX_EPSILON_MIN_MAX, EPISODES_FOR_QUITING_PROGRAM, INFINITE_EPISODES, MAX_ALPHA, Q_LEARNING_WITHOUT_OPPONENT_AGENT, Q_LEARNING_WITHOUT_OPPONENT_PLAYER
from view import draw_field, redraw_game_state
from entities import Player, Agent, Ball, ScoreBoard, Button
from utilities import get_clicked_cell, neighbor_hex, check_for_goal_or_corners, log_to_csv, determine_direction, print_successful_action_percentages, modify_state_for_small_env
from qlearning import QLearningAgent
from minimaxQLearning import MinimaxQLearningAgent
from beliefQLearning import BeliefBasedQLearningAgent
from state import State
import datetime

if os.path.exists("states_no_rewards.csv"):
    open("states_no_rewards.csv", 'w').close()

def main():
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Hex Soccer Game")

    # Initialize counters
    fallback_player_qlearning_count = 0
    successful_player_qlearning_count = 0
    fallback_agent_qlearning_count = 0
    successful_agent_qlearning_count = 0
    fallback_minimax_count = 0
    successful_minimax_count = 0
    fallback_belief_count = 0
    successful_belief_count = 0

    backup_dir_q = 'C:\\Users\\baios\Desktop\\small_env_q_table_backups'
    backup_dir_minmax = 'C:\\Users\\baios\Desktop\\small_env_minimax_tables_backups'
    backup_dir_belief = 'C:\\Users\\baios\Desktop\\small_env_belief_tables_backups'
    os.makedirs(backup_dir_q, exist_ok=True)
    os.makedirs(backup_dir_minmax, exist_ok=True)
    os.makedirs(backup_dir_belief, exist_ok=True)

    player = Player(8, 2)
    agent = Agent(2, 2)
    ball = Ball(5, 2)
    scoreboard = ScoreBoard(window_width // 2, 30)
    buttons = [
        Button(50, 430, "Q-Learning", player=player, agent=agent, selected=True, buttons=None),
        Button(50, 490, "MinMax Q-Learning", player=player, agent=agent, selected=False, buttons=None),
        Button(50, 550, "TBRL", player=player, agent=agent, selected=False, buttons=None),
        Button(550, 430, "User-Controlled", player=player, agent=agent, selected=True, buttons=None),
        Button(550, 490, "Q-Learning", player=player, agent=agent, selected=False, buttons=None)
    ]
    for button in buttons:
        button.buttons = buttons 

   
    player_q_agent = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='player')
    player_q_agent.load_q_table('player_q_table.json')

    agent_q_agent = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='agent')
    agent_q_agent.load_q_table('agent_q_table.json')

    player_q_agent_small_env_1 = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='player')
    player_q_agent_small_env_1.load_q_table('small_env_1_player_q_table.json')
    player_q_agent_small_env_2 = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='player')
    player_q_agent_small_env_2.load_q_table('small_env_2_player_q_table.json')

    agent_q_agent_small_env_1 = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='agent')
    agent_q_agent_small_env_1.load_q_table('small_env_1_agent_q_table.json')
    agent_q_agent_small_env_2 = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='agent')
    agent_q_agent_small_env_2.load_q_table('small_env_2_agent_q_table.json')


    agent_minimax_agent_env_1 = MinimaxQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)
    agent_minimax_agent_env_1.load_tables('small_env_1_agent_minimax_q_table.json', 'small_env_1_agent_minimax_pi_table.json', 'small_env_1_agent_minimax_v_table.json')

    agent_minimax_agent_env_2 = MinimaxQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)
    agent_minimax_agent_env_2.load_tables('small_env_2_agent_minimax_q_table.json', 'small_env_2_agent_minimax_pi_table.json', 'small_env_2_agent_minimax_v_table.json')


    agent_belief_agent_env_1 = BeliefBasedQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)
    agent_belief_agent_env_1.load_tables('small_env_1_agent_belief_q_table.json', 'small_env_1_agent_belief_v_table.json', 'small_env_1_agent_belief_table.json')

    agent_belief_agent_env_2 = BeliefBasedQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)
    agent_belief_agent_env_2.load_tables('small_env_2_agent_belief_q_table.json', 'small_env_2_agent_belief_v_table.json', 'small_env_2_agent_belief_table.json')

    running = True
    agent_take_turn = False
    while running:
        before_loop_state_for_logging_1 = State(player, agent, ball, scoreboard).get_state()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_cell = get_clicked_cell(*event.pos)

                for button in buttons:
                    button.check_click(event.pos)

                if player.User_Controlled:
                    if clicked_cell == (player.column, player.row):
                        player.selected = not player.selected
                        ball.selected = False
                        continue

                    if player.selected:
                        if clicked_cell in neighbor_hex(player.column, player.row) and clicked_cell != (ball.column, ball.row) and clicked_cell != (agent.column, agent.row):
                            direction = determine_direction((player.x, player.y), event.pos)
                            player_action = f"move_{direction}"
                            player.move_to(*clicked_cell)
                            ball.selected = False
                            agent_take_turn = True
                            continue
                        if clicked_cell in neighbor_hex(player.column, player.row) and clicked_cell == (ball.column, ball.row) and clicked_cell != (agent.column, agent.row):
                            ball.selected = not ball.selected
                            continue

                    if player.selected and ball.selected:
                        if clicked_cell in neighbor_hex(ball.column, ball.row) and clicked_cell not in neighbor_hex(player.column, player.row) and clicked_cell != (player.column, player.row) and clicked_cell != (agent.column, agent.row):
                            direction = determine_direction((ball.x, ball.y), event.pos)
                            player_action = f"kick_{direction}"
                            ball.move_to(*clicked_cell)
                            player.selected = False
                            agent_take_turn = True
                            continue

        if player.QLearning and Q_LEARNING_WITHOUT_OPPONENT_PLAYER:
            current_player_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_player()
            player_action = player_q_agent.choose_action(current_player_state)
            player.perform_action(ball, player_action)
            agent_take_turn = True

        if player.QLearning and not Q_LEARNING_WITHOUT_OPPONENT_PLAYER:
            current_player_state = State(player, agent, ball, scoreboard).get_state_for_small_env()
            
            ball_x = current_player_state['ball_position'][0]
            modified_player_state = modify_state_for_small_env(current_player_state)

            if ball_x in [4, 5, 6]:
                player_action = player_q_agent_small_env_1.choose_action(modified_player_state)
            elif ball_x in [0, 1, 2, 3, 7, 8, 9, 10]:
                player_action = player_q_agent_small_env_2.choose_action(modified_player_state)
            
            if player_action is None:
                fallback_player_qlearning_count += 1
                fallback_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_player()
                player_action = player_q_agent.choose_action(fallback_state)
            else:
                successful_player_qlearning_count += 1

            player.perform_action(ball, player_action)
            agent_take_turn = True

        next_player_state_for_agent_training = State(player, agent, ball, scoreboard).get_state()

        before_loop_state_for_logging_2 = State(player, agent, ball, scoreboard).get_state()
        check_for_goal_or_corners(player, agent, ball, scoreboard)
        redraw_game_state(player, agent, ball, scoreboard, screen, buttons)
        after_player_turn_state_for_logging = State(player, agent, ball, scoreboard).get_state()

        if agent_take_turn:
            if agent.QLearning and Q_LEARNING_WITHOUT_OPPONENT_AGENT:
                current_agent_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                agent_action = agent_q_agent.choose_action(current_agent_state)
                agent.perform_action(ball, agent_action)
                agent_take_turn = False
            elif agent.QLearning and not Q_LEARNING_WITHOUT_OPPONENT_AGENT:
                current_agent_state = State(player, agent, ball, scoreboard).get_state_for_small_env()

                ball_x = current_player_state['ball_position'][0]
                modified_agent_state = modify_state_for_small_env(current_agent_state)

                if ball_x in [4, 5, 6]:
                    agent_action = agent_q_agent_small_env_1.choose_action(modified_agent_state)
                elif ball_x in [0, 1, 2, 3, 7, 8, 9, 10]:
                    agent_action = agent_q_agent_small_env_2.choose_action(modified_agent_state)

                if agent_action is None:
                    fallback_agent_qlearning_count += 1
                    fallback_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                    agent_action = agent_q_agent.choose_action(fallback_state)
                else:
                    successful_agent_qlearning_count += 1

                agent.perform_action(ball, agent_action)
                agent_take_turn = False
            elif agent.MinMaxQLearning:
                current_agent_state = State(player, agent, ball, scoreboard).get_state_for_small_env()

                ball_x = current_agent_state['ball_position'][0]
                modified_agent_state = modify_state_for_small_env(current_agent_state)

                if ball_x in [4, 5, 6]:
                    agent_action = agent_minimax_agent_env_1.choose_action(modified_agent_state)
                elif ball_x in [0, 1, 2, 3, 7, 8, 9, 10]:
                    agent_action = agent_minimax_agent_env_2.choose_action(modified_agent_state)

                if agent_action is None:
                    fallback_minimax_count += 1
                    fallback_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                    agent_action = agent_q_agent.choose_action(fallback_state)
                else:
                    successful_minimax_count += 1

                agent.perform_action(ball, agent_action)
                agent_take_turn = False
            elif agent.TBRL:
                current_agent_state = State(player, agent, ball, scoreboard).get_state_for_small_env()

                ball_x = current_agent_state['ball_position'][0]
                modified_agent_state = modify_state_for_small_env(current_agent_state)

                if ball_x in [4, 5, 6]:
                    agent_action = agent_belief_agent_env_1.choose_action(modified_agent_state)
                elif ball_x in [0, 1, 2, 3, 7, 8, 9, 10]:
                    agent_action = agent_belief_agent_env_2.choose_action(modified_agent_state)

                if agent_action is None:
                    fallback_belief_count += 1
                    fallback_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                    agent_action = agent_q_agent.choose_action(fallback_state)
                else:
                    successful_belief_count += 1

                agent.perform_action(ball, agent_action)
                agent_take_turn = False

        after_loop_state_for_logging_1 = State(player, agent, ball, scoreboard).get_state()
        check_for_goal_or_corners(player, agent, ball, scoreboard)
        redraw_game_state(player, agent, ball, scoreboard, screen, buttons)
        after_loop_state_for_logging_2 = State(player, agent, ball, scoreboard).get_state()
        if after_loop_state_for_logging_2 != before_loop_state_for_logging_1:
            log_to_csv('Player', before_loop_state_for_logging_1, player_action)
            if before_loop_state_for_logging_2 != after_player_turn_state_for_logging:
                log_to_csv('Player', before_loop_state_for_logging_2, "Goal_reset")
            log_to_csv('Agent', after_player_turn_state_for_logging, agent_action)
            if after_loop_state_for_logging_1 != after_loop_state_for_logging_2:
                log_to_csv('Agent', after_loop_state_for_logging_1, "Goal_reset")
            
        if scoreboard.left_score + scoreboard.right_score == EPISODES_FOR_QUITING_PROGRAM and not INFINITE_EPISODES:
            print_successful_action_percentages(fallback_player_qlearning_count,successful_player_qlearning_count,fallback_agent_qlearning_count,successful_agent_qlearning_count,fallback_minimax_count,successful_minimax_count,fallback_belief_count,successful_belief_count)
            pygame.quit()

    print_successful_action_percentages(fallback_player_qlearning_count,successful_player_qlearning_count,fallback_agent_qlearning_count,successful_agent_qlearning_count,fallback_minimax_count,successful_minimax_count,fallback_belief_count,successful_belief_count)
    pygame.quit()

if __name__ == "__main__":
    main()

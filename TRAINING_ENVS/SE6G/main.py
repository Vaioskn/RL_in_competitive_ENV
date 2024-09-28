import pygame
import os
from settings import window_width, window_height, white, MAX_EPSILON_Q, MAX_EPSILON_MIN_MAX, EPISODES_FOR_QUITING_PROGRAM, INFINITE_EPISODES, MAX_ALPHA, ENABLE_Q_LEARNING_TRAINING, ENABLE_MINIMAX_Q_LEARNING_TRAINING, ENABLE_BELIEF_Q_LEARNING_TRAINING
from view import draw_field, redraw_game_state
from entities import Player, Agent, Ball, ScoreBoard, Button
from utilities import get_clicked_cell, neighbor_hex, check_for_goal_or_corners, log_to_csv, determine_direction
from qlearning import QLearningAgent
from minimaxQLearning import MinimaxQLearningAgent
from beliefQLearning import BeliefBasedQLearningAgent
from state import State
import datetime

if os.path.exists("states.csv"):
    open("states.csv", 'w').close()

def main():
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Hex Soccer Game")

    backup_dir_q = 'C:\\Users\\baios\Desktop\\small_env_q_table_backups'
    backup_dir_minmax = 'C:\\Users\\baios\Desktop\\small_env_minimax_tables_backups'
    backup_dir_belief = 'C:\\Users\\baios\Desktop\\small_env_belief_tables_backups'
    os.makedirs(backup_dir_q, exist_ok=True)
    os.makedirs(backup_dir_minmax, exist_ok=True)
    os.makedirs(backup_dir_belief, exist_ok=True)
    next_save_time = datetime.datetime.now() + datetime.timedelta(hours=1)

    player = Player(5, 2)
    agent = Agent(1, 2)
    ball = Ball(3, 2)
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
    agent_q_agent = QLearningAgent(0.1, 0.9, MAX_EPSILON_Q, player, agent, ball, scoreboard, agent_type='agent')

    agent_minimax_agent = MinimaxQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)
    agent_belief_agent = BeliefBasedQLearningAgent(MAX_ALPHA, 0.9, MAX_EPSILON_MIN_MAX, player, agent, ball, scoreboard)

    player_q_agent.load_q_table('small_env_1_player_q_table.json')
    agent_q_agent.load_q_table('small_env_1_agent_q_table.json')

    if ENABLE_MINIMAX_Q_LEARNING_TRAINING:
        agent_minimax_agent.load_tables('small_env_1_agent_minimax_q_table.json', 'small_env_1_agent_minimax_pi_table.json', 'small_env_1_agent_minimax_v_table.json')
    if ENABLE_BELIEF_Q_LEARNING_TRAINING:
        agent_belief_agent.load_tables('small_env_1_agent_belief_q_table.json', 'small_env_1_agent_belief_v_table.json', 'small_env_1_agent_belief_table.json')

    running = True
    agent_take_turn = False
    deferred_agent_state = None
    while running:
        before_loop_state_for_logging_1 = State(player, agent, ball, scoreboard).get_state()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ENABLE_Q_LEARNING_TRAINING:
                    player_q_agent.save_q_table('small_env_1_player_q_table.json')
                    agent_q_agent.save_q_table('small_env_1_agent_q_table.json')
                if ENABLE_MINIMAX_Q_LEARNING_TRAINING:
                    agent_minimax_agent.save_tables('small_env_1_agent_minimax_q_table.json', 'small_env_1_agent_minimax_pi_table.json', 'small_env_1_agent_minimax_v_table.json')
                if ENABLE_BELIEF_Q_LEARNING_TRAINING:
                    agent_belief_agent.save_tables('small_env_1_agent_belief_q_table.json', 'small_env_1_agent_belief_v_table.json', 'small_env_1_agent_belief_table.json')
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

        if player.QLearning:
            current_player_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_player()
            player_action = player_q_agent.choose_action(current_player_state)
            player.perform_action(ball, player_action)
            next_player_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_player()
            player_reward = player_q_agent.calculate_reward(current_player_state, player_action, next_player_state)
            if ENABLE_Q_LEARNING_TRAINING:
                player_q_agent.update_q_table(current_player_state, player_action, player_reward, next_player_state)
            agent_take_turn = True

        next_player_state_for_agent_training = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()

        if deferred_agent_state and agent_take_turn and agent.MinMaxQLearning and ENABLE_MINIMAX_Q_LEARNING_TRAINING:
            previous_agent_state, agent_action = deferred_agent_state
            agent_reward = agent_minimax_agent.calculate_reward(previous_agent_state, agent_action, next_player_state_for_agent_training)
            if after_loop_state_for_logging_1 and deferred_agent_state and agent_take_turn:
                if after_loop_state_for_logging_1["ball_position"] in [(6,2), (6,0), (6,4)]:
                    agent_reward += 100
                elif after_loop_state_for_logging_1["ball_position"] in [(0,2), (0,0), (0,4)]:
                    agent_reward -= 100
            agent_minimax_agent.update_q_table(previous_agent_state, agent_action, player_action, agent_reward, next_player_state_for_agent_training)
            agent_minimax_agent.update_pi_table(previous_agent_state)
            agent_minimax_agent.update_v_table(previous_agent_state)

            if after_player_turn_state_for_logging != after_loop_state_for_logging_1:
                log_to_csv('Agent', after_player_turn_state_for_logging, agent_reward, agent_action)
                if after_loop_state_for_logging_1 != after_loop_state_for_logging_2 and agent_reward > 50:
                    log_to_csv('System', after_loop_state_for_logging_1, None, "Goal_reset")
                elif after_loop_state_for_logging_1 != after_loop_state_for_logging_2 and agent_reward < 50 and agent_reward > -50:
                    log_to_csv('System', after_loop_state_for_logging_1, None, "Corner_goal_reset")

            deferred_agent_state = None

        if deferred_agent_state and agent_take_turn and agent.TBRL and ENABLE_BELIEF_Q_LEARNING_TRAINING:
            previous_agent_state, agent_action = deferred_agent_state
            agent_reward = agent_belief_agent.calculate_reward(previous_agent_state, agent_action, next_player_state_for_agent_training)
            if after_loop_state_for_logging_1 and deferred_agent_state and agent_take_turn:
                if after_loop_state_for_logging_1["ball_position"] in [(6,2), (6,0), (6,4)]:
                    agent_reward += 100
                elif after_loop_state_for_logging_1["ball_position"] in [(0,2), (0,0), (0,4)]:
                    agent_reward -= 100
            agent_belief_agent.update_q_table(previous_agent_state, agent_action, player_action, agent_reward, next_player_state_for_agent_training)
            agent_belief_agent.update_v_table(previous_agent_state)

            if after_player_turn_state_for_logging != after_loop_state_for_logging_1:
                log_to_csv('Agent', after_player_turn_state_for_logging, agent_reward, agent_action)
                if after_loop_state_for_logging_1 != after_loop_state_for_logging_2 and agent_reward > 50:
                    log_to_csv('System', after_loop_state_for_logging_1, None, "Goal_reset")
                elif after_loop_state_for_logging_1 != after_loop_state_for_logging_2 and agent_reward < 50 and agent_reward > -50:
                    log_to_csv('System', after_loop_state_for_logging_1, None, "Corner_goal_reset")

            deferred_agent_state = None

        before_loop_state_for_logging_2 = State(player, agent, ball, scoreboard).get_state()
        check_for_goal_or_corners(player, agent, ball, scoreboard)
        redraw_game_state(player, agent, ball, scoreboard, screen, buttons)
        after_player_turn_state_for_logging = State(player, agent, ball, scoreboard).get_state()

        if agent_take_turn:
            if agent.QLearning:
                current_agent_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                agent_action = agent_q_agent.choose_action(current_agent_state)
                agent.perform_action(ball, agent_action)
                next_agent_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                agent_reward = agent_q_agent.calculate_reward(current_agent_state, agent_action, next_agent_state)
                if ENABLE_Q_LEARNING_TRAINING:
                    agent_q_agent.update_q_table(current_agent_state, agent_action, agent_reward, next_agent_state)
                agent_take_turn = False
            elif agent.MinMaxQLearning:
                current_agent_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                agent_action = agent_minimax_agent.choose_action(current_agent_state)
                deferred_agent_state = ({
                    'agent_position': current_agent_state["agent_position"],
                    'ball_position': current_agent_state["ball_position"],
                    'player_position': current_agent_state["player_position"]
                }, agent_action)
                agent.perform_action(ball, agent_action)
                agent_take_turn = False
            elif agent.TBRL:
                current_agent_state = State(player, agent, ball, scoreboard).get_qlearning_state_for_agent()
                agent_action = agent_belief_agent.choose_action(current_agent_state)
                deferred_agent_state = ({
                    'agent_position': current_agent_state["agent_position"],
                    'ball_position': current_agent_state["ball_position"],
                    'player_position': current_agent_state["player_position"]
                }, agent_action)
                agent.perform_action(ball, agent_action)
                agent_take_turn = False

        after_loop_state_for_logging_1 = State(player, agent, ball, scoreboard).get_state()
        check_for_goal_or_corners(player, agent, ball, scoreboard)
        redraw_game_state(player, agent, ball, scoreboard, screen, buttons)
        after_loop_state_for_logging_2 = State(player, agent, ball, scoreboard).get_state()

        if before_loop_state_for_logging_1 != before_loop_state_for_logging_2:
            log_to_csv('Player', before_loop_state_for_logging_1, player_reward, player_action)
            if before_loop_state_for_logging_2 != after_player_turn_state_for_logging and player_reward > 50:
                log_to_csv('System', before_loop_state_for_logging_2, None, "Goal_reset")
            elif before_loop_state_for_logging_2 != after_player_turn_state_for_logging and player_reward < 50 and player_reward > -50:
                log_to_csv('System', before_loop_state_for_logging_2, None, "Corner_goal_reset")

        current_time = datetime.datetime.now()
        if current_time >= next_save_time:
            if ENABLE_Q_LEARNING_TRAINING:
                QLearningAgent.save_q_tables_with_backup(player_q_agent, agent_q_agent, backup_dir_q)
            if ENABLE_MINIMAX_Q_LEARNING_TRAINING:
                MinimaxQLearningAgent.save_q_tables_with_backup(agent_minimax_agent, backup_dir_minmax)
            if ENABLE_BELIEF_Q_LEARNING_TRAINING:
                BeliefBasedQLearningAgent.save_q_tables_with_backup(agent_belief_agent, backup_dir_belief)
            next_save_time = current_time + datetime.timedelta(hours=1)

        if scoreboard.left_score + scoreboard.right_score == EPISODES_FOR_QUITING_PROGRAM and not INFINITE_EPISODES:
            if ENABLE_Q_LEARNING_TRAINING:
                QLearningAgent.save_q_tables_with_backup(player_q_agent, agent_q_agent, backup_dir_q)
            if ENABLE_MINIMAX_Q_LEARNING_TRAINING:
                MinimaxQLearningAgent.save_q_tables_with_backup(agent_minimax_agent, backup_dir_minmax)
            if ENABLE_BELIEF_Q_LEARNING_TRAINING:
                BeliefBasedQLearningAgent.save_q_tables_with_backup(agent_belief_agent, backup_dir_belief)

            pygame.quit()

    if ENABLE_Q_LEARNING_TRAINING:
        player_q_agent.save_q_table('small_env_1_player_q_table.json')
        agent_q_agent.save_q_table('small_env_1_agent_q_table.json')

    if ENABLE_MINIMAX_Q_LEARNING_TRAINING:
        agent_minimax_agent.save_tables('small_env_1_agent_minimax_q_table.json', 'small_env_1_agent_minimax_pi_table.json', 'small_env_1_agent_minimax_v_table.json')

    if ENABLE_BELIEF_Q_LEARNING_TRAINING:
        agent_belief_agent.save_tables('small_env_1_agent_belief_q_table.json', 'small_env_1_agent_belief_v_table.json', 'small_env_1_agent_belief_table.json')
    pygame.quit()

if __name__ == "__main__":
    main()

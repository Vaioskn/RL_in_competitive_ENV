import os
import json
import random
import datetime
from state import StateActionPair
from settings import CORNER_CELLS, EPISODES_FOR_TRAINING_Q, MIN_EPSILON_Q, MAX_EPSILON_Q, Q_LEARNING_WITHOUT_OPPONENT_AGENT, Q_LEARNING_WITHOUT_OPPONENT_PLAYER
from utilities import hex_center, manhattan_distance

class QLearningAgent:
    def __init__(self, learning_rate, discount_factor, epsilon, player, agent, ball, scoreboard, agent_type='agent'):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.agent_type = agent_type  # player or agent
        self.q_table = {}
        self.player = player
        self.agent = agent
        self.ball = ball
        self.scoreboard = scoreboard

    def load_q_table(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                self.q_table = json.load(file)
                self.q_table = {eval(key): value for key, value in self.q_table.items()}
        else:
            with open(filename, 'w') as file:
                json.dump({}, file)
            self.q_table = {}

        print(f"Loaded Q-table from {filename}. Current Q-table size: {len(self.q_table)}")

    def save_q_table(self, filename):
        with open(filename, 'w') as file:
            json.dump({str(key): value for key, value in self.q_table.items()}, file)

    def save_q_tables_with_backup(agent1, agent2, backup_dir):
        agent1.save_q_table('small_env_1_player_q_table.json')
        agent2.save_q_table('small_env_1_agent_q_table.json')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        player_backup_filename = os.path.join(backup_dir, f'small_env_1_player_q_table_{timestamp}.json')
        agent_backup_filename = os.path.join(backup_dir, f'small_env_1_agent_q_table_{timestamp}.json')
        agent1.save_q_table(player_backup_filename)
        agent2.save_q_table(agent_backup_filename)
        print(f'Q-tables backed up at {timestamp}')

    def update_q_table(self, state, action, reward, next_state):
        sap = StateActionPair(state, action).get()

        if self.agent_type == 'agent':
            possible_actions = self.agent.get_possible_actions(self.player, self.ball)
        else:
            possible_actions = self.player.get_possible_actions(self.agent, self.ball)

        next_sap_values = [self.q_table.get(StateActionPair(next_state, a).get(), 0) for a in possible_actions]
        max_future_q = max(next_sap_values) if next_sap_values else 0
        self.q_table[sap] = (1 - self.learning_rate) * self.q_table.get(sap, 0) + self.learning_rate * (reward + self.discount_factor * max_future_q)

    def calculate_reward(self, current_state, action, next_state):
        player_goal = (6, 2)
        agent_goal = (0, 2)

        player_corners = [(6,0),(6,4)]
        agent_corners = [(0,0),(0,4)]

        own_goal = agent_goal
        enemy_goal = player_goal
        current_entity_position = current_state['agent_position']
        next_entity_position = next_state['agent_position']
        own_corners = agent_corners
        enemy_corners = player_corners

        old_distance_to_ball = manhattan_distance(hex_center(*current_entity_position), hex_center(*current_state['ball_position']))
        new_distance_to_ball = manhattan_distance(hex_center(*next_entity_position), hex_center(*next_state['ball_position']))

        old_distance_to_enemy_goal = manhattan_distance(hex_center(*current_state['ball_position']), hex_center(*enemy_goal))
        new_distance_to_enemy_goal = manhattan_distance(hex_center(*next_state['ball_position']), hex_center(*enemy_goal))

        reward = -1 if new_distance_to_ball < old_distance_to_ball else -5

        if action.startswith('kick') and new_distance_to_enemy_goal < old_distance_to_enemy_goal:
            reward += 1

        if next_state['ball_position'] == enemy_goal:
            reward += 100
        elif next_state['ball_position'] == own_goal:
            reward -= 100

        if next_state['ball_position'] in CORNER_CELLS:
            reward -= 5

        return reward

    def choose_action(self, state):
        if self.agent_type == 'agent':
            possible_actions = self.agent.get_possible_actions(self.player, self.ball)
        else:
            possible_actions = self.player.get_possible_actions(self.agent, self.ball)

        if random.uniform(0, 1) < self.calculate_epsilon(self.scoreboard.left_score + self.scoreboard.right_score, EPISODES_FOR_TRAINING_Q, MIN_EPSILON_Q, MAX_EPSILON_Q):
            return random.choice(possible_actions)
        else:
            sap_values = {a: self.q_table.get(StateActionPair(state, a).get(), 0) for a in possible_actions}

            if sap_values and max(sap_values.values()) != 0:
                return max(sap_values, key=sap_values.get)
            else:
                if (self.agent_type == 'agent' and Q_LEARNING_WITHOUT_OPPONENT_AGENT) or (self.agent_type == 'player' and Q_LEARNING_WITHOUT_OPPONENT_PLAYER):
                    return random.choice(possible_actions)
                else:
                    return None 

    def get_state_action_pair(self, state, action):
        return StateActionPair(state, action).get()

    def calculate_epsilon(self, episode, max_episodes, min_epsilon, max_epsilon):
        decay_rate = (max_epsilon - min_epsilon) / max_episodes
        return max(min_epsilon, max_epsilon - decay_rate * episode)

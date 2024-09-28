class State:
    def __init__(self, player, agent, ball, score):
        self.agent = agent
        self.ball = ball
        self.player = player
        self.score = score

    def get_state(self):
        return {
            'agent_position': (self.agent.column, self.agent.row),
            'ball_position': (self.ball.column, self.ball.row),
            'player_position': (self.player.column, self.player.row),
            'score': (self.score.left_score, self.score.right_score)
        }

    def get_qlearning_state_for_player(self):
        return {
            'agent_position': (self.agent.column, self.agent.row),
            'ball_position': (self.ball.column, self.ball.row),
            'player_position': (self.player.column, self.player.row)
        }

    def get_qlearning_state_for_agent(self):
        return {
            'agent_position': (self.agent.column, self.agent.row),
            'ball_position': (self.ball.column, self.ball.row),
            'player_position': (self.player.column, self.player.row)
        }

class StateActionPair:
    def __init__(self, state, action):
        self.state_action_pair = (tuple(sorted(state.items())), action)

    def __repr__(self):
        return f"StateActionPair(state={self.state_action_pair[0]}, action='{self.state_action_pair[1]}')"

    def get(self):
        return self.state_action_pair
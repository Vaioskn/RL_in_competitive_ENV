
# Abstract

This thesis focuses on the development and study of a program where two agents play soccer in a deterministic environment. The agents are opponents and use reinforcement learning algorithms for decision-making, specifically the Q-Learning, Minimax-Q, and Belief-Q algorithms. The objective of the work is to evaluate the effectiveness of various reinforcement learning algorithms in a competitive environment where the agents have opposing goals. The methodology involves developing a simulated soccer environment where agents learn to play soccer by interacting with the environment and the opponent. The Q- Learning algorithm was used as the primary learning algorithm, while Minimax-Q and Belief-Q were introduced to study agent performance in more complex decisionmaking scenarios that require predicting the opponent's moves. The main steps taken include the initial implementation of the algorithms, adapting them to the soccer environment, and conducting experiments to evaluate the strategies developed. The results showed that each algorithm has different strengths and weaknesses depending on the opponent's strategy.

# 1. Introduction

Reinforcement Learning (RL) has emerged as one of the most dynamic fields in artificial intelligence, offering tools for solving complex decision-making problems. In this study, three reinforcement learning algorithms—Q-Learning, Minimax-Q, and Belief-Q—are examined within a competitive soccer game environment. The agents, who are opponents, make decisions in a deterministic environment with the aim of achieving opposing goals. Markov Decision Processes (MDPs) form the foundation for modeling decision-making in environments where outcomes depend both on the actions of the agent and on randomness due to the agent's willingness to explore the environment. These processes include states, actions, transition functions between states, and reward functions, allowing agents to learn the best strategy (policy) to maximize their performance. The Q-Learning algorithm is widely known for its ability to learn the optimal policy through iterative updates of Q-values, which correspond to the estimated utility of taking specific actions in given states. On the other hand, the Minimax-Q and BeliefQ algorithms introduce more advanced techniques that take into account possible strategies of the opponent, either by finding the optimal strategy under conditions of complete opposition (Minimax-Q) or by using probabilities for the opponent's actions (Belief-Q). This research focuses on evaluating the performance of these algorithms in a complex environment with a large number of states. To address the complexity of the environment, it is proposed to divide it into two smaller sub-environments, where the algorithms are trained separately. The study then evaluates the algorithms' performance in both the smaller environments and the original larger environment, analyzing the strategies developed and the results obtained from their comparison. The structure of the thesis is as follows: The first section provides a general overview of MDPs and their role in reinforcement learning. The second section analyzes the QLearning, Minimax-Q, and Belief-Q algorithms, along with their implementation methods. The third section describes the specific environment and the complexity issue it presents. In the following sections, the approach to dividing the environment, the training of the algorithms, the results of their comparison, and, finally, the evaluation of their performance in the original, larger environment are presented.

# 2. Theoretical Background And Reinforcement Learning Algorithms

## 2.1 Markov Decision Processes (Mdps)

A Markov Decision Process (MDP), also known as "stochastic dynamic programming" or a "stochastic control problem," is a model for sequential decision-making where outcomes are uncertain. [1] The term "Markov Decision Process" comes from its connection to Markov chains, a mathematical concept developed by Russian mathematician Andrey Markov. A Markov chain is a sequence of states where the probability of transitioning to the next state depends only on the current state, not on the sequence of events that preceded it. This property is known as the Markov property or memorylessness. [2]

### Markov Decision Process (MDP)

A Markov Decision Process is defined as a tuple $`(S, A, P_a, R_a)`$ where:

- $`S`$ is a set of states called the state space. The state space can be discrete or continuous, such as the set of real numbers.
- $`A`$ is a set of actions called the action space (alternatively $`A_s`$ is the set of actions available from state $`s`$).
- $`P_a(s, s')`$ is the probability that action $`a`$ in state $`s`$ at time $`t`$ will lead to state $`s'`$ at time $`t+1`$.
- $`R_a(s, s')`$ is the immediate reward (or expected immediate reward) received after transitioning from state $`s`$ to state $`s'`$ due to action $`a`$.

Additionally, there is at least one initial state $`S_0`$ and possibly a terminal state $`S_{Goal}`$.

The objective in a Markov Decision Process is to find a good "policy" for the decision-maker, meaning a function $`\pi`$ that specifies the action $`\pi(s)`$ to be taken when in state $`s`$. Once an MDP is paired with a policy in this way, the resulting state-action combinations behave like a Markov chain, as the action chosen in state $`s`$ is entirely determined by $`\pi(s)`$, assuming the policy is deterministic [2].

To manage randomness, we maximize the expected sum of rewards. Generally [3]:

$`\pi^* = \arg\max_{\pi} \mathbb{E} \left[ \sum_{t \geq 0} \gamma^t r_t \mid s_0 \sim p(s_0), a_t \sim \pi(\cdot \mid s_t), s_{t+1} \sim p(\cdot \mid s_t, a_t) \right]`$

Where:

- $`\pi^*`$: The optimal policy that maximizes the expected total reward.
- $`\gamma^t`$: The discount factor raised to the power of $`t`$, with $`0 \leq \gamma \leq 1`$. It determines the present value of future rewards. A lower $`\gamma`$ makes future rewards less important.
- $`r_t`$: The reward received at time step $`t`$.
- $`\mid \pi`$: Given policy $`\pi`$, it shows that the expectation is conditional on the sequence of policy $`\pi`$.
- $s_0 \sim p(s_0)$: The initial state $s_0$ which follows the distribution $p(s_0)$.
- $a_t \sim \pi(\cdot \mid s_t)$: The action $a_t$ at time $t$, determined by the policy $\pi$ given the current state $s_t$.
- $s_{t+1} \sim p(\cdot \mid s_t, a_t)$: The next state $s_{t+1}$ which is drawn from the transition probability distribution $p(\cdot \mid s_t, a_t)$, which depends on the current state $s_t$ and action $a_t$.

To solve MDPs, we use variations of the Bellman equation. Specifically, through the Bellman equation, we can:

1. **Compute optimal state values** using the value iteration method, where the Bellman equation characterizes the optimal values [3]:

   
   $`V^*(s) = \max_a \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V^*(s') \right]`$
   

   Where:

   - $V^*(s)$: The optimal value of state $s$, i.e., the maximum expected total reward that can be obtained from state $s$.
   - $T(s,a,s')$: The probability of transitioning from state $s$ to state $s'$ when action $a$ is taken.
   - $R(s,a,s')$: The reward received when action $a$ is taken in state $s$ and the transition to state $s'$ occurs.
   - $\gamma$: The discount factor, determining the present value of future rewards.
   - $V^*(s')$: The optimal value of the next state $s'$.

   Value iteration computes these values iteratively:

   
   $`V_{k+1}(s) \leftarrow \max_a \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right]`$
   

2. **Compute optimal state values via policy iteration**, where the Bellman equation is used to evaluate both the states and the $i$-th policy [3]:

   
   $`V_{k+1}^{\pi_i}(s) = \sum_{s'} T(s, \pi_i(s), s') \left[ R(s, \pi_i(s), s') + \gamma V_k^{\pi_i}(s') \right]`$
   

   Where:

   - $V_k^{\pi_i}(s)$: The estimated value of state $s$ at the $k$-th iteration under policy $\pi_i$.
   - $T(s,\pi_i(s),s')$: This is the probability of transitioning from state $s$ to state $s'$ when following the action proposed by policy $\pi_i$ in state $s$.
   - $R(s,\pi_i(s),s')$: This is the reward received when the action chosen by policy $\pi_i$ in state $s$ leads to state $s'$.
   - $V_k^{\pi_i}(s')$: This represents the estimated value of state $s'$ at the $k$-th iteration of policy evaluation under policy $\pi_i$.

   This equation allows us to estimate how good the policy $`\pi_i`$ is in each state $`s`$. We then use the results of this evaluation to improve the policy with the following equation:

   $`\pi_{i+1}(s) = \arg\max_a \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V^{\pi_i}(s') \right]`$

   Where:

   - $`\pi_{i+1}(s)`$: This is the updated policy for state $`s`$, which results from optimizing the expected total reward based on the current value function estimates.
   - $`V^{\pi_i}(s')`$: This is the value of the next state $`s'`$ under policy $`\pi_i`$, obtained from the previous iterative update.

   Using this equation, we choose the best action for each state, updating the policy from $`\pi_i`$ to $`\pi_{i+1}`$.

### How good is a state?

The value function for state $`s`$ is the expected cumulative reward from following the policy starting from state $`s`$ [3]:

$`V^{\pi}(s) = \mathbb{E} \left[ \sum_{t \geq 0} \gamma^t r_t \mid s_0 = s, \pi \right]`$

Where:

- $`V^{\pi}(s)`$: The value function for state $`s`$ under policy $`\pi`$, representing the expected total reward starting from state $`s`$ and following policy $`\pi`$.
- $`s_0`$: The initial state, which is equal to state $`s`$, indicating that the expected total reward starts from state $`s`$.
- $`\pi`$: The policy, i.e., the rule that determines which action will be executed in each state.

### How good is a state-action pair?

The Q-value function for state $`s`$ and action $`a`$ is the expected cumulative reward from taking action $`a`$ in state $`s`$ and then following the policy [3]:

$`Q^{\pi}(s, a) = \mathbb{E} \left[ \sum_{t \geq 0} \gamma^t r_t \mid s_0 = s, a_0 = a, \pi \right]`$

Where:

- $`Q^{\pi}(s, a)`$: The action-value function for state $`s`$ and action $`a`$ under policy $`\pi`$. It represents the expected total reward starting from state $`s`$, taking action $`a`$, and then following policy $`\pi`$.
- $`a_0`$: The initial action, which is equal to action $`a`$, indicating that the policy starts by taking action $`a`$ from state $`s`$.


Having mentioned the term of state-action pairs, we can extend and further analyze the first of the implemented reinforcement learning algorithms, the Q - Learning algorithm.

### 2.2 Q-Learning

The Q-Learning algorithm is based on the concept of the Q-value, which expresses the quality of an action when executed in a particular state. The Q-value of a state-action pair is denoted as $`Q(s, a)`$, representing the expected cumulative reward that the agent will receive if it takes action $`a`$ in state $`s`$ and then follows the optimal policy.

The agent starts with a Q-value table where all Q-values are typically initialized to zero or some small random number. This table is updated gradually as the agent explores the environment.

At each step, the agent finds itself in a specific state $`s`$ and must select an action $`a`$. To choose the action, the algorithm uses an exploration-exploitation strategy, such as epsilon-greedy. The agent sometimes selects the action with the highest Q-value (exploitation) and other times selects a random action to explore new possibilities (exploration).

After executing action $`a`$ in state $`s`$, the agent transitions to a new state $`s'`$ and receives a reward $`r`$.

It then updates the Q-value for the state-action pair $`(s, a)`$ using the following formula [4] [5]:

$`
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left[ r(s, a, s') + \gamma \max_{a'} Q(s', a') \right]
`$

Where:

- $`\alpha`$ is the learning rate, which determines how quickly the Q-value is adjusted.
- $`r(s, a, s')`$ is the immediate reward received in state $`s`$ after performing action $`a`$ to transition to state $`s'`$.
- $`\max_{a'} Q(s', a')`$ is the maximum expected future reward from the next state $`s'`$.

The agent repeats this process for many episodes, continuously updating the Q-value table as it learns the optimal actions for each state. The goal is to learn the optimal policy, i.e., the strategy that selects the actions with the highest Q-values in each state, thereby maximizing long-term rewards.

### 2.3 Minimax-Q

The Minimax-Q algorithm is an extension of the classical Q-Learning, designed for environments where competition exists between two or more agents with conflicting interests. The main objective of Minimax-Q is to compute the optimal policy for an agent, considering that the opposing agent will choose actions that minimize the first agent's gains.

The algorithm is based on zero-sum game theory, where the gains of one agent are exactly the losses of the other. Although the environment described here is a non-zero-sum game, Minimax-Q can still be applied since the agents are opponents. To find the optimal strategy, the agent solves a linear programming problem that maximizes the minimum reward it can receive, taking the opponent's actions into account.

#### Initialization:

For all states $`s`$ in the set of states $`S`$, for all actions $`a`$ in the set of actions $`A`$, and for all opponent actions $`o`$ in the set of possible opponent actions $`O`$ [6]:

$`
Q[s, a, o] := 1
`$

For all states $`s`$ in the state set $`S`$:

$`
V[s] := 1
`$

For all states $`s`$ in the set of states $`S`$ and for all actions $`a`$ in the set of actions $`A`$:

$`
\pi[s, a] := \frac{1}{|A|}
`$

(At the beginning, the probability of selecting each action is equal for all actions.)

Learning rate $\alpha$ initialization:

$`
\alpha := 1.0
`$

#### Action Selection in State $`s`$:

- With probability $`explor`$, the agent selects an action randomly from the set of possible actions (exploration).
- Otherwise, the agent selects the action $`a`$ with probability $`\pi[s, a]`$ (exploitation). In this implementation, to maintain the deterministic nature of the environment, action $`a`$ is chosen with probability $`1.0`$, and it corresponds to the action with the highest probability according to the distribution $`\pi[s, \cdot]`$.


In this implementation, to maintain the deterministic nature of the environment, action a is chosen with probability 1.0, and it corresponds to the action with the highest probability according to the distribution π[s, ⋅].

#### Learning:

After the agent receives a reward $`r(s,a,o,s')`$ for transitioning from state $`s`$ to state $`s'`$ via action $`a`$ and the opponent's action $`o`$, the Q-value is updated [6]:

$`
Q[s, a, o] := (1 - \alpha) \cdot Q[s, a, o] + \alpha \cdot \left( r(s, a, o, s') + \gamma \cdot V[s'] \right)
`$

Where:

- $`Q[s,a,o]`$: is the action-value function for state $`s`$, action $`a`$, and the opponent's response $`o`$. It evaluates how good it is to take action $`a`$ in state $`s`$, considering the opponent's response.
- $`r(s, a, o, s')`$: is the immediate reward received for transitioning from state $`s`$ to state $`s'`$ by performing action $`a`$ and the opponent's response $`o`$.
- $`V[s']`$: is the value function for the next state $`s'`$, representing the expected total reward from state $`s'`$ onward.

#### Finding the Optimal Policy through Linear Programming:

The agent uses linear programming to find the policy $`\pi[s, \cdot]`$ that maximizes the minimum possible Q-value, considering all possible opponent actions [6]:

$`
\pi[s, \cdot] := \arg \max_{\pi[s, \cdot]} \min_{o'} \left\{ \sum_{a'} \pi[s, a'] \cdot Q[s, a', o'] \right\}
`$

Where:

- $`\pi[s, \cdot]`$: is the policy for state $`s`$. The policy $`\pi`$ defines the probability of executing each action in state $`s`$.
- $`\pi[s,a']`$: is the probability of executing action $`a'`$ from state $`s`$ under policy $`\pi`$.

Here, $`\arg \max_{\pi[s, \cdot]}`$ selects the policy $`\pi[s, \cdot]`$ that maximizes the expression. The updated policy $`\pi`$ must maximize the minimum expected benefit, considering the probabilities of various actions. The $`\min_{o'}`$ operator chooses the least favorable opponent action $`o'`$ (i.e., the worst-case scenario for the agent).

This process ensures that the agent chooses a policy that is as safe as possible against the worst possible actions from the opponent.

#### Updating the Value $`V`$ [6]:

$`
V[s] := \min_{o'} \left( \sum_{a'} \pi[s, a'] \cdot Q[s, a', o'] \right)
`$

Where:

- $`\pi[s,a']`$: is the probability of executing action $`a'`$ from state $`s`$ under policy $`\pi`$.

Here, the $`\min_{o'}`$ operator selects the minimum value of the expression that follows, taking into account all possible responses $`o'o'`$ from the opponent. This minimization ensures that the agent prepares for the worst possible outcome from the opponent.

This minimization represents the choice of the worst-case scenario (i.e., the least favorable action of the opponent for the Minimax-Q agent).

#### Updating the Learning Rate:

$`
\alpha := \alpha \cdot decay
`$

While Q-Learning computes the policy based on the expected reward in a static environment, Minimax-Q computes the policy by considering the opponent’s potential actions that would cause the most harm. This makes Minimax-Q more suitable for competitive environments where there is a direct conflict of interests between agents.

Minimax-Q requires solving linear programming problems to compute the policy in each state. This is essential to ensure that the agent’s policy is resilient against the possible harmful actions of the opponent. In contrast, Q-Learning simply selects the action with the highest Q-value without considering the opponent’s strategy.

### 2.4 Belief-Q

The **Belief-Q** algorithm is a variation of Q-Learning, adapted for environments where agents must account for uncertainty regarding the strategies or actions of their opponents. This algorithm allows the agent to form **beliefs** about the possible actions of their opponents and update those beliefs as new information is received.

**Belief-Q** aims to optimize the agent’s strategy based on the potential reactions of opponents by combining Q-values with the probabilities that opponents will choose certain actions.

As in classical Q-Learning, we initialize a Q-value table for all states $`s`$ and actions $`a`$. Simultaneously, we initialize the beliefs regarding the actions of the opponent. Specifically, at the start, for each state and action of the agent, we create a probability table that represents our beliefs about which actions the opponent might choose. Initially, these beliefs are set to equal values, indicating no preference for any of the opponent's actions (i.e., all opponent actions are considered equally probable). As the agent interacts with the environment, these beliefs are updated based on observations of the opponent's actions, adjusting the probabilities of different opponent actions. Specifically, after each learning step, the probability of each opponent action in the belief table is updated to incorporate the new information.

The agent selects an action $`a`$ based on the policy derived from the updated Q-values and the beliefs about the opponent’s actions. For each possible action, the agent consults the Q-value table (Q-table) to find the corresponding Q-value. This value depends on the agent’s beliefs about the opponent’s possible actions. These beliefs determine the probability that the opponent will choose a particular action.
The agent multiplies the Q-value for each combination of its own action and the opponent's action by the probability corresponding to the opponent's action, as expressed by the agent's beliefs. The result is a weighted sum of Q-values, where each Q-value for an opponent's action is weighted by the probability that the agent believes the opponent will choose that action. This process produces an overall expected value for the agent’s action, taking into account the different opponent responses and their probabilities. The agent then selects the action that maximizes this expected value, attempting to find the best possible strategy based on its beliefs about the opponent’s actions.

After the agent receives a reward $`r(s_t, a_t)`$ for transitioning from state $`s_t`$ to state $`s_{t+1}`$ by executing action $`a`$, the Q-value is updated [7]:

$`
Q_{t+1}(s_t, a_t) \leftarrow (1 - \alpha) Q_t(s_t, a_t) + \alpha \left[ r(s_t, a_t, s_{t+1}) + \gamma V_t(s_{t+1}) \right]
`$

Where:

- $`Q_{t+1}(s_t, a_t)`$: is the updated action-value function Q for state $`s_t`$ and action $`a_t`$ at time step $`t+1`$.

#### Updating the Value Function $`V`$ Based on Beliefs [7]:

$`
V_t(s) \leftarrow \max_{a_i} \left[ \sum_{a_{-i} \subseteq A_{-i}} Q_t(s, (a_i, a_{-i})) \cdot Pr_i(a_{-i}) \right]
`$

Where:

- $`a_i`$: represents the agent’s action.
- $`a_{-i}`$: represents the opponent’s actions.
- $`Pr_i(a_{-i})`$: is the agent’s belief about the probability that the opponent will choose action $`a_{-i}`$.
- $`Q_t(s, (a_i, a_{-i}))`$: is the action-value function (Q-value) for state $`s`$ at time $`t`$, when agent $`i`$ selects action $`a_i`$ and the opponent selects action $`a_{-i}`$.


# 3. Design Of The Environment And Handling State Complexity

3.1 **Description of the Environment**

![Image_1.PNG](ASSETS/Image_1.PNG)

The **Full Environment (FE)**, or complete environment (Image 1), consists of a hexagonal grid where each cell represents a possible position for an agent or the ball. Each agent tries to score a goal in the opponent’s goal. The goals are located at the two ends of the grid, to the left and right, marked with red hexagons. The agent located on the right side, represented by a circle, will be referred to as "Player" from this point on (since the user can control its movements), while the agent on the left will be called "Agent." Neither the player nor the agent can move into the goal cells. No entity (Player, Agent, or ball) can occupy or move into a cell already occupied by another entity. This is a variation of the environment described in [4].

No entity can move outside the grid. If the ball reaches the left goal, one point is added to the score on the right side (as it is considered a point for the Player). Similarly, if the ball reaches the right goal, a point is added to the left side of the score. If the ball enters a goal or a corner, the environment resets to its initial state (Image 1). The Player takes the first move at the start of the simulation, while after each reset, the first move is given to the one who did *not* make the last move before the reset (i.e., the one who conceded the goal or did not cause the ball to go to a corner). The buttons below the grid define the logic for selecting the corresponding agent’s movement.

The movements of an agent depend on its current state. For example, if an agent is far from the ball and far from the boundaries of the grid, the available movements include six possible directions (light blue cells).

![Image_2.PNG](ASSETS/Image_2.PNG)

Available actions are:
- Move_up_left
- Move_up_right
- Move_left
- Move_right
- Move_down_left
- Move_down_right

![Image_3.PNG](ASSETS/Image_3.PNG)

In case an agent is close to the ball and away from the grid boundaries, the available actions are three kick actions (yellow cells) and five move actions (blue cells)

Available actions are:
- Move_up_left
- Move_up_right
- Move_right
- Move_down_left
- Move_down_right
- Kick_up_left
- Kick_left
- Kick_down_left

And they change depending on the angle of the agent and the ball accordingly.

Of course, each agent does not have the ability to kick the ball or move to a position outside the boundaries of the grid:
































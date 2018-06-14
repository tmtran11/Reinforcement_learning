# Reinforcement learning algorithm
# Self-study
# Under instruction of Lazy Programmer Inc. https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python/

# Featured project: tic-tac-toe, an AI game-playing agent

Implement algorithms and statistical structure of Reinforcement Learning from scratch.
https://github.com/tmtran11/Reinforcement_learning/blob/master/RL/tick-tac-toe.py

*Class Environment: The tic-tac-toe board
- update_state(self): Using numpy array of 1 and -1 for each player and 0 for blank space, this method use special arithmetic value to - determine winning state
- draw(self): print out the board in special format

*Class Player: Artificial Intelligence agent, with special learning rate(y)
- update_self(self,action): Update when self have an action
- update_oppponent(self,action): Update when opponent have an action
- update_value(self, reward): Update value using Gradient Descent, each state update base on the previous state, and the rate of updating is present be given learning rate of 0.3(Experimental result)
- update_history(self, state): Update each time there is an action taken by self
- enum_state(self): encoding the state in trinary
- take_action(self, state, actions, draw_verbose = False): Using Upper Confidence Bound (UCB) algorithm to pick best action. UCB algorithm balance between exploitation and exploration, and it is at the central reinforcement learning
- reset(self): reset all attribute, except values of states 

*Class Human: Human Player with same method as Player; however, there is no calculation involve, also action come from direct input
- update_self(self,action)
- update_oppponent(self,action)
- update_value(self, reward):
- update_history(self, state):
- enum_state(self):
- take_action(self, state, actions, draw_verbose = False)
- reset(self):

*game():
- Create an object Environment and take in 3 players as parameter.
- Player vs. Player: Create 2 AI players, perform reinforcement learning on them by let them play 10000 games together to set up statistical value for states. 

* Human vs. Player: Testing phase. Human against trained AI player.

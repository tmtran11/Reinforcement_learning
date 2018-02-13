#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:20:02 2018
Policy iteration and value iteration
@author: cc-user
"""
# Create class Grid, np.array, initialize random policy, end_state, policy
# Create a movement def, draw_value def
# Do you want it to be deterministic, or stochastic but with nice ratio?
# if do policy iteration: go by policy to cal stuff, one step ahead and then go through everything, readjust policy
# if do value iteration: find argmax of value around

import numpy as np
all_move = [1,2,3,4]
def move(i, j, n, max_i, max_j, walls):
    if n==1:
        if i>0 and (i-1, j) not in walls:
            return i-1, j # up
    if n==2:
        if j<max_j and (i, j+1) not in walls:
            return i, j+1 # right
    if n==3:
        if i<max_i and (i+1, j) not in walls:
            return i+1, j # down
    if n==4:
        if j>0 and (i, j-1) not in walls:
            return i, j-1 # left
    return i, j
        
class Grid:
    def __init__(self, size, end_state, walls, reward): 
        self.walls = walls
        self.size = size # a tuple
        self.value = np.zeros(size)
        self.policy = np.zeros(size)
        self.all_states = {}
        for i in range(size[0]):
            for j in range(size[1]):
                if (i,j) not in walls:
                    self.policy[i, j] = np.random.randint(1,5)
                    self.all_states[(i,j)] = reward
        for x in end_state:
            self.all_states[x] = end_state[x]
    def draw_value(self, show = True):
        print('-----------------------------')
        for i in range(self.size[0]):
            value = ''
            for j in range(self.size[1]):
                value = value + str(self.value[i,j])
                value = value + '\t'
            print(value)
        print('-----------------------------')
    def draw_policy(self, show = True):
        print('-----------------------------')
        for i in range(self.size[0]):
            policy = ''
            for j in range(self.size[1]):
                policy = policy + str(self.policy[i,j])
                policy = policy + '\t'
            print(policy)
        print('-----------------------------')
class Robot:
    def __init__(self, lr):
        self.lr = lr
        self.x = 0
        self.y = 0
    def reset_pos(self, grid):
        self.x = np.random.randint(0, grid.size[0])
        self.y = np.random.randint(0, grid.size[1])
    def update_value_p(self, value, policy, all_states, size, walls):
        enough = False
        error = 0.0000016
        count = 0
        # how to make the iteration random? how about sometime randomly vist other node?
        # no worry if there is some space that is not visited. why, because there will be minus and stuff
        # we should just randomly iterate the value. oh god, such a pain
        while not enough:
            # remember to loop through
            all_states_keys = all_states.keys()
            np.random.shuffle(all_states_keys)
            enough = True
            count += 1
            print(count)
            for state in all_states_keys:
                i, j = state[0], state[1]
                x, y = move(i, j, policy[i, j], size[0]-1, size[1]-1, walls)
                x1, y1 = move(i, j, all_move[int(policy[i, j])-2], size[0]-1, size[1]-1, walls)
                x2, y2 = move(i, j, all_move[int(policy[i, j]-4)], size[0]-1, size[1]-1, walls)
                old_value = value[i,j]
                new_value = all_states[(i,j)] + self.lr*(0.8*value[x,y]+0.2*value[x1,y1]+0.2*value[x2,y2])
                value[i,j] = new_value
                if abs(old_value-new_value)>error:
                    enough = False
            for state in all_states:
                i, j = state[0], state[1]
                max_val = -1000.0
                max_action = 0
                if i>0 and value[i-1,j]>max_val and value[i-1,j]>value[i, j] and (i-1,j) in all_states:
                    max_val = value[i-1,j]
                    max_action = 1
                if j>0 and value[i,j-1]>max_val and value[i,j-1]>value[i, j] and (i,j-1) in all_states:
                    max_val = value[i,j-1]
                    max_action = 4
                if i<len(value)-1 and value[i+1,j]>max_val and value[i+1,j]>value[i, j] and (i+1,j) in all_states:
                    max_val = value[i+1,j]
                    max_action = 3
                if j<len(value[0])-1 and value[i,j+1]>max_val and value[i,j+1]>value[i, j] and (i,j+1) in all_states:
                    max_val = value[i,j+1]
                    max_action = 2
                policy[(i,j)] = max_action
            # value = reward + y*sum(prob.max(action))
            # stop if new value - old value close to an epsilon
            # In_loop, after that iterate through all the  policy by check the surrounded value.
            # pick random from max
        return value, policy
    def update_value_v(self, value, policy, all_states, size, walls):
        error = 0.0000016
        enough = False
        count = 0
        while not enough:
            all_states_keys = all_states.keys()
            np.random.shuffle(all_states_keys)
            enough = True
            count += 1
            print(count)
            for state in all_states_keys:
                i, j = state[0], state[1]
                x1, y1 = move(i, j, 1, size[0]-1, size[1]-1, walls)
                x2, y2 = move(i, j, 2, size[0]-1, size[1]-1, walls)
                x3, y3 = move(i, j, 3, size[0]-1, size[1]-1, walls)
                x4, y4 = move(i, j, 4, size[0]-1, size[1]-1, walls)
                old_value = value[i,j]
                new_value_1 = all_states[(i,j)] + self.lr*(0.8*value[x1,y1]+0.2*value[x2,y2]+0.2*value[x4,y4])
                new_value_2 = all_states[(i,j)] + self.lr*(0.8*value[x2,y2]+0.2*value[x1,y1]+0.2*value[x3,y3])
                new_value_3 = all_states[(i,j)] + self.lr*(0.8*value[x3,y3]+0.2*value[x2,y2]+0.2*value[x4,y4])
                new_value_4 = all_states[(i,j)] + self.lr*(0.8*value[x4,y4]+0.2*value[x1,y1]+0.2*value[x3,y3])
                if abs(old_value-max([new_value_1, new_value_2, new_value_3, new_value_4]))>error:
                    enough = False
                value[i,j] = max([new_value_1, new_value_2, new_value_3, new_value_4])
        for state in all_states:
            i, j = state[0], state[1]
            max_val = -1000.0
            max_action = 0
            if i>0 and value[i-1,j]>max_val and value[i-1,j]>value[i, j] and (i-1,j) in all_states:
                max_val = value[i-1,j]
                max_action = 1
            if j>0 and value[i,j-1]>max_val and value[i,j-1]>value[i, j] and (i,j-1) in all_states:
                max_val = value[i,j-1]
                max_action = 4
            if i<len(value)-1 and value[i+1,j]>max_val and value[i+1,j]>value[i, j] and (i+1,j) in all_states:
                max_val = value[i+1,j]
                max_action = 3
            if j<len(value[0])-1 and value[i,j+1]>max_val and value[i,j+1]>value[i, j] and (i,j+1) in all_states:
                max_val = value[i,j+1]
                max_action = 2
            policy[(i,j)] = max_action
            # check all action.
            # each action have probablity of others action. calculate those
            # check if new value - old value is close to the epsilon
            # after, re write the policy.
        return value, policy
robot = Robot(0.01)
size = (3, 4)
end_states = {}
end_states[(0,3)] = 1.0
end_states[(1,3)] = -1.0
reward = -0.2
walls = [(1,1)]
grid = Grid(size, end_state, walls, reward)
grid.value, grid.policy = robot.update_value_p(grid.value, grid.policy, grid.all_states, size, walls)
grid.draw_value()
grid.draw_policy()
robot = Robot(0.01)
size = (3, 4)
end_state = {}
end_state[(0,3)] = 1.0
end_state[(1,3)] = -1.0
reward = -0.2
walls = [(1,2)]
policy = grid.policy
grid = Grid(size, end_state, walls, reward)
grid.value, grid.policy = robot.update_value_v(grid.value, grid.policy, grid.all_states, size, walls)
grid.draw_value()
grid.draw_policy()
x = 0.1 + 0.006



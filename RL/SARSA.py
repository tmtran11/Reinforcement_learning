#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:16:04 2018
State-action-reward-state-action-reward, Markov Decision Process
Grid problem
@author: cc-user
"""
# t: time, update like real time
# epsilon-greedy = 1/t, else pick argmax(Q[s])
# alpha decay by count
# use one policy first, policy update once

import numpy as np
from matplotlib import pyplot as plt
class Grid():
    def __init__(self, size, walls, end_states):
        self.size = size
        self.walls = walls
        self.end_states = end_states
        self.all_states = {}
        for i in range(size[0]):
            for j in range(size[1]):
                if (i,j) not in walls:
                    self.all_states[(i,j)] = -0.1
        for x in end_states:
            self.all_states[x] = end_states[x]
        self.game_over = False
    def action(self, a):
        p = np.random.randn()
        if p > 0.5:
            return a
        else:
            n = np.random.randint(1,5)
            return n
    def move(self, a, i, j):
        n = (i,j)
        reward = 0.0
        if a==1:
            if i>0 and (i-1, j) not in self.walls:
                n = (i-1, j) # up
        if a==2:
            if j<self.size[1]-1 and (i, j+1) not in self.walls:
                n = (i, j+1) # right
        if a==3:
            if i<self.size[0]-1 and (i+1, j) not in self.walls:
                n = (i+1, j) # down
        if a==4:
            if j>0 and (i, j-1) not in self.walls:
                n = (i, j-1) # left
        reward = self.all_states[n]
        if n in self.end_states:
            self.game_over = True
        return n[0], n[1], a, reward
    def reset(self):
        self.game_over = False
# initiate grid
size = (3, 4)
end_states = {}
end_states[(0,3)] = 1.0
end_states[(1,3)] = -1.0
walls = [(1,1)]
grid = Grid(size, walls, end_states)
    
def draw_policy(policy, show = True):
    print('-----------------------------')
    for i in range(size[0]):
        p = ''
        for j in range(size[1]):
            if (i,j) in policy:
                p = p + str(policy[(i,j)])
                p = p + '\t'
            else:
                p = p + str(0)
                p = p + '\t'
        print(p)
    print('-----------------------------')
        
def eval_policy(policy, V, all_states): 
    for s in all_states:
        max_value = -100000000000.0
        max_action = 0
        for a in range(1,5):
            if V[s][a][0] > max_value:
                max_value = V[s][a][0]
                max_action = a
        policy[s] = max_action
    return policy

def max_dict(Q, s):
    max_value = -100000.0
    max_action = 0
    for a in Q[s]:
        if Q[s][a][0]>max_value:
            max_value = Q[s][a][0]
            max_action = a
    return max_action

def game(lr, ALPHA):
    deltas = []
    policy = {}
    for i in range(size[0]):
        for j in range(size[1]):
            if (i,j) not in walls:
                policy[(i, j)] = np.random.randint(1,5)
    Q = {}
    for s in grid.all_states:
        Q[s] = {}
        for a in range(1,5):
            Q[s][a] = [0,1.0] # value, count
    t = 1.0
    for i in range(10000):
        print(i)
        if i%100==0: t += 0.01
        E = 0.5/t
        x, y = 0,0
        p = np.random.randn()
        if p > E:
            nx, ny, a, reward = grid.move(grid.action(max_dict(Q,(x,y))), x, y)
        else:
            nx, ny, a, reward = grid.move(grid.action(np.random.randint(1,5)), x,y)
        # x, y, a
        while not grid.game_over:
            p = np.random.randn()
            if p > E:
                nnx, nny, na, reward = grid.move(grid.action(max_dict(Q,(nx,ny))), nx, ny)
            else:
                nnx, nny, na, reward = grid.move(grid.action(np.random.randint(1,5)), nx,ny)
            alpha = ALPHA/(Q[(x,y)][a][1])
            # fix something with the order of action
            Q[(x,y)][a][0] = Q[(x,y)][a][0] + alpha*(reward + lr*Q[(nx,ny)][na][0] - Q[(x,y)][a][0])
            Q[(x,y)][a][1] = Q[(x,y)][a][1] + 0.005
            deltas.append(abs(alpha*(reward + lr*Q[(nx,ny)][na][0] - Q[(x,y)][a][0])))
            x, y, a = nx, ny, na
            nx, ny = nnx, nny
        grid.reset()
    policy = eval_policy(policy, Q, grid.all_states)
    draw_policy(policy)
    return deltas

deltas = game(0.9, 0.1)
plt.plot(deltas)
plt.show()

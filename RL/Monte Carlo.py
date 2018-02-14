#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:51:43 2018
Monte Carlo
Grid Problem
@author: cc-user
"""
# Windy (0.5 a), negative grid (-0.2), with game_over, move, init a starting point
# in grid, store step and reward altogther
# only need a grid class
# Do 2000 episode
# Move until game_over, return value and reward, reward -100 and end game if end up in the same pos
# Go backward to trace each value and update, if value updated †hen pass
# Use epsilon greedy when decide follow policy or not
# Update policy until E, update each episode

import numpy as np
import matplotlib.pyplot as plt
class Grid():
    def __init__(self, size, walls, end_states):
        self.size = size
        self.walls = walls
        self.end_states = end_states
        self.all_states = {}
        for i in range(size[0]):
            for j in range(size[1]):
                if (i,j) not in walls:
                    self.all_states[(i,j)] = -0.2
        for x in end_states:
            self.all_states[x] = end_states[x]
        self.sa_history = []
        self.reward_history = []
        self.game_over = False
    def action(self, a):
        p = np.random.randn()
        if p > 0.5:
            return a
        else:
            n = np.random.randint(1,5)
            return n
    def move(self, a, i, j):
        self.sa_history.append(((i, j), a))
        old_n = (i,j)
        n = (i,j)
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
        if n == old_n:
            self.reward_history.append(-100.0)
            self.game_over = True
        else:
            self.reward_history.append(self.all_states[n])
        if n in self.end_states:
            self.game_over = True
        return n[0], n[1]
    def reset(self):
        self.sa_history = []
        self.reward_history = []
        self.game_over = False
        
# initiate grid
size = (3, 4)
end_states = {}
end_states[(0,3)] = 1.0
end_states[(1,3)] = -1.0
walls = [(1,1)]
grid = Grid(size, walls, end_states)
def draw_value(value, show = True):
    print('-----------------------------')
    for i in range(size[0]):
        v = ''
        for j in range(size[1]):
            v = v + str(value[(i,j)])
            v = v + '\t'
        print(v)
    print('-----------------------------')
    
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
            if V[s][a] > max_value:
                max_value = V[s][a]
                max_action = a
        policy[s] = max_action
    return policy
# how to put Enough:
def game(E, lr):
    deltas = []
    policy = {}
    for i in range(size[0]):
        for j in range(size[1]):
            if (i,j) not in walls:
                policy[(i, j)] = np.random.randint(1,5)
    V = {}
    Q = {}
    for s in grid.all_states:
        V[s] = {}
        Q[s] = {}
        for a in range(1,5):
            V[s][a] = 0
            Q[s][a] = []
    x, y = 0, 0
    for i in range(5000):
        print(i)
        while not grid.game_over:
            p = np.random.randn()
            if p > E:
                x, y = grid.move(grid.action(policy[(x,y)]), i = x, j = y)
            else:
                x, y = grid.move(grid.action(np.random.randint(1,5)), i = x, j = y)
        grid.reward_history.reverse()
        grid.sa_history.reverse()
        visited_state = []
        for n, m in enumerate(grid.sa_history):
            s = m[0]
            a = m[1]
            v = 0
            biggest_change = -100.0
            if s not in visited_state:
                v = grid.reward_history[n] + lr*v
                Q[s][a].append(v)
                old_value = V[s][a]
                new_value = np.mean(Q[s][a])
                biggest_change = max(biggest_change, abs(old_value-new_value))
                V[s][a] = np.mean(Q[s][a])
        policy = eval_policy(policy, V, grid.all_states.keys())
        deltas.append(biggest_change)
        grid.reset()
    # value to do policy
    value = {}
    for s in V:
        total = 0
        count = 0
        for a in V[s]:
            if V[s][a]!=-1000.0:
                total += V[s][a]
                count += 1
        value[s] = total/count
    draw_policy(policy)
    return deltas
deltas = game(0.06, 0.4)
ax = plt.plot(deltas)
ax.set_xscale("log")
show()

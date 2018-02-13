# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:23:58 2018
Shochastic Gradient Descent
MonteCarlo with Approximation
@author: FPTShop
"""

# use game epissode, pdate value in episode, return state and value
# initialize a theta, size 4?
# while initiate theta, have to use random.randn, and also, devide by 2 for convinient derivaitve
# initialize a f(x), vectorize state, size 4?
# alpha learning rate of theta, decay by t for convergence. Anything else need to decay, like epsilon??
# x = f(s)
# V_hat = f(s)*(theta)
# update theta = theta + alpha(G - V_hat)*x //this is because minimize the error and gradient descent smartly
# do this for 20000 time

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
            self.all_states[x] = end_states[x] # reward and count
        self.policy = {}
        for i in range(size[0]):
            for j in range(size[1]):
                    if (i,j) not in walls and (i,j) not in end_states:
                        self.policy[(i, j)] = np.random.randint(1,5)
                    else:
                        self.policy[(i, j)] = 0
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
        self.sa_history.append((i,j))
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
    def eval_policy(self, value): 
        for s in self.all_states:
            best_policy = 0
            best_value = -10000
            if s not in self.end_states:
                i = s[0]
                j = s[1]
                if i>0 and (i-1, j) not in self.walls:
                    if value[(i-1, j)]>best_value:
                        best_value = value[(i-1, j)]
                        best_policy = 1
                if j<self.size[1]-1 and (i, j+1) not in self.walls:
                    if value[(i, j+1)]>best_value:
                        best_value = value[(i, j+1)]
                        best_policy = 2
                if i<self.size[0]-1 and (i+1, j) not in self.walls:
                    if value[(i+1, j)]>best_value:
                        best_value = value[(i+1, j)]
                        best_policy = 3
                if j>0 and (i, j-1) not in self.walls:
                    if value[(i, j-1)]>best_value:
                        best_value = value[(i, j-1)]
                        best_policy = 4
                self.policy[s] = best_policy
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

def f(s):
    return np.array([s[0], s[1], s[0]*s[1], 1]) # maybe flex the one a little bit

def game(grid, E, lr):
    G = []
    x, y = 0, 2
    while not grid.game_over:
        p = np.random.randn()
        if p > E:
            x, y = grid.move(grid.action(grid.policy[(x,y)]), i = x, j = y)
        else:
            x, y = grid.move(grid.action(np.random.randint(1,5)), i = x, j = y)
    grid.reward_history.reverse()
    grid.sa_history.reverse()
    visited_state = []
    v = 0
    for n, s in enumerate(grid.sa_history):
        if s not in visited_state:
            v = grid.reward_history[n] + lr*v
            G.append(v)
    return grid.sa_history, G

# training part
theta = np.random.random(4)
ALPHA = 0.01
value = {}  
for i in range(size[0]):
    for j in range(size[1]):
        value[(i,j)] = 0
deltas = []

x, y = 0, 0
t = 1.0
for i in range(5000):
    print(i)
    if i%100==0:
        t += 0.01
    alpha = ALPHA/t
    states, G = game(grid, 0.05, 0.2)
    biggest_diff = 0
    for n,s in enumerate(states):
        x = f(s)
        V_hat = theta.dot(x)
        theta += alpha*2*(V_hat-G[n])*x
        old_value = value[s]
        new_value = theta.dot(x)
        biggest_diff = max(biggest_diff, abs(old_value-new_value))
        value[s] = new_value
    deltas.append(biggest_diff)
    grid.eval_policy(value) # fix, now is a key
    grid.reset()
    
plt.plot(deltas)
plt.show()

# fix
print('-----------------------------')
for i in range(size[0]):
    v = ''
    for j in range(size[1]):
        if (i,j) not in end_states and (i,j) not in walls:
            v = theta.dot(f((i,j))) # be careful with end_states and walls
            v = str(v) + '\t'
        else:
            v = '0' + '\t'
    print(v)
print('-----------------------------')
        
    
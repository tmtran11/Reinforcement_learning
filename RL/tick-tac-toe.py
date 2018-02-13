# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:03:07 2017
My first AI game agent, who learn to play Tic-Tac_Toe

Practice
@author: FPTShop
"""
# class environment: init: x, o, board, winner. enumerate state, is_over, is_win, draw_board ..
# class player: init: value, history. initialize value function, update value, history_state, reset_history_state
# class human: same with player, but pass several method. print: take your turn
# def take_action: switch between player, do epsilon greedy, if done with episode
# loop training
import numpy as np
from math import sqrt, log
LENGTH = 3
class Environment:
    def __init__(self):
        self.players = []
        self.board = np.zeros((LENGTH, LENGTH))
        self.winner = None
        self.over = False
    def update_state(self):
        self.over = np.all((self.board==0)==False)
        for x in range(LENGTH):
            if sum(self.board[x,:]) in [3,-3]:
                self.over = True
                self.winner = self.players[int(sum(self.board[x,:])/3)]
                break
            if sum(self.board[:,x]) in [3,-3]:
                self.over = True
                self.winner = self.players[int(sum(self.board[:,x])/3)]
                break
        sum_diag1 = 0
        sum_diag2 = 0
        for x in range(LENGTH):
            sum_diag1 += self.board[x,x]
            sum_diag2 += self.board[x,LENGTH-x-1]
        if sum_diag1 in [3,-3]:
            self.over = True
            self.winner = self.players[int(sum_diag1/3)]
        if sum_diag2 in [3,-3]:
            self.over = True
            self.winner = self.players[int(sum_diag2/3)]
    def draw(self, do):
        if do:
            print('--------------')
            for r in range(LENGTH):
                xo = ''
                for c in range(LENGTH):
                    if self.board[r,c]==1:
                        xo += '  X'
                    elif self.board[r,c]==-1:
                        xo += '  O'
                    else:
                        xo += '  -'
                print(xo)
            print('--------------')
        if self.over and do:
            print('Game over')
        
class Player:
    def __init__(self, y):
        self.value = {} #state_number: value
        for x in range(3**(LENGTH**2)):
            self.value[x] = [0,0.001]
        self.history = []
        self.y = y
        self.board = np.zeros((LENGTH, LENGTH))
        self.N = 0
    def update_self(self,action):
        self.board[action[0],action[1]] = 1
    def update_opponent(self,action):
        self.board[action[0],action[1]] = 2
    def update_value(self, reward): 
        # combination between learning rate and 1/n. Is this correct?
        v_bef = 0
        for i in self.history[::-1]:
            v = reward + (1/self.value[i][1])*self.y*v_bef
            self.value[i][0] = v
            self.value[i][1] = self.value[i][1] + 1
            v_bef = v
            reward = 0
    def update_history(self, state):
        self.history.append(state)
    def enum_state(self):
        n = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                n += 3**k*self.board[i,j]
                k += 1
        return n
    def take_action(self, state, actions, draw_verbose = False):  
        self.N = self.N + 1
        pos = []
        for action in actions:
            pos.append((state + 3**(action[0]*3+action[1]), action))
        np.random.shuffle(pos)
        all_val = []
        m = -100000000
        best_action = None
        for x in pos:
            if self.value[int(x[0])][0] + sqrt(2.0*log(self.N)/(self.value[int(x[0])][1]+0.01)) > m:
                m = self.value[int(x[0])][0] + sqrt(2.0*log(self.N)/(self.value[int(x[0])][1]+0.001))
                all_val.append((self.value[int(x[0])][0], x[1]))
                best_state = int(x[0])
                best_action = x[1]
        self.value[best_state][1] = self.value[best_state][1] + 1
        if draw_verbose:
            board = self.board.copy()
            for x in all_val:
                board[x[1][0], x[1][1]] = x[0]
            print('--------------')
            for r in range(LENGTH):
                st = ' '
                for c in range(LENGTH):
                    st = st + str(board[r,c])
                    st = st + ' '
                print(st)
            print('--------------')
        return best_action
    def reset(self):
        self.history = []
        self.board = np.zeros((LENGTH, LENGTH))
        
class Human:
    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
    def update_self(self,action):
        self.board[action[0],action[1]] = 1
    def update_opponent(self,action):
        self.board[action[0],action[1]] = 2
    def enum_state(self):
        n = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                n += 3**k*self.board[i,j]
                k += 1
        return n
    def take_action(self, state, actions, draw_verbose = False):  
        action = input('Your turn, r space c')
        action = action.split()
        return (int(action[0]), int(action[1]))
    def update_value(self, reward): 
        pass
    def update_history(self, state):
        pass
    def reset(self):
        pass
    
      
def game(p1, p2, show=False):
    env = Environment()
    prob1 = np.random.random()
    if prob1 > 0.5:
        env.players = [0,p1,p2]
    else:
        env.players = [0,p2,p1]
    xo = -1
    while not env.over:
        xo = 0 - xo
        actions = []
        self_state = env.players[xo].enum_state()
        print('self state: '+ str(self_state))
        for i in range(LENGTH):
            for j in range(LENGTH):
                if env.board[i,j]==0:
                   actions.append((i,j)) 
        action = env.players[xo].take_action(self_state, actions, draw_verbose = show)
        env.players[xo].update_self(action)
        env.players[0-xo].update_opponent(action)
        env.board[action[0], action[1]] = xo
        env.update_state()
        env.draw(show)
        self_state = env.players[xo].enum_state()
        env.players[xo].update_history(self_state)
    if env.winner != None:
            env.players[xo].update_value(1.0)
            env.players[0-xo].update_value(-2.0)
    else:
        env.players[xo].update_value(0)
        env.players[0-xo].update_value(0)
    p1.reset()
    p2.reset()

p1 = Player(0.3)
p2 = Player(0.3)
for i in range(30000):
    game(p1, p2)
    print(i)

game(p1, p2, show = True)

human = Human()
game(human, p1, show = True)


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:32:53 2017
Epsilon Greedy Algorithm
Bandits problem
@author: FPTShop
"""

import numpy as np
import matplotlib.pyplot as plt


class Bandit():
    def __init__(self, m, mean):
        self.m = m
        self.N = 0
        self.mean = mean
    def pull(self):
        return np.random.randn() + self.m
    def update(self, x):
        self.N += 1
        self.mean = float(self.mean * (self.N-1) + x)/float(self.N)
    
def E_greedy(m1, m2, m3, N, E, mean):
    bandits = [Bandit(m1, mean), Bandit(m2, mean), Bandit(m3,mean)]
    data = np.empty(N)
    for n in range(N):
        p = np.random.random()
        if p<E:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[n] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    return cumulative_average
    
E_greedy(1.0, 2.0, 3.0, 100000, 0.01, 0.0)
E_greedy(1.0, 2.0, 3.0, 100000, 0.01, 1.0)
E_greedy(1.0, 2.0, 3.0, 100000, 0.01, 2.0)

        

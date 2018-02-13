# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:38:58 2017

@author: FPTShop
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from math import log


class Bandit():
    def __init__(self, m):
        self.m = m
        self.N = 0
        self.mean = 0
    def pull(self):
        return np.random.randn() + self.m
    def update(self, x):
        self.N += 1.0
        self.mean = float(self.mean * (self.N-1) + x)/float(self.N)
    
def E_greedy(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)
    for n in range(1, N+1):
        j = np.argmax([b.mean + sqrt(2.0*log(n)/(b.N+0.001)) for b in bandits])
        print(j)
        x = bandits[j].pull()
        bandits[j].update(x)
        data[n-1] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    return cumulative_average

E_greedy(1.0, 2.0, 3.0, 100000)

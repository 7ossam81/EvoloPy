# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016

@author: Hossam Faris

Vectorized by AmirPouya Hemmasian in Dec 2021

"""

import numpy as np
from solution import solution
import time


def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # initialize alpha, beta, and delta_pos
    top3_score = np.array([float('inf')]*3)
    top3_pos = np.zeros((3,dim))

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize the positions of search agents
    Positions = np.random.uniform(lb, ub, (SearchAgents_no,dim))

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Loop counter
    print(f'GWO is optimizing {objf.__name__}')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    for t in range(Max_iter):
        
        f_list = objf(Positions)
        
        # new alpha, beta, delta
        f_race = np.concatenate((top3_score, f_list))
        pos_race = np.concatenate((top3_pos, Positions))
        winners = np.argsort(f_race)[:3]
        top3_score = f_race[winners]
        top3_pos = pos_race[winners]
        
        # a decreases linearly fron 2 to 0
        a = 2*(1-t/Max_iter)
        
        # sampling the random values r
        r = np.random.rand(6, SearchAgents_no, dim)
        
        # A and C for alpha, beta, delta
        A = a*(2*r[:3]-1)
        C = 2*r[3:]
        
        Positions = np.mean(
                    top3_pos[:, None, :] -
                    A*np.abs(C*top3_pos[:, None, :] - Positions)
                    , axis=0)
                    
        Positions = np.clip(Positions, lb, ub)
        
        Convergence_curve[t] = top3_score[0]

        if (t+1) % 100 == 0:
            print(f'Iteration: {t+1}  |  Best Fitness: {top3_score[0]}')

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s

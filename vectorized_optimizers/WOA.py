# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam

Vectorized by AmirPouya Hemmasian in Dec 2021
"""

import numpy as np
from solution import solution
import time


def WOA(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # initialize position vector and score for the leader
    Leader_pos = np.zeros(dim)
    Leader_score = float("inf")  # change this to -inf for maximization problems

    # Initialize the positions of search agents
    
    lb = np.array(lb)
    ub = np.array(ub)

    Positions = np.random.uniform(lb, ub, (SearchAgents_no,dim))

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    ############################
    s = solution()

    print(f'WOA is optimizing {objf.__name__}')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    # Main loop
    for t in range(Max_iter):
        
        f_list = objf(Positions)
        
        bi = f_list.argmin()
        
        if f_list[bi] < Leader_score: 
            Leader_score = f_list[bi]
            Leader_pos = Positions[bi].copy()

         
        a = 2*(1-t/Max_iter)
        a2 = -1-t/Max_iter

        r1, r2, p = np.random.rand(3, SearchAgents_no)
        
        A = a*(2*r1-1)
        C = 2*r2
        
        bl = p>=0.5 # bubble net
        es = (p<0.5) & (np.abs(A)>=1) # encircle search
        eh = (p<0.5) & (np.abs(A)<1)    # encircle hunt
        
        li = np.random.choice(SearchAgents_no, es.sum())
        l = (a2 - 1) * np.random.rand(bl.sum()) + 1
        b = 1
        A = A[:,None]
        C = C[:,None]
        
        Positions[es] = Positions[li] - A[es]*np.abs(C[es]*Positions[li] - Positions[es])
        
        Positions[eh] = Leader_pos - \
                A[eh]*np.abs(C[eh]*Leader_pos - Positions[eh])
                
        Positions[bl] = np.abs(Leader_pos - Positions[bl])*\
                np.exp(b*l[:,None])*np.cos(np.pi*2*l[:,None]) + Leader_pos
                
        
        Positions = np.clip(Positions, lb, ub)

        convergence_curve[t] = Leader_score
        
        if (t+1) % 100 == 0:
            print(f'Iteration: {t+1}  |  Best Fitness: {Leader_score}')

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "WOA"
    s.objfname = objf.__name__
    s.best = Leader_score
    s.bestIndividual = Leader_pos

    return s

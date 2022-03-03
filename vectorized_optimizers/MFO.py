# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:42:18 2016

@author: hossam

Vectorized by AmirPouya Hemmasian in Dec 2021
"""


import numpy as np
import math
from solution import solution
import time


def MFO(objf, lb, ub, dim, N, Max_iter):
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of moths
    lb = np.array(lb)
    ub = np.array(ub)
    
    
    Moth_pos = np.random.uniform(lb, ub, (N, dim))

    Convergence_curve = np.zeros(Max_iter)

    s = solution()

    print(f'MFO is optimizing {objf.__name__}')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    b = 1
    Moth_fitness = objf(Moth_pos)
    I = np.argsort(Moth_fitness)
    flame_pos = Moth_pos[I]
    flame_fitness = Moth_fitness[I]

    # Main loop
    for t in range(Max_iter):

        # Number of flames Eq. (3.14) in the paper
        Flame_no = int(N-(N-1)*t/Max_iter)
        
        a = -1-t/Max_iter
        
        # Eq. 3.12 and 3,.13 to update the moth positions:
        T = (a-1)*np.random.rand(N,dim) + 1
        
        spiral = np.exp(b*T)*np.cos(2*np.pi*T)
        
        Moth_pos[:Flame_no] = flame_pos[:Flame_no] + spiral[:Flame_no]*\
                                np.abs(flame_pos[:Flame_no]-Moth_pos[:Flame_no]) 
        if Flame_no<N:
            Moth_pos[Flame_no:] = flame_pos[Flame_no-1] + spiral[Flame_no:]+\
                            np.abs(flame_pos[Flame_no-1]-Moth_pos[Flame_no:])
        
        # Check if moths go out of the search spaceand bring it back
        Moth_pos = np.clip(Moth_pos, lb, ub)
        
        # evaluate moths
        Moth_fitness = objf(Moth_pos)
        
        double_population = np.concatenate((flame_pos, Moth_pos))
        double_fitness = np.concatenate((flame_fitness, Moth_fitness))
        # Sort the moths
        I = np.argsort(double_fitness)[:N]
        flame_fitness = double_fitness[I]
        flame_pos = double_population[I]
        
        Best_flame_score = flame_fitness[0]
        
        Convergence_curve[t] = Best_flame_score
        
        if (t+1) % 100 == 0:
            print(f'Iteration: {t+1}  |  Best Fitness:  {Best_flame_score}')
        

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MFO"
    s.objfname = objf.__name__

    return s

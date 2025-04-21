# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
"""

import numpy as np
from EvoloPy.solution import solution
import time
from typing import Callable, Union, List

def PSO(objf: Callable, lb: Union[float, List[float]], ub: Union[float, List[float]], 
        dim: int, PopSize: int, iters: int) -> solution:
    """
    Particle Swarm Optimization (PSO) algorithm
    
    Parameters
    ----------
    objf : callable
        The objective function to be minimized
    lb : float or list
        Lower bounds for decision variables
    ub : float or list
        Upper bounds for decision variables
    dim : int
        Problem dimension
    PopSize : int
        Population size
    iters : int
        Maximum number of iterations
        
    Returns
    -------
    s : solution
        Solution containing optimization results
    """
    # PSO parameters
    Vmax = 6.0
    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0

    s = solution()
    
    # Convert bounds to arrays if they're scalar
    if not isinstance(lb, list) and not isinstance(lb, np.ndarray):
        lb = [lb] * dim
    if not isinstance(ub, list) and not isinstance(ub, np.ndarray):
        ub = [ub] * dim
    
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Ensure bounds are the right shape
    if len(lb) != dim:
        lb = np.array([lb[0]] * dim)
    if len(ub) != dim:
        ub = np.array([ub[0]] * dim)

    # Initialize positions and velocities
    pos = np.zeros((PopSize, dim))
    vel = np.zeros((PopSize, dim))
    
    # Initialize personal best positions and scores
    pBestScore = np.full(PopSize, float("inf"))
    pBest = np.zeros((PopSize, dim))
    
    # Initialize global best
    gBestScore = float("inf")
    gBest = np.zeros(dim)

    # Initialize positions randomly within bounds
    pos = np.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    
    # Initialize convergence tracking
    convergence_curve = np.zeros(iters)
    
    print('PSO is optimizing  "' + objf.__name__ + '"')
    
    # Start timing
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for l in range(iters):
        # Update inertia weight
        w = wMax - l * ((wMax - wMin) / iters)
        
        # For each particle
        for i in range(PopSize):
            # Clip position to bounds
            pos[i] = np.clip(pos[i], lb, ub)
            
            # Evaluate fitness
            fitness = objf(pos[i])
            
            # Update personal best
            if fitness < pBestScore[i]:
                pBestScore[i] = fitness
                pBest[i] = pos[i].copy()
            
            # Update global best
            if fitness < gBestScore:
                gBestScore = fitness
                gBest = pos[i].copy()
        
        # Update velocities and positions for all particles
        r1 = np.random.random((PopSize, dim))
        r2 = np.random.random((PopSize, dim))
        
        # Calculate new velocities
        vel = (w * vel + 
               c1 * r1 * (pBest - pos) + 
               c2 * r2 * (gBest - pos))
        
        # Clip velocities
        vel = np.clip(vel, -Vmax, Vmax)
        
        # Update positions
        pos = pos + vel
        
        # Record best fitness
        convergence_curve[l] = gBestScore
        
        if l % 1 == 0:
            print(["At iteration " + str(l + 1) + " the best fitness is " + str(gBestScore)])
    
    # End timing
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    
    # Record results
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.bestIndividual = gBest
    s.best_score = gBestScore  # Store the best score in the solution object
    s.objfname = objf.__name__
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = PopSize
    s.maxiers = iters
    
    return s

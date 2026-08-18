"""
Created on Tu August 18 11:36:40 2026

@author: hhsophy

"""

import numpy as np
import time
from EvoloPy.solution import solution

def SA(objf,lb,ub,dim,equilibrium=100,x0=None,T0=None,Tmin=1e-3,alpha=0.9,step_size=0.1,max_temperatures=10000,target_acceptance=0.8,seed=None):
    """
    Simulated Annealing for continuous minimization.

    Parameters
    ----------
    objf : callable
        Objective function. Receives a 1-D numpy array and returns a scalar.

    lb, ub : float or array-like
        Lower and upper bounds.

    dim : int
        Number of dimensions.

    x0 : array-like, optional
        Initial solution. If None, generated randomly.

    T0 : float, optional
        Initial temperature.
        If None, estimated automatically using preliminary moves.

    Tmin : float
        Final temperature / stopping temperature.

    alpha : float
        Geometric cooling factor.
        T_new = alpha * T

    equilibrium : int
        Number of neighbor moves performed at each temperature.

    step_size : float
        Size of neighborhood perturbation relative to the search range.

    max_temperatures : int
        Maximum number of temperature levels.

    target_acceptance : float
        Target acceptance probability used when automatically
        estimating T0.

    seed : int, optional
        Random seed.
    """

    rng = np.random.default_rng(seed)

    # Bounds
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if lb.ndim == 0:
        lb = np.full(dim, lb)

    if ub.ndim == 0:
        ub = np.full(dim, ub)

    # Initial solution
    if x0 is None:
        current = rng.uniform(lb, ub)
    else:
        current = np.asarray(x0, dtype=float).copy()
        current = np.clip(current, lb, ub)

    current_fitness = objf(current)

    # Best solution found so far
    best = current.copy()
    best_fitness = current_fitness

    # Neighborhood size
    search_range = ub - lb
    sigma = step_size * search_range
    s = solution()

    # Loop counter
    print('Simulated Annealing is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    

    # Initial temperature
    if T0 is None:

        # Generate preliminary uphill moves
        positive_deltas = []

        test_solution = current.copy()
        test_fitness = current_fitness

        for I in range(100):

            neighbor = test_solution + rng.normal(
                0,
                sigma,
                size=dim
            )

            neighbor = np.clip(neighbor, lb, ub)

            neighbor_fitness = objf(neighbor)

            delta_E = neighbor_fitness - test_fitness

            if delta_E > 0:
                positive_deltas.append(delta_E)

            test_solution = neighbor
            test_fitness = neighbor_fitness

        if len(positive_deltas) == 0:
            T0 = 1.0
        else:
            mean_delta = np.mean(positive_deltas)

            T0 = -mean_delta / np.log(target_acceptance)

    T = T0

    # Main SA loop
    history =np.zeros(max_temperatures)

    temperature_iteration = 0

    while T > Tmin and temperature_iteration < max_temperatures:
      

        # Equilibrium state
      
        for I in range(equilibrium):

            # Generate random neighbor
            neighbor = current + rng.normal(0,sigma,size=dim)

            # Keep solution inside bounds
            neighbor = np.clip(neighbor, lb, ub)

            neighbor_fitness = objf(neighbor)

            # Move acceptance
            delta_E = neighbor_fitness - current_fitness

            # Improving solution
            if delta_E <= 0:

                current = neighbor
                current_fitness = neighbor_fitness

            # Worse solution
            else:

                acceptance_probability = np.exp(-delta_E / T)

                R = rng.random()

                if acceptance_probability > R:

                    current = neighbor
                    current_fitness = neighbor_fitness

            # Update global best

            if current_fitness < best_fitness:

                best = current.copy()
                best_fitness = current_fitness
            
            if I % 10 == 0:
              print(f"At iteration {I} of temperature {T} the best fitness is {best_fitness}")
        # Save best fitness at this temperature
        history[temperature_iteration]=best_fitness

        # Geometric cooling
        T = alpha * T

        temperature_iteration += 1


    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = history
    s.optimizer = "Simulated Annealing"
    s.bestIndividual = best
    s.best_score = best_fitness
    s.lb = lb
    s.ub = ub
    s.dim = dim
    s.popnum = 1
    s.objfname = objf.__name__

    return s
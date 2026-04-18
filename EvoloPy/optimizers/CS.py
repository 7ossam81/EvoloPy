# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""
import math
import numpy as np
import random
import time
from EvoloPy.solution import solution


def get_cuckoos(nest, best, lb, ub, n, dim):

    # perform Levy flights
    tempnest = np.zeros((n, dim))
    tempnest = np.array(nest)
    beta = 3 / 2
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    s = np.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = np.random.randn(len(s)) * sigma
        v = np.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * np.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = np.clip(s[k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    tempnest = np.zeros((n, dim))
    tempnest = np.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :])
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = np.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim):

    # Discovered or not
    tempnest = np.zeros((n, dim))

    K = np.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (
        nest[np.random.permutation(n), :] - nest[np.random.permutation(n), :]
    )

    tempnest = nest + stepsize * K

    return tempnest


##########################################################################


def CS(objf, lb, ub, dim, n, N_IterTotal):

    # lb=-1
    # ub=1
    # n=50
    # N_IterTotal=1000
    # dim=30

    # Discovery rate of alien eggs/solutions
    pa = 0.25

    nd = dim

    #    Lb=[lb]*nd
    #    Ub=[ub]*nd
    convergence = []
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # RInitialize nests randomely
    nest = np.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = np.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    new_nest = np.zeros((n, dim))
    new_nest = np.copy(nest)

    bestnest = [0] * dim

    fitness = np.zeros(n)
    fitness.fill(float("inf"))

    s = solution()

    print('CS is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    fmin, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)
    convergence = np.zeros(N_IterTotal)
    # Main loop counter
    for iter in range(0, N_IterTotal):
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        new_nest = empty_nests(new_nest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        if iter % 1 == 0:
            print(["At iteration " + str(iter) + " the best fitness is " + str(fmin)])
        convergence[iter] = fmin

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "CS"
    s.bestIndividual = bestnest
    s.best_score = fmin
    s.objfname = objf.__name__

    return s

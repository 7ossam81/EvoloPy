import random
import numpy
import time
from solution import solution


# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(objf,lb,ub,dim,PopSize,iters):
    
    mutation_factor=0.5
    crossover_ratio=0.7
    stopping_func=None

    # convert lb, ub to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    # solution
    s = solution()
    
    s.best = float("inf")

    # initialize population
    population = []

    population_fitness = numpy.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = numpy.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = objf(population[i, :])
        population_fitness[p] = fitness
        #s.func_evals += 1

        # is leader ?
        if fitness < s.best:
            s.best = fitness
            s.leader_solution = population[i, :]

    convergence_curve=numpy.zeros(iters)
    # start work
    print("DE is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    t = 0
    while t < iters:
        # should i stop
        if stopping_func is not None and stopping_func(s.best, s.leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in  range(PopSize) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (population[id_2, d] - population[id_3, d])

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = numpy.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = objf(mutant_sol)
            #s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < s.best:
                    s.best = mutant_fitness
                    s.leader_solution = mutant_sol

        convergence_curve[t]=s.best
        if (t%1==0):
               print(['At iteration '+ str(t+1)+ ' the best fitness is '+ str(s.best)]);

        # increase iterations
        t = t + 1
        
        timerEnd=time.time()  
        s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime=timerEnd-timerStart
        s.convergence=convergence_curve
        s.optimizer="DE"
        s.objfname=objf.__name__

    # return solution
    return s

import random
import numpy
import math
from colorama import Fore, Back, Style
import time
import collections

class solution:
    def __init__(self):
        self.leader_fitness = 0
        self.leader_solution=[]
        self.func_evals = 0

# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(population_size, individual_size,
        lower_bound, upper_bound, fitness_func,iters_count,
       mutation_factor=0.5, crossover_ratio=0.7,
        stopping_func=None, verbose=False):

    # convert lower_bound, upper_bound to array
    if not isinstance(lower_bound, list):
        lower_bound = [lower_bound for _ in range(individual_size)]
        upper_bound = [upper_bound for _ in range(individual_size)]

    # solution
    result = solution()
    result.leader_fitness = float("inf")

    # initialize population
    population = []

    population_fitness = numpy.array([float("inf") for _ in range(population_size)])

    for s in range(population_size):
        sol = []
        for d in range(individual_size):
            d_val = random.uniform(lower_bound[d], upper_bound[d])
            sol.append(d_val)

        population.append(sol)

    population = numpy.array(population)

    # calculate fitness for all the population
    for i in range(population_size):
        fitness = fitness_func(population[i, :])
        population_fitness[s] = fitness
        result.func_evals += 1

        # is leader ?
        if fitness < result.leader_fitness:
            result.leader_fitness = fitness
            result.leader_solution = population[i, :]

    # start work
    t = 0
    while t < iters_count:
        # should i stop
        if stopping_func is not None and stopping_func(result.leader_fitness, result.leader_solution, t):
            break

        # loop through population
        for i in range(population_size):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in  range(population_size) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(individual_size):
                d_val = population[id_1, d] + mutation_factor * (population[id_2, d] - population[id_3, d])

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = numpy.clip(mutant_sol, lower_bound, upper_bound)

            # calc fitness
            mutant_fitness = fitness_func(mutant_sol)
            result.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < result.leader_fitness:
                    result.leader_fitness = mutant_fitness
                    result.leader_solution = mutant_sol

        # log
        if t % 10 == 2 and verbose == True:
            txt = "Generation=%s , Best Fitness=%s"
            print(txt % (str(t + 1), result.leader_fitness))

        # increase iterations
        t = t + 1

    # return solution
    return result

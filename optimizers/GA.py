"""
Created on Sat May 24 20:18:05 2024

@author: Raneem
"""
import numpy as np
import random
import time
import sys

from solution import solution


def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    """
    The crossover of all individuals

    Parameters
    ----------
    population : list
        The list of individuals
    scores : list
        The list of fitness values for each individual
    popSize: int
        Number of chrmosome in a population
    crossoverProbability: float
        The probability of crossing a pair of individuals
    keep: int
        Number of best individuals to keep to the next generation which have high scores


    Returns
    -------
    N/A
    """
    # initialize a new population
    newPopulation = np.zeros_like(population)
    newPopulation[0:keep] = population[0:keep]
    # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
    for i in range(keep, popSize, 2):
        # pair of parents selection
        parent1, parent2 = pairSelection(population, scores, popSize)
        individualLength = len(parent1)
        
        if random.random() < crossoverProbability:
            offspring1, offspring2 = crossover(individualLength, parent1, parent2)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        # Add offsprings to population
        newPopulation[i] = offspring1
        newPopulation[i + 1] = offspring2

    return newPopulation


def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
    """
    The mutation of all individuals

    Parameters
    ----------
    population : list
        The list of individuals
    popSize: int
        Number of chrmosome in a population
    mutationProbability: float
        The probability of mutating an individual
    keep: int
        Number of best individuals to keep without mutating for the next generation
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list

    Returns
    -------
    N/A
    """
    for i in range(keep, popSize):
        # Mutation
        
        if random.random() < mutationProbability:
            mutation(population[i], len(population[i]), lb, ub)

def pairSelection(population, scores, popSize):
    """
    This is used to select one pair of parents using roulette Wheel Selection mechanism

    Parameters
    ----------
    population : list
        The list of individuals
    scores : list
        The list of fitness values for each individual
    popSize: int
        Number of chrmosome in a population

    Returns
    -------
    list
        parent1: The first parent individual of the pair
    list
        parent2: The second parent individual of the pair
    """

    def rouletteWheelSelectionId(scores, popSize):
        # Normalizing the scores
        total_fitness = sum(scores)
        normalized_scores = [score / total_fitness for score in scores]
        # Generating cumulative probabilities
        cumulative_probs = np.cumsum(normalized_scores)
        # Roulette wheel selection
        random_num = random.random()
        for i, cumulative_prob in enumerate(cumulative_probs):
            if random_num <= cumulative_prob:
                return i
    
    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = parent1Id
    while parent2Id == parent1Id:
        parent2Id = rouletteWheelSelectionId(scores, popSize)

    parent2 = population[parent2Id].copy()

    return parent1, parent2


def crossover(individualLength, parent1, parent2):
    """
    The crossover operator of a two individuals

    Parameters
    ----------
    individualLength: int
        The maximum index of the crossover
    parent1 : list
        The first parent individual of the pair
    parent2 : list
        The second parent individual of the pair

    Returns
    -------
    list
        offspring1: The first updated parent individual of the pair
    list
        offspring2: The second updated parent individual of the pair
    """

    # The point at which crossover takes place between two parents.
    crossover_point = random.randint(0, individualLength - 1)
    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return offspring1, offspring2


def mutation(individual, individualLength, lb, ub):
    """
    The mutation operator of a single individual

    Parameters
    ----------
    individual : list
        A generated individual after the crossover
    individualLength: int
        The maximum index of the crossover
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list

    Returns
    -------
    N/A
    """
    mutationIndex = random.randint(0, individualLength - 1)
    individual[mutationIndex] = random.uniform(lb[mutationIndex], ub[mutationIndex])


def clearDups(Population, lb, ub):

    """
    It removes individuals duplicates and replace them with random ones

    Parameters
    ----------
    objf : function
        The objective function selected
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list

    Returns
    -------
    list
        newPopulation: the updated list of individuals
    """
    newPopulation = np.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newPopulation = np.append(
            newPopulation,
            np.random.uniform(0, 1, (nDuplicates, len(Population[0])))
            * (np.array(ub) - np.array(lb))
            + np.array(lb),
            axis=0,
        )

    return newPopulation


def calculateCost(objf, population, popSize, lb, ub):

    """
    It calculates the fitness value of each individual in the population

    Parameters
    ----------
    objf : function
        The objective function selected
    population : list
        The list of individuals
    popSize: int
        Number of chrmosomes in a population
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list

    Returns
    -------
    list
        scores: fitness values of all individuals in the population
    """
    scores = np.full(popSize, np.inf)

    # Loop through individuals in population
    for i in range(0, popSize):
        # Return back the search agents that go beyond the boundaries of the search space
        population[i] = np.clip(population[i], lb, ub)

        # Calculate objective function for each search agent
        scores[i] = objf(population[i, :])

    return scores


def sortPopulation(population, scores):
    """
    This is used to sort the population according to the fitness values of the individuals

    Parameters
    ----------
    population : list
        The list of individuals
    scores : list
        The list of fitness values for each individual

    Returns
    -------
    list
        population: The new sorted list of individuals
    list
        scores: The new sorted list of fitness values of the individuals
    """
    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]

    return population, scores


def GA(objf, lb, ub, dim, popSize, iters):

    """
    This is the main method which implements GA

    Parameters
    ----------
    objf : function
        The objective function selected
    lb: list
        lower bound limit list
    ub: list
        Upper bound limit list
    dim: int
        The dimension of the indivisual
    popSize: int
        Number of chrmosomes in a population
    iters: int
        Number of iterations / generations of GA

    Returns
    -------
    obj
        s: The solution obtained from running the algorithm
    """

    cp = 1  # crossover Probability
    mp = 0.01  # Mutation Probability
    keep = 2
    # elitism parameter: how many of the best individuals to keep from one generation to the next

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = np.zeros(dim)
    scores = np.random.uniform(0.0, 1.0, popSize)#raneem
    bestScore = float("inf")

    ga = np.zeros((popSize, dim))
    for i in range(dim):
        ga[:, i] = np.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]
    convergence_curve = np.zeros(iters)

    print('GA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # print("bestIndividual", bestIndividual)
    # print("scores", scores)
    # print("bestScore", bestScore)
    # print("convergence_curve", convergence_curve)

    for l in range(iters):

        # crossover
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)

        # mutation
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)

        ga = clearDups(ga, lb, ub)

        scores = calculateCost(objf, ga, popSize, lb, ub)

        bestScore = min(scores)

        # Sort from best to worst
        ga, scores = sortPopulation(ga, scores)

        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(bestScore)
                ]
            )

    timerEnd = time.time()
    s.bestIndividual = bestIndividual
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "GA"
    s.objfname = objf.__name__

    return s

"""
Created on Sat May 24 20:18:05 2024

@author: Raneem
"""
import numpy as np
import random
import time
import sys

from EvoloPy.solution import solution


def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    """
    The crossover of all individuals

    Parameters
    ----------
    population : list
        The list of indiiduals
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

    # Step 1: Initialize a new population array. Keep the best individuals as is.
    newPopulation = np.zeros_like(population)  # Create an array of zeros with the same shape as the population
    newPopulation[0:keep] = population[0:keep]  # Keep the top 'keep' best individuals in the new population

    # Step 2: Create pairs of parents and generate offspring
    for i in range(keep, popSize, 2):  # Start from 'keep' and go up to population size, step by 2 to select pairs
        # Pair selection: Choose two parents from the population
        parent1, parent2 = pairSelection(population, scores, popSize)
        individualLength = len(parent1)  # Get the length of the individual (assumed to be same for all)

        # Step 3: Perform crossover if random probability is below crossoverProbability
        if random.random() < crossoverProbability:
            offspring1, offspring2 = crossover(individualLength, parent1, parent2)  # Perform crossover to produce two offspring
        else:
            # If no crossover occurs, the offspring are copies of the parents
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        # Step 4: Add the new offspring to the new population
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
    
    # Step 1: Start from the 'keep' individuals and iterate through the rest of the population
    for i in range(keep, popSize):
        # Step 2: Decide if mutation occurs based on mutationProbability
        if random.random() < mutationProbability:
            # Step 3: Perform mutation on the selected individual
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

    def rouletteWheelSelectionId(inverted_scores, popSize):

        # Step 1: Check if all scores are identical
        if len(set(inverted_scores)) == 1:
            # If all scores are the same, perform random selection
            return random.randint(0, popSize - 1)
        
        # Step 2: Calculate the total fitness of the population
        total_fitness = sum(inverted_scores)
        
        # Step 3: Normalize the inverted scores to create probabilities
        normalized_scores = [score / total_fitness for score in inverted_scores]
        
        # Step 4: Generate cumulative probabilities for roulette wheel
        cumulative_probs = np.cumsum(normalized_scores)
        
        # Step 5: Select a random number between 0 and 1
        random_num = random.random()
        
        # Step 6: Find the first index where the random number is less than or equal to the cumulative probability
        for i, cumulative_prob in enumerate(cumulative_probs):
            if random_num <= cumulative_prob:
                return i

    # Step A: Invert scores so lower scores (better fitness) have higher probabilities
    max_score = max(scores)
    inverted_scores = [max_score - score for score in scores]

    # Step B: Select the first parent using the roulette wheel mechanism
    parent1Id = rouletteWheelSelectionId(inverted_scores, popSize)
    parent1 = population[parent1Id].copy()  # Copy the selected parent to avoid altering the original population

    # Step C: Select the second parent, ensuring it's different from the first parent
    parent2Id = parent1Id
    while parent2Id == parent1Id:  # Keep selecting until a different parent is chosen
        parent2Id = rouletteWheelSelectionId(inverted_scores, popSize)
    parent2 = population[parent2Id].copy()  # Copy the selected parent to avoid altering the original population

    # Return the selected pair of parents
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
    
    # Generate a random index within the bounds of the individual's length
    mutationIndex = random.randint(0, individualLength - 1)
    # Assign a new value to the individual at the mutation index
    # The new value is a random number within the bounds specified by lb and ub for the chosen index
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
    # Step 1: Remove duplicate individuals from the population using np.unique, keeping only unique rows
    newPopulation = np.unique(Population, axis=0)

    # Step 2: Calculate the old and new population sizes
    oldLen = len(Population)
    newLen = len(newPopulation)

    # Step 3: If there were duplicates (i.e., the new population size is smaller), add random individuals
    if newLen < oldLen:
        # Calculate how many duplicates were removed
        nDuplicates = oldLen - newLen

        # Step 4: Generate random new individuals to replace the duplicates
        randomIndividuals = np.random.uniform(0, 1, (nDuplicates, len(Population[0]))) * (np.array(ub) - np.array(lb)) + np.array(lb)
        
        # Step 5: Append the random individuals to the new population
        newPopulation = np.append(newPopulation, randomIndividuals, axis=0)

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

    # Step 1: Initialize an array to store fitness values, with each set to infinity initially
    scores = np.full(popSize, np.inf)

    # Step 2: Loop through each individual in the population
    for i in range(0, popSize):
        # Step 3: Ensure that each individual is within the defined bounds of the search space
        population[i] = np.clip(population[i], lb, ub)

        # Step 4: Calculate the fitness value (objective function) for each individual
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

    # Step 1: Sort the scores array and get the indices of the sorted elements
    sortedIndices = scores.argsort() # argsort() returns the indices that would sort the array
    # Step 2: Use the sorted indices to reorder both the population and scores
    population = population[sortedIndices] # Sort the population based on the indices
    scores = scores[sortedIndices] # Sort the scores based on the indices

    # Step 3: Return the sorted population and scores
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

    cp = 1  # Crossover probability: Probability of two parents crossing over
    mp = 0.01  # Mutation probability: Probability of an individual mutating
    keep = 2  # Elitism parameter: The number of top individuals to carry over to the next generation

    s = solution()  # Initialize a solution object to store results

    # Ensure lower and upper bounds are lists
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize variables for tracking the best individual and score
    bestIndividual = np.zeros(dim)  # Initialize an array to store the best individual (solution) in the population. Set it initially to zeros.
    bestScore = float("inf")  # Initialize the best score (fitness value) to infinity, as we are looking for the minimum fitness score.

    # Create a population matrix where each row is an individual, and each column represents a dimension of the individual.
    ga = np.zeros((popSize, dim))  # Initialize the population matrix with zero values. It has `popSize` rows and `dim` columns.

    # Generate an initial set of fitness values for each individual in the population
    scores = np.random.uniform(0.0, 1.0, popSize)  # Create a random array of fitness scores, each between 0 and 1, for the population.

    # Initialize a list to track the convergence curve (best fitness over iterations)
    convergence_curve = np.zeros(iters)

    # For each dimension in the individual, populate it with random values within the specified bounds (lower and upper bounds).
    for i in range(dim):  
        # For each dimension `i`, assign random values to the population for that dimension.
        # The random values are scaled to the range defined by the lower bound `lb[i]` and upper bound `ub[i]`.
        ga[:, i] = np.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]
        # This scales the random values (between 0 and 1) by `(ub[i] - lb[i])` to match the range and then shifts them by `lb[i]` to fit within the bounds.


    print('GA is optimizing  "' + objf.__name__ + '"')

    # Start the timer to track the execution time of the GA
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")  # Record start time

    # Main GA loop: Iterate through generations
    for l in range(iters):
        # Crossover: Create offspring by crossing over pairs of parents
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)

        # Mutation: Mutate individuals in the population with a given probability
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)

        # Remove duplicate individuals and replace them with random ones if needed
        ga = clearDups(ga, lb, ub)

        # Calculate the fitness scores of the new population
        scores = calculateCost(objf, ga, popSize, lb, ub)

        # Sort the population from best to worst based on fitness scores
        ga, scores = sortPopulation(ga, scores)

        # Record the best individual from the sorted population
        bestIndividual = ga[0]

        # Update the best score (minimum fitness value)
        bestScore = min(scores)

        # Track the best score at this iteration for convergence
        convergence_curve[l] = bestScore

        # Print the progress every iteration
        if l % 1 == 0:
            print(f"At iteration {l + 1} the best fitness is {bestScore}")

    # End the timer and calculate execution time
    timerEnd = time.time()  # Capture the end time
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")  # Record the end time in a specific format (Year-Month-Day-Hour-Minute-Second)
    s.executionTime = timerEnd - timerStart  # Calculate the total execution time (end time - start time)

    s.bestIndividual = bestIndividual  # Store the best individual (solution) found during the optimization process
    s.convergence = convergence_curve  # Store the convergence curve (best scores for each iteration)
    s.best = bestScore
    s.optimizer = "GA"  # Store the name of the optimization algorithm used (Genetic Algorithm in this case)
    s.objfname = objf.__name__  # Store the name of the objective function being optimized

    return s

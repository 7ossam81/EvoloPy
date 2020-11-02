""" JAYA Algorithm """

import random
import numpy
import math
from solution import solution
import time


def JAYA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # Best and Worst position initialization
    Best_pos = numpy.zeros(dim)
    Best_score = float("inf")

    Worst_pos = numpy.zeros(dim)
    Worst_score = float(0)

    fitness_matrix = numpy.zeros((SearchAgents_no))

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    for i in range(0, SearchAgents_no):

        # Return back the search agents that go beyond the boundaries of the search space
        for j in range(dim):
            Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

        # Calculate objective function for each search agent
        fitness = objf(Positions[i])
        fitness_matrix[i] = fitness

        if fitness < Best_score:
            Best_score = fitness  # Update Best_Score
            Best_pos = Positions[i]

        if fitness > Worst_score:
            Worst_score = fitness  # Update Worst_Score
            Worst_pos = Positions[i]

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('JAYA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(0, Max_iter):

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            New_Position = numpy.zeros(dim)
            for j in range(0, dim):

                # Update r1, r2
                r1 = random.random()
                r2 = random.random()

                # JAYA Equation
                New_Position[j] = (
                    Positions[i][j]
                    + r1 * (Best_pos[j] - abs(Positions[i, j]))
                    - r2 * (Worst_pos[j] - abs(Positions[i, j]))
                )

                # checking if New_Position[j] lies in search space
                if New_Position[j] > ub[j]:
                    New_Position[j] = ub[j]
                if New_Position[j] < lb[j]:
                    New_Position[j] = lb[j]

            new_fitness = objf(New_Position)
            current_fit = fitness_matrix[i]

            # replacing current element with new element if it has better fitness
            if new_fitness < current_fit:
                Positions[i] = New_Position
                fitness_matrix[i] = new_fitness

        # finding the best and worst element
        for i in range(SearchAgents_no):
            if fitness_matrix[i] < Best_score:
                Best_score = fitness_matrix[i]
                Best_pos = Positions[i, :].copy()

            if fitness_matrix[i] > Worst_score:
                Worst_score = fitness_matrix[i]
                Worst_pos = Positions[i, :].copy()

        Convergence_curve[l] = Best_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Best_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "JAYA"
    s.objfname = objf.__name__

    return s

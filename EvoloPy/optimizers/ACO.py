from functorch import dim
import numpy as np
from EvoloPy.solution import solution
import time
from typing import Callable, Union, List


def ACO(objf: Callable,
        lb: Union[float, List[float]],
        ub: Union[float, List[float]],
        dim: int,
        PopSize: int,
        iters: int) -> solution:
    """
    Ant Colony Optimization for Continuous Domains (ACO_R).

    Parameters
    ----------
    objf : callable
        The objective function to be minimized.
    lb : float or list
        Lower bounds for decision variables.
    ub : float or list
        Upper bounds for decision variables.
    dim : int
        Problem dimension.
    PopSize : int
        Number of ants per iteration (population size).
    iters : int
        Maximum number of iterations.

    Returns
    -------
    s : solution
        Solution object containing all optimization results,
        compatible with the EvoloPy framework.
    """

    # ACO_R hyper-parameters
    archive_size = max(50, 10 * dim)     # Number of elite solutions in the archive
    q            = 0.5     # Locality of search (selection sharpness)
    xi           = 0.65    # Convergence speed (std deviation scaling)

    # Initialise solution object
    s = solution()

    # Bounds handling - identical pattern to PSO.py
    if not isinstance(lb, list) and not isinstance(lb, np.ndarray):
        lb = [lb] * dim
    if not isinstance(ub, list) and not isinstance(ub, np.ndarray):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)

    if len(lb) != dim:
        lb = np.array([lb[0]] * dim)
    if len(ub) != dim:
        ub = np.array([ub[0]] * dim)

    # Initialise the solution archive with random solutions
    # Archive holds 'archive_size' solutions; each row is one solution
    archive_positions = np.random.uniform(0, 1, (archive_size, dim)) * (ub - lb) + lb
    archive_scores    = np.array([objf(archive_positions[i, :])
                                  for i in range(archive_size)])

    # Sort archive best → worst (ascending fitness)
    sort_idx         = np.argsort(archive_scores)
    archive_positions = archive_positions[sort_idx, :]
    archive_scores    = archive_scores[sort_idx]

    # Pre-compute the selection weights (Gaussian rank-based)
    # These weights reflect the pheromone - better-ranked solutions
    # attract more ants. Weights are fixed across iterations because
    # the ranking mechanism (not the weights) drives learning.
    ranks   = np.arange(1, archive_size + 1)          # 1 = best
    weights = (np.exp(-((ranks - 1) ** 2) /
                (2 * (q * archive_size) ** 2)))        # Gaussian decay
    weights = weights / np.sum(weights)                # Normalise → prob

    # Tracking
    convergence_curve = np.zeros(iters)

    # Global best (initialised from the archive)
    gBestScore = archive_scores[0]
    gBest      = archive_positions[0, :].copy()

    # Start timing - same pattern as PSO.py and GWO.py
    print('ACO is optimizing  "' + objf.__name__ + '"')

    timerStart   = time.time()
    s.startTime  = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(iters):

        # Generate PopSize new ant solutions
        new_positions = np.zeros((PopSize, dim))

        for i in range(PopSize):

            # Step 1 - Select a template solution from the archive
            # using the pre-computed rank-based probability weights
            chosen_idx = np.random.choice(archive_size, p=weights)

            # Step 2 - Sample each dimension from a Gaussian centred on
            # the chosen template. The standard deviation is computed as
            # the weighted average distance of all archive solutions from
            # the chosen template (Socha & Dorigo, 2008, Eq. 5),
            # scaled by xi to control convergence speed.
            new_solution = np.zeros(dim)
            for d in range(dim):
                # Weighted std across archive for dimension d
                sigma = xi * np.sum(
                    np.abs(archive_positions[:, d]
                        - archive_positions[chosen_idx, d])
                ) / (archive_size - 1)
                # Prevent sigma collapsing to zero (would freeze search)
                if sigma == 0:
                    sigma = 1e-6

                # Sample from Gaussian centred on chosen template
                new_solution[d] = (archive_positions[chosen_idx, d]
                                   + sigma * np.random.randn())

            # Step 3 - Clip to bounds (same pattern as PSO.py)
            new_solution      = np.clip(new_solution, lb, ub)
            new_positions[i]  = new_solution

        # Evaluate fitness of all new ants
        new_scores = np.array([objf(new_positions[i, :])
                                for i in range(PopSize)])

        # Update archive - merge, re-sort, keep best archive_size
        combined_positions = np.vstack([archive_positions, new_positions])
        combined_scores    = np.concatenate([archive_scores, new_scores])

        sort_idx          = np.argsort(combined_scores)
        archive_positions = combined_positions[sort_idx[:archive_size], :]
        archive_scores    = combined_scores[sort_idx[:archive_size]]

        # Update global best
        if archive_scores[0] < gBestScore:
            gBestScore = archive_scores[0]
            gBest      = archive_positions[0, :].copy()

        # Record convergence - same pattern as PSO.py and GWO.py
        convergence_curve[l] = gBestScore

        if l % 1 == 0:
            print(["At iteration " + str(l + 1)
                   + " the best fitness is " + str(gBestScore)])

    # End timing and populate solution object
    # Exact same field names and order as PSO.py
    timerEnd          = time.time()
    s.endTime         = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime   = timerEnd - timerStart

    s.convergence     = convergence_curve
    s.optimizer       = "ACO"
    s.bestIndividual  = gBest
    s.best_score      = gBestScore
    s.objfname        = objf.__name__
    s.lb              = lb
    s.ub              = ub
    s.dim             = dim
    s.popnum          = PopSize
    s.maxiers         = iters

    return s
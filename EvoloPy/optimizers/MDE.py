""" Multi-population based Differential Evolution (MDE)

Karkinli, A.E. (2023). Detection of object boundary from point cloud by using
multi-population based differential evolution algorithm.
Neural Computing and Applications, 35, 5193-5206.
https://doi.org/10.1007/s00521-022-07969-w

Original MATLAB code (by the author):
https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm

MDE keeps a full population of 3*N vectors. In every iteration N distinct
vectors are drawn at random from it to form a sub-population; the
sub-population is evolved with a parameter-free DE step (random scale factor,
random dx/dy difference vectors, random crossover map and a small noise term)
and the improved vectors are written back to the full population.
MDE has no control parameters to tune.
"""

import numpy
import time
from EvoloPy.solution import solution


def MDE(objf, lb, ub, dim, PopSize, iters):
    """
    Parameters
    ----------
    objf : callable
        Objective function to minimise, ``objf(x)`` with ``x`` of shape (dim,).
    lb, ub : float or list
        Lower / upper bounds (scalar or one value per dimension).
    dim : int
        Number of decision variables.
    PopSize : int
        Size of the full population (3*N vectors, must be >= 9).
        The sub-population size is N = PopSize // 3; if PopSize is not a
        multiple of 3 the full population uses 3*N vectors.
    iters : int
        Number of iterations (MaxCycle). Each iteration evaluates the N
        sub-population trial vectors, i.e. one iteration costs PopSize // 3
        function evaluations (plus 3*N evaluations for the initial population).
        For an equal number of function evaluations against an optimizer that
        evaluates PopSize vectors per iteration, use 3 * iters.

    Reproducibility: MDE draws all random numbers from ``numpy.random``;
    call ``numpy.random.seed(...)`` before running it.
    """

    t = 3  # number of sub-populations held in the full population
    N = int(PopSize) // t
    if N < 3:
        raise ValueError("MDE requires PopSize >= 9 (full population of 3*N vectors, N >= 3)")

    lb = numpy.asarray(lb, dtype=float)
    ub = numpy.asarray(ub, dtype=float)
    if lb.ndim == 0:
        lb = numpy.full(dim, float(lb))
    if ub.ndim == 0:
        ub = numpy.full(dim, float(ub))

    s = solution()

    # Initialize the full population A (3*N vectors)
    A = lb + numpy.random.rand(t * N, dim) * (ub - lb)
    fitA = numpy.array([objf(A[i, :]) for i in range(t * N)], dtype=float)
    k = int(numpy.argmin(fitA))
    best = A[k, :].copy()
    best_score = fitA[k]

    # noise term (Eq. 13), carried over from the previous iteration
    noise = numpy.zeros((N, dim))

    Convergence_curve = numpy.zeros(iters)

    print('MDE is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):

        # select a random sub-population B of N distinct vectors
        j0 = numpy.random.permutation(t * N)[:N]
        B = A[j0, :].copy()
        fitB = fitA[j0].copy()

        # mutation (Eqs. 7-12)
        temp = numpy.empty((N, dim))
        for i in range(N):
            # scale = |a - rand^b| * randn^d,  a in {0,1}, b in {1..10}, d in {1..5}
            scale = abs(
                numpy.random.randint(0, 2)
                - numpy.random.rand() ** numpy.random.randint(1, 11)
            ) * numpy.random.randn() ** numpy.random.randint(1, 6)

            # two distinct random members of B, both different from i
            while True:
                r1, r2 = numpy.random.choice(N, 2, replace=False)
                if r1 != i and r2 != i:
                    break

            dx = numpy.where(numpy.random.rand(dim) < 0.5, B[r1, :], best)
            dy = numpy.where(numpy.random.rand(dim) < 0.5, B[i, :], B[r2, :])
            temp[i, :] = B[r2, :] + scale * (dx - dy)

        # parameter-free crossover map (c = 1: one random map per vector,
        # c = dim: one random map per variable)
        c = 1 if numpy.random.rand() ** numpy.random.randint(1, 6) < 0.5 else dim
        cmap = (
            numpy.abs(
                numpy.random.randint(0, 2, (N, c))
                - numpy.random.rand(N, dim) ** numpy.random.randint(1, 6, (N, c))
            )
            < 0.5
        )
        T = B + cmap * (temp + noise - B)

        # boundary control (Eq. 15): re-sample out-of-bounds variables in [lb, ub]
        out = (T < lb) | (T > ub)
        if out.any():
            R = lb + numpy.random.rand(N, dim) * (ub - lb)
            T = numpy.where(out, R, T)

        # greedy selection
        fitT = numpy.array([objf(T[i, :]) for i in range(N)], dtype=float)
        better = fitT < fitB
        B[better, :] = T[better, :]
        fitB[better] = fitT[better]

        # write the sub-population back to the full population
        A[j0, :] = B
        fitA[j0] = fitB

        k = int(numpy.argmin(fitA))
        best = A[k, :].copy()
        best_score = fitA[k]

        Convergence_curve[l] = best_score

        # noise for the next iteration (Eq. 13): B .* 10^phi .* (rand(N,1) - 0.5)
        noise = (
            B
            * 10.0 ** numpy.random.randint(-12, -8, (N, dim))
            * (numpy.random.rand(N, 1) - 0.5)
        )

        if l % 1 == 0:
            print(
                ["At iteration " + str(l + 1) + " the best fitness is " + str(best_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "MDE"
    s.bestIndividual = best
    s.best = float(best_score)
    s.best_score = float(best_score)
    s.objfname = objf.__name__

    return s

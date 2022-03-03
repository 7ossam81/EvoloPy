
import numpy as np
from solution import solution
import time


def SSA(objf, lb, ub, dim, N, Max_iter):
    
    n = N-N//2
    # constructing the chain matrix
    chain = np.zeros((n, n+1))
    chain[0,:2] = 0.5
    for i in range(1,n):
        chain[i] = 0.5*chain[i-1]
        chain[i, i+1] = 0.5    
    
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    Convergence_curve = np.zeros(Max_iter)

    lb = np.array(lb)
    ub = np.array(ub)
    
    SalpPositions = np.random.uniform(lb, ub, (N,dim))
    
    SalpFitness = objf(SalpPositions)

    bi = np.argmin(SalpFitness)
    FoodPosition = SalpPositions[bi].copy()
    FoodFitness = SalpFitness[bi]

    s = solution()

    print(f'SSA is optimizing {objf.__name__}')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for t in range(Max_iter):

        # Eq. (3.2) in the paper
        c1 = 2 * np.exp(-(4*t/Max_iter)**2)
        # Eq. (3.1) in the paper
        c2, c3 = np.random.rand(2, N//2, dim)
        sgn = np.where(c3<0.5, +1, -1)
        
        SalpPositions[:N//2] = FoodPosition + c1*sgn*(c2*(ub-lb) + lb)
        # Eq. (3.4) in the paper                        
        SalpPositions[N//2:] = chain@SalpPositions[N//2-1:]
        
        SalpPositions = np.clip(SalpPositions, lb, ub)

        SalpFitness = objf(SalpPositions)

        bi = np.argmin(SalpFitness)
        
        if SalpFitness[bi] < FoodFitness:     
            FoodPosition = SalpPositions[bi].copy()
            FoodFitness = SalpFitness[bi]

        # Display best fitness along the iteration
        if (t+1) % 100 == 0:
            print(f'Iteration" {t+1}  |  Best Fitness: {FoodFitness}')

        Convergence_curve[t] = FoodFitness

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "SSA"
    s.objfname = objf.__name__

    return s

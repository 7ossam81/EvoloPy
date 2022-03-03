""" 
Sine Cosine OPtimization Algorithm 

Author: Hossam Faris

Vectorized by: AmirPouya Hemmasian

"""



import numpy as np
from solution import solution
import time


def SCA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # destination_pos
    Dest_pos = np.zeros(dim)
    Dest_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    lb = np.array(lb)
    ub = np.array(ub)
    # Initialize the positions of search agents
    Positions = np.random.uniform(lb, ub, (SearchAgents_no,dim))

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    # Loop counter
    print(f'SCA is optimizing {objf.__name__ }')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    a = 2
    # Main loop
    for t in range(Max_iter):
        
        f_list = objf(Positions)
            
        bi = f_list.argmin()
        
        if f_list[bi] < Dest_score:
            Dest_score = f_list[bi]
            Dest_pos = Positions[bi].copy()
            Dest = np.repeat(Dest_pos[None,:], SearchAgents_no, 0)
            
        r1 = a*(1-t/Max_iter)
        
        r2, r3, r4 = np.random.rand(3, SearchAgents_no, dim)
        
        r2 *= 2*np.pi
        r3 *= 2
        
        sin = r4 < 0.5
        cos = ~sin
                            
        Positions[sin] +=  r1*np.sin(r2[sin])*\
                            np.abs(r3[sin]*Dest[sin]-Positions[sin])
                            
        Positions[cos] += r1*np.cos(r2[cos])*\
                            np.abs(r3[cos]*Dest[cos]-Positions[cos])
                            
        Positions = np.clip(Positions, lb, ub)  
        
        Convergence_curve[t] = Dest_score

        if (t+1) % 100 == 0:
            print(f'Iteration: {t+1}  |  Best Fitness: {Dest_score}')

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "SCA"
    s.objfname = objf.__name__

    return s

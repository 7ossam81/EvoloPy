# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import PSO as pso
import benchmarks
import csv


def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    
    if(algo==0):
        #x=pso.PSO()
        print(function_name)
        x=pso.PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
PSO= True
GWO= False
SCA= False
WOA= False
MVO= False
MFO = False

# Select benchmark function
F1=True
F2=True
F3=True
F4=True
F5=True
F6=True


optimizer=[PSO, GWO, SCA, WOA, MVO, MFO]
benchmarkfunc=[F1,F2,F3,F4,F5,F6] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations= 100

#Export results ?
Export=True


for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): 
            for k in range (0,NumOfRuns):
                func_details=benchmarks.getFunctionDetails(j)
                x=selector(i,func_details,PopulationSize,Iterations)
                if(Export==True):
                    with open('convergence.csv', 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        writer.writerow(x.convergence)
              
                    out.close()
            
        
        
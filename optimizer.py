# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import benchmarks
import csv
import numpy
import time


def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    
    if(algo==0):
        x=pso.PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==1):
        x=mvo.MVO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==2):
        x=gwo.GWO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
PSO= False
MVO= False
GWO = True
#SCA= False
#WOA= False
#MFO = False

# Select benchmark function
F1=True
F2=False
F3=False
F4=False
F5=False
F6=False


optimizer=[PSO, MVO, GWO]
benchmarkfunc=[F1,F2,F3,F4,F5,F6] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations= 500

#Export results ?
Export=True
#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): 
            for k in range (0,NumOfRuns):
                func_details=benchmarks.getFunctionDetails(j)
                x=selector(i,func_details,PopulationSize,Iterations)
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        a=numpy.concatenate([[x.optimizer,x.startTime,x.endTime,x.executionTime,"Cnvg"],x.convergence])
                        writer.writerow(a)
              
                    out.close()
            
        
        
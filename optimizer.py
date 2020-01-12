# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import WOA as woa
import FFA as ffa
import SSA as ssa
import GA as ga
import HHO as hho
import SCA as sca
import JAYA as jaya
import benchmarks
import csv
import numpy
import time
import matplotlib.pyplot as plt
import os


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
    if(algo==3):
        x=mfo.MFO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==4):
        x=cs.CS(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==5):
        x=bat.BAT(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==6):
        x=woa.WOA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==7):
        x=ffa.FFA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==8):
        x=ssa.SSA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==9):
        x=ga.GA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==10):
        x=hho.HHO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==11):
        x=sca.SCA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==12):
        x=jaya.JAYA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
PSO= True
MVO= False
GWO = False
MFO = False
CS = False
BAT = False
WOA = False
FFA = False
SSA = False
GA = True
HHO = False
SCA = False
JAYA = False


# Select benchmark function
F1=True
F2=True
F3=True
F4=True
F5=True
F6=True
F7=True
F8=True
F9=True
F10=True
F11=True
F12=True
F13=True
F14=True
F15=True
F16=True
F17=True
F18=True
F19=True

plot_convergence = False



optimizer=[PSO, MVO, GWO, MFO, CS, BAT, WOA, FFA, SSA, GA, HHO, SCA, JAYA]
benchmarkfunc=[F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=3

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 500
Iterations= 50

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 
ConvergenceFolder = "experiment"+time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
if(plot_convergence):
    os.mkdir(ConvergenceFolder)
# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))


for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): # start experiment if an optimizer and an objective function is selected
            
            convergence = [0]*NumOfRuns
            for k in range (0,NumOfRuns):
                
                func_details=benchmarks.getFunctionDetails(j)
                x=selector(i,func_details,PopulationSize,Iterations)
                convergence[k] = x.convergence
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime],x.convergence])
                        writer.writerow(a)
                    out.close()
                Flag=True # at least one experiment
                
                if(plot_convergence):
                    # plot fitness progression
                    allGenerations = [x+1 for x in range(Iterations)]
                    plt.plot(allGenerations, x.convergence)
                    plt.title(x.optimizer + ' - ' + x.objfname)
                    plt.xlabel('Generations')
                    plt.ylabel('Fitness')
                    plt.grid()
                    plt.savefig(ConvergenceFolder + x.optimizer + '-' + x.objfname + '-' + str(k + 1) + ".png")
                    plt.clf()
                    #plt.show() 
                                        
            if(plot_convergence):
                avgConvergence = numpy.around(numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2).tolist()
                # plot fitness progression
                allGenerations = [x+1 for x in range(Iterations)]
                plt.plot(allGenerations, avgConvergence)
                plt.title(x.optimizer + ' - ' + x.objfname)
                plt.xlabel('Generations')
                plt.ylabel('Fitness')
                plt.grid()
                plt.savefig(ConvergenceFolder + x.optimizer + '-' + x.objfname + "-avg.png")
                plt.clf()
                #plt.show() 
                
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        

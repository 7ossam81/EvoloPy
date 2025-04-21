# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:50:25 2016

@author: raneem
"""
import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)


from EvoloPy.optimizer import run

# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizer = ["GA", "PSO", "GWO"]

# Select benchmark function"
# "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","F23","F24"
# "ackley", "rosenbrock", "rastrigin", "griewank"
objectivefunc = ["F3", "F4"]

# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 3

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 30, "Iterations": 50}

# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)

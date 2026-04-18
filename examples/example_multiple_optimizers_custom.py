import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
from EvoloPy.api import run_multiple_optimizers

# Define your custom objective function
def my_equation(x):
    return np.sum(x**2) + np.sum(np.sin(x))

results = run_multiple_optimizers(
    optimizers=["PSO", "GWO", "MVO"],
    objective_funcs=[my_equation],
    dim=10,
    lb=-100,
    ub=100,
    population_size=30,
    iterations=50,
    num_runs=3,
    display_plots=True
)

for opt in ["PSO", "GWO", "MVO"]:
    print(opt, results[opt]["my_equation"]["best_fitness"])
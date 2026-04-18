import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
from EvoloPy.api import run_optimizer

"""
Output structure:

my_equation/
    results/
        details.csv
        avg.csv
        best.csv
        PSO_config.txt

    plots/
        PSO/
            convergence.png
            boxplot.png
"""

# Define your custom objective function
def my_equation(x):
    # Example: Minimize f(x) = sum(x^2) + sum(sin(x))
    return np.sum(x**2) + np.sum(np.sin(x))

# Run optimization on your custom function
result = run_optimizer(
    optimizer="PSO",
    objective_func=my_equation,
    lb=[-10] * 5,
    ub=[10] * 5,
    dim=5,
    population_size=30,
    iterations=100,
    num_runs=5,
    results_directory=None
)

print(f"Best solution: {result['best_solution']}")
print(f"Best fitness: {result['best_fitness']}")
print(f"Execution time: {result['execution_time']} seconds")
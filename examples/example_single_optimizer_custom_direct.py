import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
from EvoloPy.optimizers import PSO

# Define your custom objective function
def my_equation(x):
    # Example: Minimize f(x) = sum(x^2) + sum(sin(x))
    return np.sum(x**2) + np.sum(np.sin(x))

# Run optimization on your function
result = PSO.PSO(
    objf=my_equation,        # Your custom function
    lb=[-10] * 5,            # Lower bounds as list (v4.0+ supports lists)
    ub=[10] * 5,             # Upper bounds as list
    dim=5,                   # Dimension
    PopSize=30,              # Population size
    iters=100                # Max iterations
)

# Get results
best_solution = result.bestIndividual
best_fitness = result.best_score  # v4.0+ exposes best_score directly

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
import sys
import os

"""
Output structure:

FunctionName/
    results/
        details.csv
        avg.csv
        best.csv
    plots/
        OptimizerName/
            convergence.png
            boxplot.png
"""

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

from EvoloPy.api import run_optimizer

# Run PSO on benchmark function F1
result = run_optimizer(
    optimizer="PSO",
    objective_func="F1",
    dim=30,
    lb=-100,                 # Lower bounds (scalar or list)
    ub=100,                  # Upper bounds (scalar or list)
    population_size=50,
    iterations=100,
    num_runs=5,              # Number of independent runs
    results_directory=None   # Auto-create timestamped directory
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best solution: {result['best_solution']}")
print(f"Execution time: {result['execution_time']} seconds")
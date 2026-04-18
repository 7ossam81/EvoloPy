import sys
import os

"""
Output structure (per function):

FunctionName/
    results/
        details.csv
        avg.csv
        best.csv
    plots/
        optimizerName/
            convergence.png
            boxplot.png
        optimizerName/
            convergence.png
            boxplot.png
        optimizerName/
            convergence.png
            boxplot.png
        ...
        comparison/
            convergence_avg.png
            convergence_best.png
            boxplot.png
"""


# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

from EvoloPy.api import run_multiple_optimizers

results = run_multiple_optimizers(
    optimizers=["PSO", "GWO", "MVO"],
    objective_funcs=["F1", "F5"],
    dim=10,
    lb=-100,
    ub=100,
    population_size=30,
    iterations=50,
    num_runs=3,
    display_plots=True
)
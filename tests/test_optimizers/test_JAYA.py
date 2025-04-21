import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
import pytest
from EvoloPy.optimizers.JAYA import JAYA

@pytest.fixture
def sample_bounds():
    """Returns lower and upper bounds for the search space."""
    lb = [-10] * 5
    ub = [10] * 5
    return lb, ub

@pytest.fixture
def sample_objective_function():
    """Defines a simple objective function for testing."""
    def objective_function(x):
        return np.sum(x ** 2)  # Sum of squares (minimized at x=0)
    return objective_function

def test_JAYA(sample_objective_function, sample_bounds):
    """Tests if the JAYA algorithm runs correctly and produces a valid output."""
    lb, ub = sample_bounds
    dims = 5  # Number of dimensions
    PopSize = 10  # Population size
    iters = 30  # Number of iterations

    sol = JAYA(sample_objective_function, lb, ub, dims, PopSize, iters)

    assert sol.bestIndividual.shape == (dims,), "Best individual should match input dimensions"
    assert sol.convergence.shape == (iters,), "Convergence curve should match iteration count"
    assert isinstance(sol.best, (int, float)), "Best score should be a numeric value"
    assert sol.best >= 0, "Best score should be non-negative"

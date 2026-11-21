import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
import pytest
from EvoloPy.optimizers.MDE import MDE

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

def test_MDE(sample_objective_function, sample_bounds):
    """Tests if the MDE algorithm runs correctly and produces a valid output."""
    lb, ub = sample_bounds
    dims = 5  # Number of dimensions
    PopSize = 30  # Full population size (3 sub-populations of 10)
    iters = 30  # Number of iterations

    sol = MDE(sample_objective_function, lb, ub, dims, PopSize, iters)

    assert sol.bestIndividual.shape == (dims,), "Best individual should match input dimensions"
    assert sol.convergence.shape == (iters,), "Convergence curve should match iteration count"
    assert isinstance(sol.best, (int, float)), "Best score should be a numeric value"
    assert sol.best >= 0, "Best score should be non-negative"
    assert sol.optimizer == "MDE"
    assert np.all(np.diff(sol.convergence) <= 0), "Convergence curve should be non-increasing"
    assert np.all(sol.bestIndividual >= np.array(lb)) and np.all(sol.bestIndividual <= np.array(ub))

def test_MDE_reproducible(sample_objective_function, sample_bounds):
    """Two runs with the same numpy seed give the same result."""
    lb, ub = sample_bounds
    np.random.seed(7)
    sol1 = MDE(sample_objective_function, lb, ub, 5, 30, 20)
    np.random.seed(7)
    sol2 = MDE(sample_objective_function, lb, ub, 5, 30, 20)
    assert np.array_equal(sol1.convergence, sol2.convergence)

def test_MDE_small_population(sample_objective_function, sample_bounds):
    """MDE needs a full population of at least 9 vectors (N >= 3)."""
    lb, ub = sample_bounds
    with pytest.raises(ValueError):
        MDE(sample_objective_function, lb, ub, 5, 8, 10)

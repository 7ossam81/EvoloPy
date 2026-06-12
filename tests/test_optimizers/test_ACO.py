import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
import pytest
from EvoloPy.optimizers.ACO import ACO

@pytest.fixture
def simple_bounds():
    """Scalar bounds for a 5-dimensional problem."""
    return -10, 10, 5          # lb, ub, dim

@pytest.fixture
def list_bounds():
    """Per-dimension list bounds for a 5-dimensional problem."""
    lb = [-10, -5, -8, -10, -3]
    ub = [10,   5,  8,  10,  3]
    return lb, ub, 5           # lb, ub, dim

@pytest.fixture
def sphere():
    """Simple unimodal objective — global minimum 0 at origin."""
    def _sphere(x):
        return np.sum(x ** 2)
    _sphere.__name__ = "sphere"
    return _sphere

def test_solution_fields_exist(simple_bounds, sphere):
    """ACO must return a solution object with all required fields."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)

    assert hasattr(sol, "bestIndividual"),  "Missing field: bestIndividual"
    assert hasattr(sol, "best_score"),      "Missing field: best_score"
    assert hasattr(sol, "convergence"),     "Missing field: convergence"
    assert hasattr(sol, "optimizer"),       "Missing field: optimizer"
    assert hasattr(sol, "objfname"),        "Missing field: objfname"
    assert hasattr(sol, "startTime"),       "Missing field: startTime"
    assert hasattr(sol, "endTime"),         "Missing field: endTime"
    assert hasattr(sol, "executionTime"),   "Missing field: executionTime"
    assert hasattr(sol, "lb"),              "Missing field: lb"
    assert hasattr(sol, "ub"),              "Missing field: ub"
    assert hasattr(sol, "dim"),             "Missing field: dim"
    assert hasattr(sol, "popnum"),          "Missing field: popnum"
    assert hasattr(sol, "maxiers"),         "Missing field: maxiers"


def test_optimizer_label(simple_bounds, sphere):
    """optimizer field must be set to 'ACO'."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert sol.optimizer == "ACO"


def test_objfname(simple_bounds, sphere):
    """objfname must match the objective function name."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert sol.objfname == "sphere"

def test_best_individual_shape(simple_bounds, sphere):
    """bestIndividual must have length == dim."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert sol.bestIndividual.shape == (dim,), \
        f"Expected shape ({dim},), got {sol.bestIndividual.shape}"


def test_convergence_shape(simple_bounds, sphere):
    """convergence curve must have length == iters."""
    lb, ub, dim = simple_bounds
    iters = 15
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=iters)
    assert sol.convergence.shape == (iters,), \
        f"Expected shape ({iters},), got {sol.convergence.shape}"

def test_best_score_is_numeric(simple_bounds, sphere):
    """best_score must be a numeric value."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert isinstance(sol.best_score, (int, float, np.floating))


def test_best_score_non_negative_sphere(simple_bounds, sphere):
    """best_score on sphere must be >= 0."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert sol.best_score >= 0


def test_best_individual_within_bounds(simple_bounds, sphere):
    """bestIndividual must lie within [lb, ub]."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert np.all(sol.bestIndividual >= lb), "bestIndividual violates lower bound"
    assert np.all(sol.bestIndividual <= ub), "bestIndividual violates upper bound"


def test_convergence_non_increasing(simple_bounds, sphere):
    """Convergence curve must be monotonically non-increasing."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=30, iters=30)
    diffs = np.diff(sol.convergence)
    assert np.all(diffs <= 1e-10), \
        "Convergence curve is not monotonically non-increasing"


def test_execution_time_positive(simple_bounds, sphere):
    """executionTime must be a positive number."""
    lb, ub, dim = simple_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert sol.executionTime > 0

def test_scalar_bounds_broadcast(sphere):
    """Scalar lb/ub must be broadcast correctly across all dimensions."""
    dim = 8
    sol = ACO(sphere, lb=-5, ub=5, dim=dim, PopSize=20, iters=10)
    assert sol.bestIndividual.shape == (dim,)
    assert np.all(sol.bestIndividual >= -5)
    assert np.all(sol.bestIndividual <= 5)


def test_list_bounds(list_bounds, sphere):
    """Per-dimension list bounds must be respected."""
    lb, ub, dim = list_bounds
    sol = ACO(sphere, lb, ub, dim, PopSize=20, iters=10)
    assert np.all(sol.bestIndividual >= lb), "bestIndividual violates lower bound"
    assert np.all(sol.bestIndividual <= ub), "bestIndividual violates upper bound"

def test_converges_on_sphere():
    """ACO should find a near-zero solution on the sphere function."""
    def sphere(x):
        return np.sum(x ** 2)
    sphere.__name__ = "sphere"

    np.random.seed(0)
    sol = ACO(sphere, lb=-100, ub=100, dim=10, PopSize=50, iters=300)
    assert sol.best_score < 1.0, \
        f"Expected near-zero on sphere, got {sol.best_score}"
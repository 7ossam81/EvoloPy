import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import numpy as np
import random
import pytest
from EvoloPy.optimizers.GA import (
    crossoverPopulaton,
    mutatePopulaton,
    pairSelection,
    crossover,
    mutation,
    clearDups,
    calculateCost,
    sortPopulation,
    GA
)

@pytest.fixture
def sample_population():
    """Creates a sample population for testing."""
    np.random.seed(42)
    return np.random.uniform(-10, 10, (10, 5))  # 10 individuals, 5 genes each

@pytest.fixture
def sample_scores():
    """Creates sample fitness scores for testing."""
    return np.random.uniform(0, 100, 10)

@pytest.fixture
def sample_bounds():
    """Returns lower and upper bounds for mutation."""
    lb = [-10] * 5
    ub = [10] * 5
    return lb, ub

def test_crossover_population(sample_population, sample_scores):
    """Tests crossover population generation."""
    new_population = crossoverPopulaton(sample_population, sample_scores, 10, 0.8, 2)
    assert new_population.shape == sample_population.shape, "Crossover should maintain population shape"

def test_mutate_population(sample_population, sample_bounds):
    """Tests mutation function."""
    lb, ub = sample_bounds
    mutatePopulaton(sample_population, 10, 0.5, 2, lb, ub)
    assert sample_population.shape == (10, 5), "Mutation should not change shape"

def test_pair_selection(sample_population, sample_scores):
    """Tests pair selection ensuring two distinct parents are selected."""
    parent1, parent2 = pairSelection(sample_population, sample_scores, 10)
    assert not np.array_equal(parent1, parent2), "Parents should not be identical"

def test_crossover():
    """Tests single-point crossover."""
    parent1 = np.array([1, 2, 3, 4, 5])
    parent2 = np.array([5, 4, 3, 2, 1])
    offspring1, offspring2 = crossover(len(parent1), parent1, parent2)
    assert offspring1.shape == parent1.shape, "Offspring shape must match parents"
    assert offspring2.shape == parent2.shape, "Offspring shape must match parents"

def test_mutation(sample_bounds):
    """Tests mutation by ensuring an element is changed."""
    lb, ub = sample_bounds
    individual = np.array([1, 2, 3, 4, 5], dtype=float)
    original = individual.copy()
    mutation(individual, len(individual), lb, ub)
    assert not np.array_equal(individual, original), "Mutation should modify at least one gene"

def test_clear_dups(sample_population, sample_bounds):
    """Tests duplicate removal and replacement."""
    lb, ub = sample_bounds
    sample_population[1] = sample_population[0]  # Introduce a duplicate
    new_population = clearDups(sample_population, lb, ub)
    assert new_population.shape == sample_population.shape, "Cleared population must have same shape"
    assert len(np.unique(new_population, axis=0)) == sample_population.shape[0], "Duplicates should be removed"

def test_calculate_cost(sample_population, sample_bounds):
    """Tests cost calculation function."""
    lb, ub = sample_bounds

    def dummy_objective_function(x):
        return np.sum(x ** 2)  # Simple sum of squares as a fitness function

    scores = calculateCost(dummy_objective_function, sample_population, 10, lb, ub)
    assert len(scores) == 10, "Scores should match population size"
    assert all(score >= 0 for score in scores), "All scores should be non-negative"

def test_sort_population(sample_population, sample_scores):
    """Tests whether the population is sorted correctly based on scores."""
    sorted_population, sorted_scores = sortPopulation(sample_population, sample_scores)
    assert np.all(sorted_scores[:-1] <= sorted_scores[1:]), "Scores should be sorted in ascending order"
    assert sorted_population.shape == sample_population.shape, "Population shape should not change"

def test_GA(sample_bounds):
    """Tests if the Genetic Algorithm runs successfully and produces a valid output."""
    lb, ub = sample_bounds

    def dummy_objective_function(x):
        return np.sum(x ** 2)  # Simple sum of squares as a fitness function

    # GA parameters
    nPop = 10
    nGen = 5
    pc = 0.8
    pm = 0.2
    keep = 2
    dims = 5
    iters = 30

    sol = GA(dummy_objective_function, lb, ub, dims, nPop, iters)

    assert sol.bestIndividual.shape == (dims,), "Final individual shape should match input dimensions"
    assert sol.convergence.shape == (iters, ), "Final convergence shape should match input iterations"
    assert isinstance(sol.best, (int, float)), "Best score should be a numeric value"
    assert sol.best >= 0, "Best score should be non-negative"

import sys
import os
import warnings
warnings.simplefilter(action="ignore")

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import pytest
import re
from EvoloPy.optimizer import selector, run
from unittest.mock import patch
from EvoloPy import benchmarks

# Mock the benchmark function
mock_benchmark = benchmarks.getFunctionDetails


@pytest.fixture
def valid_func_details():
    # Return valid benchmark details (replace with actual function details if needed)
    return ("F1", -100, 100, 30)


@pytest.fixture
def optimizer_params():
    # Return valid parameters for testing
    return {"PopulationSize": 50, "Iterations": 10}


@pytest.fixture
def export_flags():
    # Export flags for the run function
    return {
        "Export_avg": False,
        "Export_details": False,
        "Export_convergence": False,
        "Export_boxplot": False,
    }


def test_selector_valid_algorithm(valid_func_details):
    """Test selector function with a valid optimizer"""
    algo = "PSO"
    x = selector(algo, valid_func_details, popSize=50, Iter=10)
    assert x.optimizer == algo, f"Expected {algo} but got {x.optimizer}"


def test_selector_invalid_algorithm(valid_func_details):
    """Test selector function with an invalid optimizer"""
    algo = "INVALID_ALGO"
    x = selector(algo, valid_func_details, popSize=50, Iter=10)
    assert x is None, "Selector should return None for an invalid algorithm."

def test_run_function():
    # Parameters for the test
    optimizers = ["SSA"]
    functions = ["F1"]
    num_runs = 1
    params = {"PopulationSize": 5, "Iterations": 10}
    export_flags = {
        "Export_avg": True,
        "Export_details": True,
        "Export_convergence": False,
        "Export_boxplot": False,
    }

    # Run the optimizer
    run(optimizers, functions, num_runs, params, export_flags)

    # Get the current working directory
    current_dir = os.getcwd()

    # Check for directories matching the timestamp pattern
    timestamp_pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"  # Matches "YYYY-MM-DD-HH-MM-SS"
    result_dirs = [d for d in os.listdir(current_dir) if re.match(timestamp_pattern, d) and os.path.isdir(d)]
    assert result_dirs, "Results directory with a timestamp was not created."
    
    # Verify files in the found results directory
    results_dir = result_dirs[-1]
    print(f"Found results directory: {results_dir}")

    result_files = os.listdir(results_dir)
    print(f"Files in the results directory: {result_files}")
    assert any("experiment" in file for file in result_files), "Result files are not created."

def test_invalid_benchmark_function(optimizer_params, export_flags):
    """Test the run function with an invalid benchmark function"""
    optimizers = ["PSO"]
    objectivefunc = ["INVALID_FUNCTION"]
    NumOfRuns = 1

    with pytest.raises(AttributeError):
        run(optimizers, objectivefunc, NumOfRuns, optimizer_params, export_flags)


if __name__ == "__main__":
    pytest.main()

import sys
import os
import pytest
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

# Import the function to be tested
from EvoloPy.plot_convergence import run
results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
Path(results_directory).mkdir(parents=True, exist_ok=True)

# Mocking file read data for testing purposes
@pytest.fixture
def mock_data():
    # Mock data for testing
    return {
        'experiment.csv': [
            {"Optimizer": "SSA", "objfname": "F1", "ExecutionTime":0, "Iter0": 0.35, "Iter1": 0.3, "Iter2": 0.299, "Iter3": 0.29, "Iter4": 0.21},
            {"Optimizer": "PSO", "objfname": "F1", "ExecutionTime":0, "Iter0": 0.43, "Iter1": 0.32, "Iter2": 0.239, "Iter3": 0.24, "Iter4": 0.22},
            {"Optimizer": "GA", "objfname": "F1", "ExecutionTime":0, "Iter0": 0.45, "Iter1": 0.25, "Iter2": 0.22, "Iter3": 0.22, "Iter4": 0.19},
            # Add more mock data as necessary
        ]
    }

# Mock the `pandas.read_csv` method using pytest-mock
@pytest.fixture
def mock_read_csv(mock_data, mocker):
    mocker.patch('pandas.read_csv', side_effect=lambda x: pd.DataFrame(mock_data.get(os.path.basename(x), [])))

# Test for verifying the convergence plot creation
def test_run_creates_convergence_plot(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 5
    
    # Mock savefig function
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')

    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)
    
    # Check if savefig was called with the expected arguments
    mock_savefig.assert_called_once_with(results_directory + "/convergence-F1.png", bbox_inches="tight")

# Test for verifying that the function processes the right data and makes correct plot calls
def test_run_handles_data_processing(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 5
    
    # Mock the plot, savefig, and legend functions
    mock_plot = mocker.patch('matplotlib.pyplot.plot')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_legend = mocker.patch('matplotlib.pyplot.legend')

    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)

    # Assert that plt.plot() was called the correct number of times (one per optimizer)
    assert mock_plot.call_count == len(optimizer)

    # Assert that plt.legend() was called with the correct arguments
    mock_legend.assert_called_with(
        loc="upper right",
        bbox_to_anchor=(1.2, 1.02)
    )

    # Assert that savefig was called
    mock_savefig.assert_called()

# Test that plt.clf() is called after each plot is saved
def test_run_clears_figure(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 5
    
    # Mock the clf function
    mock_clf = mocker.patch('matplotlib.pyplot.clf')
    
    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)
    
    # Check if plt.clf() was called once
    mock_clf.assert_called_once()

if __name__ == "__main__":
    pytest.main()

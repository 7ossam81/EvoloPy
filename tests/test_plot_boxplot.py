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
from EvoloPy.plot_boxplot import run
results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
Path(results_directory).mkdir(parents=True, exist_ok=True)

# Mocking file read data for testing purposes
@pytest.fixture
def mock_data():
    # Mock data for testing
    return {
        'experiment_details.csv': [
            {"Optimizer": "SSA", "objfname": "F1", "Iter50": 0.25, "Iter100": 0.3},
            {"Optimizer": "PSO", "objfname": "F1", "Iter50": 0.5, "Iter100": 0.4},
            {"Optimizer": "GA", "objfname": "F1", "Iter50": 0.35, "Iter100": 0.45},
            # Add more mock data as necessary
        ]
    }

# Mock the `pandas.read_csv` method using pytest-mock
@pytest.fixture
def mock_read_csv(mock_data, mocker):
    mocker.patch('pandas.read_csv', side_effect=lambda x: pd.DataFrame(mock_data.get(os.path.basename(x), [])))

# Test for verifying the boxplot generation
def test_run_creates_boxplot(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 100
    
    # Mock savefig function
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')

    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)
    
    # Check if savefig was called with the expected arguments
    mock_savefig.assert_called_once_with(results_directory + "/boxplot-F1.png", bbox_inches="tight")

# Test for verifying that the function processes the right data and makes correct plot calls
def test_run_handles_data_processing(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 100
    
    # Mock the boxplot and savefig functions
    mock_boxplot = mocker.patch('matplotlib.pyplot.boxplot')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_legend = mocker.patch('matplotlib.pyplot.legend')

    # Mock boxplot to return a dictionary with "boxes" as a key
    mock_boxes = [mocker.Mock(), mocker.Mock()]  # Mocked box elements
    mock_boxplot.return_value = {"boxes": mock_boxes}  # Returning a dictionary with boxes

    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)

    # Assert that plt.legend() was called with the correct arguments
    mock_legend.assert_called_with(
        handles=[mock_boxes[0], mock_boxes[1]],  # Directly use the mocked boxes
        labels=['SSA', 'PSO', 'GA'],
        bbox_to_anchor=(1.2, 1.02),
        loc='upper right'
    )

    # Assert that savefig was called
    mock_savefig.assert_called()
    
    colors = ["#5c9eb7","#f77199","#cf81d2","#4a5e6a","#f45b18","#ffbd35","#6ba5a1","#fcd1a1","#c3ffc1","#68549d","#1c8c44","#a44c40","#404636"]
    
    # Now, iterate over the mock boxes and check the behavior
    for patch, color in zip(mock_boxes, colors):
        patch.set_facecolor.assert_called_with(color)  # Verify the color is applied

# Test that plt.clf() is called after each boxplot is saved
def test_run_clears_figure(mock_read_csv, mocker):

    optimizer = ["SSA", "PSO", "GA"]
    objectivefunc = ["F1"]
    iterations = 100
    
    # Mock the clf function
    mock_clf = mocker.patch('matplotlib.pyplot.clf')
    
    # Run the function
    run(results_directory, optimizer, objectivefunc, iterations)
    
    # Check if plt.clf() was called once
    mock_clf.assert_called_once()

if __name__ == "__main__":
    pytest.main()

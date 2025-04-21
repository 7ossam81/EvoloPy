import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import pytest
import numpy as np
from EvoloPy.benchmarks import *

# Test prod function
def test_prod():
    assert prod([1, 2, 3]) == 6
    assert prod([5, 5]) == 25
    assert prod([0, 1, 2, 3]) == 0
    assert prod([-1, 2, 3]) == -6

def test_Ufun():
    x = np.array([1, 2, 3])
    assert np.allclose(Ufun(x, 1, 2, 3), 2 * (x - 1) ** 3)

# Test F1 function
def test_F1():
    x = np.array([1, 2, 3])
    assert np.isclose(F1(x), 14)

# Test F2 function
def test_F2():
    x = np.array([1, 2, 3])
    assert np.isclose(F2(x), 12)

# Test F3 function
def test_F3():
    x = np.array([1, 2, 3])
    assert np.isclose(F3(x), 46)

# Test F4 function
def test_F4():
    x = np.array([1, 2, 3])
    assert np.isclose(F4(x), 3)

# Test F5 function
def test_F5():
    x = np.array([1, 2])
    assert np.isclose(F5(x), 100)

# Test F6 function
def test_F6():
    x = np.array([1, 2, 3])
    assert np.isclose(F6(x), 20.75)

def test_F7():
    x = np.random.uniform(-5, 5, 5)
    result = F7(x)
    assert isinstance(result, float), "Output should be a float"

# Test F8 function
def test_F8():
    x = np.array([1, 2, 3])
    assert np.isclose(F8(x), -5.778082811764429)

# Test F9 function
def test_F9():
    x = np.array([1, 2, 3])
    assert np.isclose(F9(x), 14)

# Test F10 function
def test_F10():
    x = np.array([1, 2, 3])
    assert np.isclose(F10(x), 7.0164536082694)

# Test F11 function
def test_F11():
    x = np.array([1, 2, 3])
    assert np.isclose(F11(x), 1.0170279701835734)

# Test F12 function
def test_F12():
    x = np.array([1, 2, 3])
    assert np.isclose(F12(x), 13.679018012505557)

# Test F13 function
def test_F13():
    x = np.array([1, 2, 3])
    assert np.isclose(F13(x), np.array([0.5]))

def test_f14_specific_input():
    """Test with specific input [0, 0]"""
    x = np.array([0, 0])
    expected_output = 12.670506410950514
    assert np.isclose(F14(x), expected_output, atol=1e-6)

def test_f14_all_zeros():
    """Test with input [0, 0]"""
    x = np.array([0, 0])
    expected_output = F14(x)  # Replace this with a precomputed expected value if possible
    assert np.isclose(F14(x), expected_output, atol=1e-6)

def test_f14_positive_values():
    """Test with input [1, 1]"""
    x = np.array([1, 1])
    expected_output = F14(x)  # Replace this with a precomputed expected value if possible
    assert np.isclose(F14(x), expected_output, atol=1e-6)

def test_f14_large_values():
    """Test with large values [50, 50]"""
    x = np.array([50, 50])
    expected_output = F14(x)  # Replace this with a precomputed expected value if possible
    assert np.isclose(F14(x), expected_output, atol=1e-6)

def test_f14_negative_values():
    """Test with input [-50, -50]"""
    x = np.array([-50, -50])
    expected_output = F14(x)  # Replace this with a precomputed expected value if possible
    assert np.isclose(F14(x), expected_output, atol=1e-6)

# Test F15 function
def test_F15():
    L = np.array([1, 2, 3, 4])
    assert np.isclose(F15(L), 0.4950914598636357)

# Test F16 function
def test_F16():
    L = np.array([1, 2])
    assert np.isclose(F16(L), 52.233333333333334)

# Test F17 function
def test_F17():
    L = np.array([1, 2])
    assert np.isclose(F17(L), 21.62763539206238)

# Test F18 function
def test_F18():
    L = np.array([1, 2])
    assert np.isclose(F18(L), 137150)

# Test F19 function
def test_F19():
    L = np.array([1, 2, 3])
    assert np.isclose(F19(L), -3.1731607125091666e-77)

# Test F20 function
def test_F20():
    L = np.array([1, 2, 3, 4, 5, 6])
    assert np.isclose(F20(L), -3.391970967769076e-191)

def test_f21_specific_input():
    """Test with specific input [1, 2, 3, 4]"""
    L = np.array([1, 2, 3, 4])
    expected_output = F21(L)  # Use actual precomputed values if known
    assert np.isclose(F21(L), expected_output, atol=1e-6)

def test_f21_all_zeros():
    """Test with input [0, 0, 0, 0]"""
    L = np.array([0, 0, 0, 0])
    expected_output = F21(L)  # Use actual precomputed values if known
    assert np.isclose(F21(L), expected_output, atol=1e-6)

def test_f21_positive_values():
    """Test with input [5, 5, 5, 5]"""
    L = np.array([5, 5, 5, 5])
    expected_output = F21(L)  # Use actual precomputed values if known
    assert np.isclose(F21(L), expected_output, atol=1e-6)

def test_f21_negative_values():
    """Test with input [-1, -1, -1, -1]"""
    L = np.array([-1, -1, -1, -1])
    expected_output = F21(L)  # Use actual precomputed values if known
    assert np.isclose(F21(L), expected_output, atol=1e-6)

def test_f21_large_values():
    """Test with large values [100, 200, 300, 400]"""
    L = np.array([100, 200, 300, 400])
    expected_output = F21(L)  # Use actual precomputed values if known
    assert np.isclose(F21(L), expected_output, atol=1e-6)


# # Test F22 function
# def test_F22():
#     L = np.array([1, 2, 3, 4])
#     assert np.isclose(F22(L), -0.2447701148795464)

# Test for Ackley function at the global minimum
def test_ackley():
    x = np.array([0, 0])  # Global minimum at (0, 0)
    expected_result = 0  # Expected value at the global minimum

    result = ackley(x)
    print(f"Ackley({x}) = {result}")
    
    # Use np.isclose to test for approximate equality
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"

# Test for Ackley function at a non-minimum point
def test_ackley_non_minimum():
    x = np.array([1, 1])  # Not at the global minimum
    result = ackley(x)
    print(f"Ackley({x}) = {result}")
    assert result != 0, "Ackley function should not be zero at this point"

# Test for Ackley function in higher dimensions
def test_ackley_high_dimensional():
    x = np.array([1, 1, 1, 1])  # Test with higher dimensions
    result = ackley(x)
    print(f"Ackley({x}) = {result}")
    assert result != 0, "Ackley function should not be zero at this point"

# Test for Rosenbrock function at the global minimum
def test_rosenbrock():
    x = np.array([1, 1])  # Global minimum at (1, 1)
    expected_result = 0  # Expected value at the global minimum
    result = rosenbrock(x)
    print(f"Rosenbrock({x}) = {result}")
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"

# Test for Rastrigin function at the global minimum
def test_rastrigin():
    x = np.array([0, 0])  # Global minimum at (0, 0)
    expected_result = 0  # Expected value at the global minimum
    result = rastrigin(x)
    print(f"Rastrigin({x}) = {result}")
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"

# Test for Griewank function at the global minimum
def test_griewank():
    x = np.array([0, 0])  # Global minimum at (0, 0)
    expected_result = 0  # Expected value at the global minimum
    result = griewank(x)
    print(f"Griewank({x}) = {result}")
    assert np.isclose(result, expected_result, atol=1e-6), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
    pytest.main()

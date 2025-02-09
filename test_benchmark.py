import numpy
from benchmarks import ackley, rosenbrock, rastrigin, griewank

# Test functions with a random input vector
x = numpy.random.uniform(-10, 10, 5)

def test_functions():
    print("Ackley:", ackley(x))
    print("Rosenbrock:", rosenbrock(x))
    print("Rastrigin:", rastrigin(x))
    print("Griewank:", griewank(x))

if __name__ == "__main__":
    test_functions()

import unittest
import numpy as np
from EvoloPy.optimizers import PSO, GWO, MVO
from EvoloPy.benchmarks import F1, F2, F3

class TestOptimizers(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters common for all optimizers"""
        self.lb = -10
        self.ub = 10
        self.dim = 5
        self.population_size = 10
        self.iterations = 20
    
    def test_pso_optimizer(self):
        """Test that PSO optimizer returns a valid solution"""
        result = PSO.PSO(F1, self.lb, self.ub, self.dim, self.population_size, self.iterations)
        
        # Check if result object has all required properties
        self.assertIsNotNone(result.bestIndividual)
        self.assertEqual(len(result.bestIndividual), self.dim)
        self.assertIsNotNone(result.convergence)
        self.assertEqual(len(result.convergence), self.iterations)
        self.assertEqual(result.optimizer, "PSO")
        self.assertEqual(result.objfname, "F1")
        
        # Check if the solution is within bounds
        self.assertTrue(np.all(result.bestIndividual >= self.lb))
        self.assertTrue(np.all(result.bestIndividual <= self.ub))
    
    def test_gwo_optimizer(self):
        """Test that GWO optimizer returns a valid solution"""
        result = GWO.GWO(F1, self.lb, self.ub, self.dim, self.population_size, self.iterations)
        
        # Check if result object has all required properties
        self.assertIsNotNone(result.bestIndividual)
        self.assertEqual(len(result.bestIndividual), self.dim)
        self.assertIsNotNone(result.convergence)
        self.assertEqual(len(result.convergence), self.iterations)
        self.assertEqual(result.optimizer, "GWO")
        self.assertEqual(result.objfname, "F1")
        
        # Check if the solution is within bounds
        self.assertTrue(np.all(result.bestIndividual >= self.lb))
        self.assertTrue(np.all(result.bestIndividual <= self.ub))
    
    def test_mvo_optimizer(self):
        """Test that MVO optimizer returns a valid solution"""
        result = MVO.MVO(F1, self.lb, self.ub, self.dim, self.population_size, self.iterations)
        
        # Check if result object has all required properties
        self.assertIsNotNone(result.bestIndividual)
        self.assertEqual(len(result.bestIndividual), self.dim)
        self.assertIsNotNone(result.convergence)
        self.assertEqual(len(result.convergence), self.iterations)
        self.assertEqual(result.optimizer, "MVO")
        self.assertEqual(result.objfname, "F1")
        
        # Check if the solution is within bounds
        self.assertTrue(np.all(result.bestIndividual >= self.lb))
        self.assertTrue(np.all(result.bestIndividual <= self.ub))
    
    def test_multiple_benchmarks(self):
        """Test that optimizers work with different benchmark functions"""
        benchmarks = [F1, F2, F3]
        
        for benchmark in benchmarks:
            result = PSO.PSO(benchmark, self.lb, self.ub, self.dim, self.population_size, self.iterations)
            self.assertEqual(result.objfname, benchmark.__name__)
            self.assertEqual(len(result.bestIndividual), self.dim)

if __name__ == '__main__':
    unittest.main() 
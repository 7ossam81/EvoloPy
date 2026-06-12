# EvoloPy – ACO Addition

This is a clone of the [EvoloPy](https://github.com/7ossam81/EvoloPy) optimization framework, originally developed by Faris, Aljarah, Mirjalili, Castillo, and Guervós. All credit for the original framework and its algorithms belongs to the original authors.

## What was added

A new optimizer, **Ant Colony Optimization for Continuous Domains (ACO_R)**, has been implemented and added to the `optimizers` folder as `ACO.py`. The algorithm is based on the work of Socha & Dorigo (2008) and follows the same interface conventions as the existing optimizers in the package (PSO, GWO, etc.).

### Implementation notes

The core of ACO_R is a solution archive ranked by fitness. At each iteration, new candidate solutions are sampled from Gaussians centred on archive solutions, with the spread (sigma) computed as the average distance between archive members. A rank-based Gaussian weighting scheme biases selection toward better-ranked solutions.

A few corrections were made relative to a naive implementation of the algorithm:

- **Sigma formula**: The standard deviation used for Gaussian sampling follows Eq. 5 of Socha & Dorigo (2008) — an unweighted mean distance divided by `k - 1`. Using pheromone weights inside the sigma calculation (a common mistake) causes premature convergence.
- **Archive size**: Scales with problem dimension (`max(50, 10 * dim)`) rather than being fixed, which matters for higher-dimensional problems.
- **Convergence speed (`xi`)**: Set to `0.65` rather than the commonly cited `0.85`, which works better with moderate archive sizes.

## Usage

```python
from EvoloPy.optimizers.ACO import ACO

result = ACO(
    objf=my_function,
    lb=-100,
    ub=100,
    dim=30,
    PopSize=50,
    iters=500
)

print(result.best_score)
print(result.bestIndividual)
```

## References

- Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains. *European Journal of Operational Research*, 185(3), 1155–1173.
- Faris, H., Aljarah, I., Mirjalili, S., Castillo, P. A., & Guervós, J. J. M. (2016). EvoloPy: An open-source nature-inspired optimization framework in Python. *IJCCI (ECTA)*, 171–177.
# DASED

DASED is a Python package for optimizing DAS (Distributed Acoustic Sensing) cable layouts for geophysical applications. It provides tools to design, visualize, and optimize fiber-optic cable deployments to maximize information gain for geoscientific problems.

> **Warning:** DASED is under active development with no comprehensive test, features beyond tutorial examples and the accompanying paper (publication pending) may not be fully tested, and breaking changes may occur at any time.

## Key Features

- **Flexible Layout Parameterization**: Define DAS cable layouts using knot points with spline interpolation
- **Evolutionary Optimization**: Use population-based algorithms to find optimal cable configurations
- **Multiple Design Criteria**: Currently supports probabilistic source location and linearised travel time tomography
- **Constraint Handling**: Incorporate spatial constraints, minimum knot distances, and deployment boundaries

## Installation

To install the package, simply run:

```bash
pip install git+https://github.com/dominik-strutz/dased.git
```

Alternatively, you can use:

```bash
uv add git+https://github.com/dominik-strutz/dased.git
```

> **Note:** The package is still in development and can change rapidly. If you want to use it, it is recommended to fix the version by running:
>
> ```bash
> pip install git+https://github.com/dominik-strutz/dased@<version>
> ```
>
> where `<version>` is the version you want to use (e.g., a commit hash or a tag).

## Quick Start

### Creating a DAS Layout

Define a DAS cable layout by specifying knot points and channel spacing:

```python
import numpy as np
from dased.layout import DASLayout

# Define knot points for the cable path
knot_points = np.array([
    [0, 0],        # Starting point
    [100, 200],    # First waypoint
    [300, 150],    # Second waypoint
    [500, 300]     # End point
])

# Create a DAS layout with 20m channel spacing
layout = DASLayout(
    knots=knot_points,
    spacing=20,  # 20 meter channel spacing
    k=1          # Linear spline interpolation
)

print(f"Cable length: {layout.cable_length:.1f} m")
print(f"Number of channels: {layout.n_channels}")

# Visualize the layout
layout.plot(show_knots=True)
```

### Optimizing a Layout

Use evolutionary algorithms to optimize cable layouts for maximum information gain:

```python
import numpy as np
from dased.optimisation import DASOptimizationProblem, DASArchipelago
from dased.criteria import EIGCriterion

# Define the optimization criterion
criterion = EIGCriterion(
    samples=prior_samples, # model parameter samples from prior distribution
    data_likelihood=likelihood_function, # function to compute data likelihood
    eig_method="NMC" # method for estimating expected information gain
)

# Set up the optimization problem
problem = DASOptimizationProblem(
    design_criterion=criterion,
    N_knots=6, 
    bounds=[[0, 1000], [0, 1000]],  # deployment area bounds
    cable_length=1500.0,
    spacing=25.0,
    spatial_constraints=forbidden_polygon  # optional shapely polygon
)

# Create an archipelago for parallel optimization
archipelago = DASArchipelago(
    problem=problem,
    n_islands=8,
    population_size=128
)

# Initialize with random or proposal layouts
archipelago.initialize()

# Optimize the population
archipelago.optimize(n_generations=100)

# Get the best layout
best_layout = archipelago.get_best()
best_layout.plot(show_knots=True)
```

## Documentation

For detailed tutorials and API documentation, visit the full documentation at [dased.readthedocs.io](https://dased.readthedocs.io).

Specific sections:
- **Tutorials**: [dased.readthedocs.io/tutorials](https://dased.readthedocs.io/en/latest/tutorials.html)
- **API Reference**: [dased.readthedocs.io/api](https://dased.readthedocs.io/en/latest/api_reference.html)

## Citation

If you use DASED in your research, please cite the accompanying paper (publication pending).

## Contributing

Contributions are welcome! This package is under active development. Please note:

- There is no comprehensive test suite yet
- Features beyond tutorial examples may not be fully tested
- Breaking changes may occur at any time

## License

See the [LICENSE](LICENSE) file for details.
"""
This module provides optimization tools for DAS layouts using evolutionary
algorithms. It includes population-based optimization with parallel
evaluation, constraint handling, and multi-objective support.
"""

import gc
import logging
import os
import warnings
from pickle import dumps, loads
from threading import Lock as _Lock

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg
import tqdm.autonotebook as tqdm_module
from joblib import Parallel, delayed, wrap_non_picklable_objects
from scipy.interpolate import make_splprep
from shapely.geometry import MultiPoint, Point, box
from shapely.ops import nearest_points
from tqdm.auto import tqdm

from .layout import DASLayout

__all__ = [
    "DASOptimizationProblem",
    "DASArchipelago",
    "InitialPopulation",
    "joblib_island",
]

# --- Constants ---
DEFAULT_PENALTY = np.inf  # Penalty for invalid design variables in fitness evaluation
GEOMETRY_BUFFER = 1e-9  # Small buffer for Shapely geometric operations

# --- Logger Configuration ---
logger = logging.getLogger(__name__)


def _ensure_logger_handler(logger_instance: logging.Logger) -> None:
    """
    Ensure the logger has at least one handler for output.

    Arguments:
        logger_instance: Logger instance to configure
    """
    if not logger_instance.handlers:
        # Create a console handler if none exist
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)


# Ensure logger has a handler at module import
_ensure_logger_handler(logger)


def _setup_logger_level(logger_instance: logging.Logger, verbose: int) -> None:
    """
    Configure logging level based on verbosity setting.

    Arguments:
        logger_instance: Logger instance to configure
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
    """
    if verbose == 0:
        log_level = logging.WARNING
    elif verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
        logger_instance.warning(
            f"Invalid verbose level '{verbose}'. Defaulting to INFO."
        )

    # Ensure the logger has at least one handler for output
    if not logger_instance.handlers:
        # Create a console handler if none exist
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)

    logger_instance.setLevel(log_level)
    for handler in logger_instance.handlers:
        handler.setLevel(log_level)

    logger_instance.info(
        f"Logging level set to {logging.getLevelName(logger_instance.level)}"
    )


def _evolve_func_joblib(ser_algo_pop):
    """
    Evolution function executed by joblib for parallel processing.

    This function deserializes an algorithm and population, runs one evolution
    step, and returns the serialized result.

    Arguments:
        ser_algo_pop: Serialized tuple of (algorithm, population)

    Returns:
        Serialized tuple of (algorithm, evolved_population)
    """
    algo, pop = loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return dumps((algo, new_pop))


class joblib_island:
    """
    Custom PyGMO island using joblib for parallel evolution.

    This user-defined island (UDI) dispatches evolution tasks using joblib's
    parallel processing capabilities. It provides efficient parallelism for
    Python functions with automatic handling of NumPy arrays and efficient
    serialization through cloudpickle.

    The island manages a shared joblib.Parallel pool across all instances
    for resource efficiency, automatically handling initialization and
    cleanup of the parallel backend.

    Arguments:
        n_jobs: Number of parallel jobs to use. If None, uses all available
            CPU cores. Defaults to None.
        backend: Joblib backend for parallel execution. Options include
            'loky', 'multiprocessing', 'threading'. If None, uses joblib's
            default backend. Defaults to None.

    Examples:

        >>> # Create a joblib island (:class:`~dased.optimisation.joblib_island`)
        >>> island = joblib_island(n_jobs=4, backend='loky')  # :class:`~dased.optimisation.joblib_island`
        >>> # Use in PyGMO archipelago
        >>> arch = pg.archipelago()
        >>> arch.push_back(pg.island(algo=pg.sga(), pop=pop, udi=island))  # uses :mod:`pygmo`

    Note:
        The joblib pool is shared across all joblib_island instances and
        persists until explicitly shut down with shutdown_pool().
    """

    # Static variables for shared joblib pool
    _pool_lock = _Lock()
    _pool = None
    _n_jobs = None

    def __init__(self, n_jobs=None, backend=None):
        """
        Initialize joblib island with parallel processing parameters.

        Arguments:
            n_jobs: Number of jobs for parallel execution. If None, uses all
                available CPU cores. Must be an integer or None.
            backend: Joblib backend to use. Supported values are 'loky',
                'multiprocessing', 'threading', etc. If None, uses default.
                Must be a string or None.

        Raises:
            TypeError: If n_jobs is not None or an int, or if backend is not
                a str or None.
        """
        self._init(n_jobs, backend)

    def _init(self, n_jobs, backend):
        """
        Internal initialization method for joblib island parameters.

        Arguments:
            n_jobs: Number of parallel jobs (int or None)
            backend: Joblib backend name (str or None)

        Raises:
            TypeError: If parameters have incorrect types
        """
        if n_jobs is not None and not isinstance(n_jobs, int):
            raise TypeError("The 'n_jobs' parameter must be None or an int")

        if backend is not None and not isinstance(backend, str):
            raise TypeError("The 'backend' parameter must be None or a str")

        self._n_jobs = n_jobs
        self._backend = backend
        joblib_island.init_pool(n_jobs, backend)

    @staticmethod
    def _init_pool_impl(n_jobs, backend):
        """
        Internal implementation for pool initialization without locking.

        Arguments:
            n_jobs: Number of parallel jobs
            backend: Joblib backend name
        """
        if joblib_island._pool is None:
            joblib_island._pool = Parallel(n_jobs=n_jobs, backend=backend)
            joblib_island._n_jobs = n_jobs

    @staticmethod
    def init_pool(n_jobs=None, backend=None):
        """
        Initialize the shared joblib parallel backend.

        This method initializes the joblib Parallel object used by all
        joblib_island instances. If the pool is already initialized or
        was previously shut down, this will create a new pool.

        Arguments:
            n_jobs: Number of jobs to run in parallel. If None, uses all
                available CPU cores. Must be an integer or None.
            backend: Joblib backend to use. Options include 'loky',
                'multiprocessing', 'threading'. If None, uses default.
                Must be a string or None.
        """
        if n_jobs is not None and not isinstance(n_jobs, int):
            raise TypeError("The 'n_jobs' parameter must be None or an int")

        if backend is not None and not isinstance(backend, str):
            raise TypeError("The 'backend' parameter must be None or a str")

        with joblib_island._pool_lock:
            joblib_island._init_pool_impl(n_jobs, backend)

    @staticmethod
    def shutdown_pool():
        """
        Shutdown the shared joblib parallel backend.

        This method shuts down the joblib Parallel object used by all
        joblib_island instances. After shutdown, the next evolution will
        automatically create a new Parallel object.

        This is useful for cleaning up resources or changing parallel
        backend settings.
        """
        with joblib_island._pool_lock:
            if joblib_island._pool is not None:
                old_pool = joblib_island._pool
                joblib_island._pool = None
                joblib_island._n_jobs = None
                del old_pool
                gc.collect()

    def __copy__(self):
        """Create a shallow copy of the joblib island."""
        return joblib_island(self._n_jobs, self._backend)

    def __deepcopy__(self, d):
        """Create a deep copy of the joblib island."""
        return self.__copy__()

    def __getstate__(self):
        """Get state for pickling."""
        return (self._n_jobs, self._backend)

    def __setstate__(self, state):
        """Set state from unpickling."""
        self._init(*state)

    def run_evolve(self, algo, pop):
        """
        Evolve population using the specified algorithm in parallel.

        This method evolves the input population using the input algorithm
        and returns both the algorithm and evolved population. Evolution
        runs in a separate process using joblib's parallel processing.

        If the algorithm or population contains non-picklable objects,
        this method automatically uses joblib's wrap_non_picklable_objects
        to handle them properly.

        Arguments:
            algo: PyGMO algorithm instance to use for evolution
            pop: PyGMO population instance to evolve

        Returns:
            tuple: (algorithm, evolved_population) after one evolution step

        Raises:
            Exception: Any exception thrown by evolution or parallel processing
        """
        ser_algo_pop = cloudpickle.dumps((algo, pop))
        with joblib_island._pool_lock:
            if joblib_island._pool is None:
                joblib_island._init_pool_impl(self._n_jobs, self._backend)
            # Use the shared pool instead of creating a new Parallel object
            results = joblib_island._pool([delayed(_evolve_func_joblib)(ser_algo_pop)])
        return cloudpickle.loads(results[0])

    def get_name(self):
        """
        Get the name of this island type.

        Returns:
            str: Always returns "Joblib island"
        """
        return "Joblib island"

    def get_extra_info(self):
        """
        Get additional information about the island's configuration.

        Returns:
            str: String containing information about the parallel backend
                configuration including backend type and number of jobs
        """
        with joblib_island._pool_lock:
            if joblib_island._pool is None:
                return "\tNo parallel backend has been created yet"
            else:
                return "\tJoblib backend: {}\n\tNumber of jobs: {}".format(
                    "default" if self._backend is None else self._backend,
                    "auto" if self._n_jobs is None else self._n_jobs,
                )

class DASOptimizationProblem:
    """
    PyGMO-compatible optimization problem for DAS layout design.

    This class defines the optimization problem for designing DAS layouts, 
    including objective functions, constraints, and decision
    variable handling. It supports both single and multi-objective optimization
    with flexible constraint handling for spatial boundaries and cable length limits.

    The problem follows PyGMO conventions, providing the necessary interface methods
    for integration with PyGMO's optimization algorithms. Design parameters (decision variables in 
    PyGMO nomeclature) represent the coordinates of variable knot points along the DAS cable path.

    Arguments:
        design_criterion: Objective function(s) for optimization. Can be:

            - Callable: Single objective function f(layout) -> float
            - Dict: Multiple objectives {name: function} for multi-objective optimization

            Functions should accept a :class:`~dased.layout.DASLayout` object and return a scalar.
            Higher values indicate better solutions (maximization convention).

        bounds: Rectangular search area as [[x_min, x_max], [y_min, y_max]].
            All variable knot points must lie within these bounds.
            Should be a 2×2 array-like structure.

        N_knots: Number of variable knot points along the cable path.
            Total design parameters = 2 × N_knots (x and y coordinates).
            More knots allow more complex cable geometries but increase
            problem dimensionality.

        cable_length: Maximum allowed cable length in the same units as bounds.
            Layouts exceeding this length receive penalty fitness values.

        spacing: Spacing between DAS channels along the cable in same units
            as bounds. Determines the number of measurement channels
            and affects layout performance calculations results and computational cost. 

        fixed_points: Fixed knot points that don't vary during optimization. Total number of knots = N_knots + N_fixed_points
            Can be:

            - None: No fixed points (all knots are variable)
            - List/Tuple/Array: Points prepended to variable knots
            - Dict: {final_index: (x, y)} for precise placement. Inserted starting at index zero.

            Useful for constraining start/end points or intermediate waypoints.
            Defaults to None.

        spatial_constraints: Forbidden areas as Shapely Polygon/MultiPolygon.
            Cable paths intersecting these areas receive penalty fitness.
            Useful for avoiding obstacles, protected areas, or infeasible regions.
            Defaults to None.

        verbose: Logging verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
            Higher values provide more detailed optimization information.
            Defaults to 1.

        n_jobs: Number of parallel jobs for batch fitness evaluation (if supported).
            Only used by fitness_batch method. Defaults to 1.

        **kwargs: Additional parameters passed to DASLayout constructor.
            Common options include elevation data, field properties, and
            signal decay parameters.

    Attributes:
        design_criterion: Objective function(s)
        is_multi_objective: Whether problem has multiple objectives
        n_objectives: Number of objective functions
        criteria_names: Names of objective functions
        bounds: Search space boundaries
        N_knots: Number of variable knot points
        cable_length: Maximum cable length constraint
        spacing: Channel spacing parameter
        fixed_points_validated: Processed fixed points
        spatial_constraints: Forbidden area geometry
        allowed_area: Allowed area geometry (bounds minus constraints)

    Examples:
        Single-objective layout optimization::

            >>> def eig_criterion(layout):
            ...     return calculate_eig(layout)
            >>>
            >>> problem = DASOptimizationProblem(
            ...     design_criterion=eig_criterion,
            ...     bounds=[[0, 1000], [0, 1000]],
            ...     N_knots=8,
            ...     cable_length=1200,
            ...     spacing=10.0
            ... )

        Multi-objective optimization with constraints::

            >>> criteria = {
            ...     "eig": eig_criterion,
            ...     "sensitivity": sensitivity_criterion
            ... }
            >>> forbidden_area = Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])
            >>>
            >>> problem = DASOptimizationProblem(
            ...     design_criterion=criteria,
            ...     bounds=[[0, 1000], [0, 1000]],
            ...     N_knots=8,
            ...     cable_length=1500,
            ...     spacing=5.0,
            ...     fixed_points={(0, (100, 100), 9: (900, 900))},  # Fix start and end
            ...     spatial_constraints=forbidden_area,
            ...     verbose=2
            ... )

    Note:
        - The class implements PyGMO's problem interface (get_bounds, get_nobj, fitness)
        - Fitness values are minimized by PyGMO, so objectives are negated internally
        - Spatial constraints are handled through penalty methods
    """
    #TODO: properly test fixed points with dictionary

    # Class attribute declarations for Sphinx linkcode extension
    design_criterion = None
    is_multi_objective = None
    n_objectives = None
    criteria_names = None
    bounds = None
    N_knots = None
    cable_length = None
    spacing = None
    fixed_points_validated = None
    spatial_constraints = None
    allowed_area = None

    def __init__(
        self,
        design_criterion,
        bounds,
        N_knots,
        cable_length,
        spacing,
        fixed_points=None,
        spatial_constraints=None,
        verbose: int = 1,
        n_jobs: int = 1,
        **kwargs,
    ):
        self.verbose = verbose
        _setup_logger_level(logger, self.verbose)

        # Support both single callable and dictionary of callables for multi-objective
        if isinstance(design_criterion, dict):
            if not all(callable(func) for func in design_criterion.values()):
                raise TypeError(
                    "All design criteria in dictionary must be callable functions."
                )
            self.design_criterion = design_criterion
            self.is_multi_objective = True
            self.n_objectives = len(design_criterion)
            self.criteria_names = list(design_criterion.keys())
        elif callable(design_criterion):
            self.design_criterion = design_criterion
            self.is_multi_objective = False
            self.n_objectives = 1
            try:
                self.criteria_names = [design_criterion.__name__]
            except AttributeError:
                self.criteria_names = ["custom_criterion"]
        else:
            raise TypeError(
                "design_criterion must be a callable function or a dictionary of callables."
            )

        self.cable_length = float(cable_length)
        self.spacing = float(spacing)
        self.bounds = np.array(bounds)
        self.N_knots = int(N_knots)
        self.fixed_points_input = fixed_points  # Store original input
        self.spatial_constraints = spatial_constraints
        self.n_jobs = n_jobs

        self.cable_properties = {
            "spacing": self.spacing,
            **kwargs,
        }

        self._validate_core_parameters()

        self.fixed_points_validated = _validate_fixed_points(self.fixed_points_input)
        self.num_fixed_points = (
            len(self.fixed_points_validated) if self.fixed_points_validated else 0
        )
        self.total_points = self.N_knots + self.num_fixed_points

        self.allowed_area = _create_allowed_area_geometry(
            self.bounds, self.spatial_constraints
        )
        self._setup_variable_bounds()

    def _validate_core_parameters(self):
        """Validates the core numerical and structural parameters."""
        if not isinstance(self.cable_length, (int, float)) or self.cable_length <= 0:
            raise ValueError("cable_length must be a positive number.")
        if not isinstance(self.spacing, (int, float)) or self.spacing <= 0:
            raise ValueError("spacing must be a positive number.")
        if not isinstance(self.bounds, np.ndarray) or self.bounds.shape != (2, 2):
            raise ValueError(
                "bounds must be a 2x2 numpy array: [[x_min, x_max], [y_min, y_max]]."
            )
        if not isinstance(self.N_knots, int) or self.N_knots <= 0:
            raise ValueError(
                "N_knots (number of variable points) must be a positive integer."
            )

    def _setup_variable_bounds(self):
        """Sets up the lower (xl) and upper (xu) bounds for design parameters."""
        n_vars_per_knot = 2  # x and y coordinate
        self.n_vars = self.N_knots * n_vars_per_knot

        # Initialize bounds arrays
        self.xl = np.zeros(self.n_vars)
        self.xu = np.zeros(self.n_vars)

        # Set x-bounds for the first N_knots variables
        self.xl[0 : self.N_knots] = self.bounds[0, 0]
        self.xu[0 : self.N_knots] = self.bounds[0, 1]
        # Set y-bounds for the next N_knots variables
        self.xl[self.N_knots : 2 * self.N_knots] = self.bounds[1, 0]
        self.xu[self.N_knots : 2 * self.N_knots] = self.bounds[1, 1]

    def get_bounds(self):
        """
        Returns the lower and upper bounds for the design parameters.

        Returns:
            A tuple containing two lists: (lower_bounds, upper_bounds).
        """
        return (self.xl.tolist(), self.xu.tolist())

    def get_nobj(self):
        """
        Returns the number of objectives for the optimization problem.

        Returns:
            An integer representing the number of objectives.
        """
        return self.n_objectives

    def get_name(self) -> str:
        """Returns the name of the optimization problem."""
        return "DAS Layout Optimization"

    def get_extra_info(self) -> str:
        """Returns a string with extra information about the problem setup."""
        if self.is_multi_objective:
            criterion_name = f"Multi-Objective ({self.n_objectives})"
        else:
            criterion_name = getattr(
                self.design_criterion, "__name__", "custom_callable"
            )

        info = (
            f"VarKnots={self.N_knots}, "
            f"FixedPts={self.num_fixed_points}, "
            f"TotalPts={self.total_points}, "
            f"MaxLen={self.cable_length:.1f}, "
            f"Spacing={self.spacing:.1f}, "
            f"Criterion={criterion_name}, "
        )
        return info

    def _dv2layout(self, x):
        """
        Constructs and returns the DASLayout corresponding to a decision vector.

        Args:
            x: The decision vector.

        Returns:
            The corresponding :class:`~dased.layout.DASLayout` object, or None if creation fails.
        """
        knot_locations = self._dv2combined_knots(x)
        try:
            layout = DASLayout(
                knots=knot_locations,
                **self.cable_properties,
            )
            return layout
        except (ValueError, np.linalg.LinAlgError):
            # Handle cases where layout creation fails due to invalid knot configurations
            return None

    def _knots2layout(self, knot_locations):
        """
        Constructs and returns the DASLayout corresponding to a set of knot locations.

        Args:
            knot_locations: The knot locations (N_total, 2).

        Returns:
            The corresponding DASLayout object, or None if creation fails.
        """
        combined_knots = self._insert_fixed_points(knot_locations)

        try:
            layout = DASLayout(
                knots=combined_knots,
                **self.cable_properties,
            )
            return layout
        except (ValueError, np.linalg.LinAlgError):
            # Handle cases where layout creation fails due to invalid knot configurations
            return None

    def _layout2dv(self, layout):
        """
        Converts a DASLayout object to a decision vector.

        Args:
            layout: The :class:`~dased.layout.DASLayout` object.

        Returns:
            A numpy array representing the decision vector.
        """
        knot_locations = layout.knots

        # remove fixed points from the layout
        if self.fixed_points_validated is not None:
            # Remove fixed points from the layout
            fixed_points = np.array(self.fixed_points_validated, dtype=float)
            mask = np.all(np.isin(knot_locations, fixed_points), axis=1)
            variable_knot_locations = knot_locations[~mask]
        else:
            variable_knot_locations = knot_locations

        dv = self._variable_knots2dv(variable_knot_locations)
        return dv

    def _get_knots(self, x):
        """
        Constructs and returns the combined knot locations for a decision vector.

        Args:
            x: The decision vector.

        Returns:
            A numpy array of knot locations (N_total, 2), or None if conversion fails.
        """
        knot_locations = self._dv2combined_knots(x)
        return knot_locations

    def get_fixed_points(self):
        """
        Returns the fixed point coordinates (x, y) as a numpy array.

        Returns:
            A numpy array of shape (N_fixed, 2) containing the coordinates
            of the fixed points, or None if no fixed points are defined.
        """
        return self.fixed_points_validated

    def _check_feasibility(self, layout):
        max_allowed_length = self.cable_length
        if layout.cable_length > max_allowed_length:
            return False

        violates_spatial = self.check_spatial_constraints(layout)
        if violates_spatial:
            return False

        return True

    def fitness(self, x):
        """
        Calculates the fitness (objective function value) for a decision vector.

        For single-objective: returns the **negative** of the design criterion if valid.
        For multi-objective: returns the **negative** of each criterion if valid.
        PyGMO aims to minimize all objectives.

        Args:
            x: The decision vector representing the coordinates of variable knots.
               Format: [x1, x2, ..., xN, y1, y2, ..., yN].

        Returns:
            A tuple containing the fitness values (lower is better).
        """
        # Default penalties for invalid solutions
        if self.is_multi_objective:
            objective_values = [DEFAULT_PENALTY] * self.n_objectives
        else:
            objective_values = [DEFAULT_PENALTY]

        knot_locations = self._dv2combined_knots(x)
        # 1. Check if all knots are within the allowed area
        if not self.allowed_area.contains(MultiPoint(knot_locations)):
            return tuple(objective_values)

        try:
            layout = DASLayout(
                knots=knot_locations,
                **self.cable_properties,
            )
        except (ValueError, np.linalg.LinAlgError):
            # Handle cases where layout creation fails due to invalid knot configurations
            # (e.g., duplicate points, insufficient points, spline interpolation errors)
            return tuple(objective_values)

        if not self._check_feasibility(layout):
            return tuple(objective_values)

        if self.is_multi_objective:
            for i, (_, criterion) in enumerate(self.design_criterion.items()):
                value = criterion(layout)
                # Minimize negative criterion -> Maximize criterion
                objective_values[i] = -float(value)
        else:
            value = self.design_criterion(layout)
            # Minimize negative criterion -> Maximize criterion
            objective_values[0] = -float(value)

        return tuple(objective_values)

    def fitness_batch(self, x, n_jobs=None):
        try:
            from mpire import WorkerPool
        except ImportError:
            raise ImportError(
                "mpire is not installed. Please install it to use fitness_batch."
            )

        if n_jobs is None:
            n_jobs = self.n_jobs

        if n_jobs == 1:
            return np.array([self.fitness(xi) for xi in x])
        else:
            with WorkerPool(n_jobs=n_jobs) as pool:
                results = pool.map(self.fitness, x.tolist())
            return np.array(results)

    def _dv2variable_knots(self, x):
        """Converts the decision vector `x` into variable knot locations (N_knots, 2)."""
        dv = np.array(x, dtype=float)
        var_x = dv[0 : self.N_knots]
        var_y = dv[self.N_knots : 2 * self.N_knots]

        variable_knot_locations = np.column_stack((var_x, var_y))
        return variable_knot_locations

    def _variable_knots2dv(self, variable_knot_locations):
        """Converts variable knot locations (N_knots, 2) to a decision vector."""
        var_x = variable_knot_locations[:, 0]
        var_y = variable_knot_locations[:, 1]
        dv = np.concatenate((var_x, var_y))
        return dv

    def _insert_fixed_points(self, variable_knot_locations):
        """
        Combines variable knot locations with pre-validated fixed points.

        Handles both list/tuple (prepend) and dict (insert at index) formats
        for fixed points.

        Args:
            variable_knot_locations: Array of shape (N_knots, 2).

        Returns:
            Array of combined knot locations, shape (N_total, 2).
        """
        if self.fixed_points_validated is None:
            return variable_knot_locations  # No fixed points to insert

        n_fixed = self.num_fixed_points
        total_points = self.total_points
        final_locations = np.zeros((total_points, 2), dtype=float)

        if isinstance(self.fixed_points_validated, (list, tuple)):
            # Prepend fixed points (list/tuple format implies prepending)
            fixed_points_array = np.array(self.fixed_points_validated, dtype=float)
            final_locations[:n_fixed, :] = fixed_points_array
            final_locations[n_fixed:, :] = variable_knot_locations

        elif isinstance(self.fixed_points_validated, dict):
            final_locations_list = [None] * total_points
            occupied_indices = set()

            # Place fixed points first
            for final_index, coords in self.fixed_points_validated.items():
                final_locations_list[final_index] = list(coords)
                occupied_indices.add(final_index)

            # Fill remaining slots with variable points
            variable_point_iter = iter(variable_knot_locations)
            variable_points_used = 0
            for i in range(total_points):
                if final_locations_list[i] is None:
                    final_locations_list[i] = next(variable_point_iter).tolist()
                    variable_points_used += 1

            final_locations = np.array(final_locations_list, dtype=float)

        return final_locations

    def _dv2combined_knots(self, x):
        """Converts a decision vector `x` to the final combined knot locations."""
        variable_knots = self._dv2variable_knots(x)
        combined_knots = self._insert_fixed_points(variable_knots)
        return combined_knots

    def check_spatial_constraints(self, layout):
        """
        Checks if the layout violates spatial constraints (forbidden areas).

        Args:
            layout: The DASLayout object to check.
        Returns:
            True if the layout violates spatial constraints, False otherwise. 
        """
        violates = False

        shapely_cable = layout.get_shapely()
        # Check if the cable intersects with the forbidden areas (i.e., is not fully contained within the allowed area)
        # Use a small negative buffer for contains check robustness
        if not self.allowed_area.contains(shapely_cable):
            violates = True

        return violates

    def __str__(self):
        """Returns a string representation of the optimization problem."""
        if self.is_multi_objective:
            criteria_str = f"{len(self.design_criterion)} objectives"
        else:
            criteria_str = str(self.design_criterion)

        return (
            f"DASOptimizationProblem("
            f"design_criterion={criteria_str}, "
            f"bounds={self.bounds.tolist()}, "
            f"N_knots={self.N_knots}, "
            f"cable_length={self.cable_length}, "
            f"spacing={self.spacing}, "
            f"fixed_points={self.fixed_points_input}, "
            f"spatial_constraints={self.spatial_constraints}, "
            f"verbose={self.verbose})"
        )

    def __repr__(self):
        """Returns a string representation of the optimization problem."""
        return self.__str__()

    def __eq__(self, other):
        """Checks equality between two DASOptimizationProblem instances."""
        if not isinstance(other, DASOptimizationProblem):
            return False

        if self.is_multi_objective != other.is_multi_objective:
            return False

        if self.is_multi_objective:
            if set(self.design_criterion.keys()) != set(other.design_criterion.keys()):
                return False
            criterion_equal = all(
                self.design_criterion[k].__class__
                == other.design_criterion[k].__class__
                for k in self.design_criterion.keys()
            )
        else:
            criterion_equal = (
                self.design_criterion.__class__ == other.design_criterion.__class__
            )

        return (
            criterion_equal
            and np.array_equal(self.bounds, other.bounds)
            and self.N_knots == other.N_knots
            and self.cable_length == other.cable_length
            and self.spacing == other.spacing
            and np.array_equal(self.fixed_points_input, other.fixed_points_input)
            and self.spatial_constraints == other.spatial_constraints
        )


class DASArchipelago:
    """
    Multi-island evolutionary optimization framework for DAS layout problems.

    This class provides a high-level interface for running evolutionary optimization
    of DAS layouts using PyGMO's archipelago model. It supports both single and
    multi-objective optimization with configurable parallelization, migration
    patterns, and algorithm selection.

    The archipelago maintains multiple populations (islands) that evolve independently
    with periodic migration of individuals between islands. This approach often
    provides better exploration and can escape local optima more effectively than
    single-population algorithms.

    Arguments:
        problem: DASOptimizationProblem instance defining the optimization problem
            including design criteria, constraints, and problem dimensions.

        n_islands: Number of independent islands (populations).

        population_size: Size of each island's population. Total population
            across all islands will be n_islands × population_size.

        algo: Algorithm(s) to use for evolution. Can be:

            - None: Uses default algorithms (NSGA-II for multi-objective,
              SGA for single-objective)
            - Single PyGMO algorithm: Applied to all islands
            - List of algorithms: One per island (cycling if fewer than n_islands)

            Defaults to None.

        migration_topology: PyGMO topology defining migration patterns between
            islands. If None, uses ring topology. Common options:

            - ``pg.ring()``: Ring topology (default). Only neighboring islands exchange migrants
            - ``pg.fully_connected()``: All islands exchange migrants
            - ``pg.unconnected()``: No migration between islands

            Defaults to None (ring topology).

        verbose: Verbosity level for logging. Higher values provide more detail:

            - 0: Warnings only
            - 1: Info messages
            - 2+: Debug information

            Defaults to False (equivalent to 0).

        random_seed: Random seed for reproducible results. If provided,
            ensures deterministic optimization runs. Defaults to 0.

        island_parallelization: Type of parallelization for island evolution:
            Some types might lead to pickling issues with certain algorithms. Joblib seems to work best. Options:

            - "joblib": Custom joblib-based parallelization (recommended)
            - "multiprocessing": PyGMO's multiprocessing island
            - "ipyparallel": IPython parallel (for cluster computing)
            - None: No parallelization (sequential evolution)

            Defaults to "multiprocessing".

    Attributes:
        problem: The optimization problem instance
        n_obj: Number of optimization objectives
        is_multi_objective: Whether problem has multiple objectives
        n_dim: Number of design parameters
        archipelago: PyGMO archipelago instance (after initialization)

    Examples:
        Basic single-objective optimization::

            >>> problem = DASOptimizationProblem(criterion, bounds, n_knots, ...)
            >>> arch = DASArchipelago(problem, n_islands=8, population_size=100)
            >>> arch.initialize(proposal_points)
            >>> arch.optimize(n_generations=50)
            >>> best_layout = arch.get_best()

        Multi-objective optimization with custom algorithms::

            >>> criteria = {"eig": eig_criterion, "sensitivity": sens_criterion}
            >>> problem = DASOptimizationProblem(criteria, bounds, n_knots, ...)
            >>> algorithms = [pg.nsga2(gen=1), pg.moead(gen=1)]
            >>> arch = DASArchipelago(
            ...     problem, n_islands=6, population_size=80,
            ...     algo=algorithms, verbose=1
            ... )
            >>> arch.initialize(proposal_points)
            >>> arch.optimize(n_generations=100, migrate_every=10)
            >>> pareto_layouts, pareto_fitness = arch.get_pareto_front()
    """
    #TODO: check if examples are all working

    # Class attribute declarations for Sphinx linkcode extension
    problem = None
    n_obj = None
    is_multi_objective = None
    n_dim = None
    archipelago = None

    def __init__(
        self,
        problem,
        n_islands,
        population_size,
        algo=None,
        migration_topology=None,
        verbose=1,
        random_seed=0,
        island_parallelization="multiprocessing",
    ):
        _setup_logger_level(logger, verbose)

        print(f"Logging level: {logging.getLevelName(logger.level)}")

        # test logger
        logger.debug("Initializing DASArchipelago")
        logger.info("DASArchipelago initialization started")

        # Store core problem information
        self.problem = problem
        self.n_obj = self.problem.get_nobj()
        self.is_multi_objective = self.n_obj > 1
        self.n_dim = self._get_problem_dim()

        # Store archipelago configuration
        self.n_islands = n_islands
        self.population_size = population_size
        self.random_seed = random_seed

        # Configure algorithms
        self.algo = self._process_algorithms_input(algo)

        # Configure migration topology
        self.migration_topology = migration_topology

        # Configure island parallelization
        if island_parallelization == "multiprocessing":
            self.island_parallelization = pg.mp_island()
        elif island_parallelization == "ipyparallel":
            self.island_parallelization = pg.ipyparallel_island()
        elif island_parallelization == "joblib":
            self.island_parallelization = joblib_island(self.n_islands)
        else:
            self.island_parallelization = None

    def _get_problem_dim(self):
        """
        Get the number of design parameters from the optimization problem.

        Returns:
            int: Number of design parameters (problem dimension)
        """
        return len(self.problem.get_bounds()[0])

    def _process_algorithms_input(self, algo):
        """
        Process and validate algorithm input, setting up algorithm list for islands.

        Converts various algorithm input formats to a list of PyGMO algorithms,
        one for each island. Handles single algorithms, lists, and default
        algorithm selection.

        Arguments:
            algo: Algorithm specification (None, single algorithm, or list)

        Returns:
            list: List of PyGMO algorithms, one per island
        """
        if algo is None:
            algo = self._set_default_algorithms()

        if not isinstance(algo, list):
            algo = [algo] * self.n_islands
        elif len(algo) != self.n_islands:
            # Cycle through provided algorithms to fill all islands
            for i in range(len(algo), self.n_islands):
                algo.append(algo[i % len(algo)])
        else:
            # Algorithm validation error - this should be ValueError, not else
            if len(algo) > self.n_islands:
                raise ValueError(
                    "More algorithms provided than islands. "
                    f"Expected at most {self.n_islands}, got {len(algo)}."
                )

        # Validate and convert algorithms to PyGMO format
        for i in range(len(algo)):
            if not isinstance(algo[i], pg.algorithm):
                algo[i] = pg.algorithm(algo[i])
            if not algo[i].has_set_seed():
                algo[i].set_seed(i + 1)

        # Ensure all algorithms have seeds set
        for i, a in enumerate(algo):
            if not a.has_set_seed():
                a.set_seed(i + 1)

        return algo

    def _set_default_algorithms(self):
        """
        Set default algorithms based on problem type.

        Returns:
            PyGMO algorithm: NSGA-II for multi-objective, SGA for single-objective
        """
        if self.is_multi_objective:
            return pg.nsga2(gen=1, m=0.2, eta_m=10, seed=self.random_seed)
        else:
            return pg.sga(
                gen=1,
                crossover="sbx",
                m=0.2,
                param_m=10,
                param_s=self.population_size // 4,
                mutation="polynomial",
                selection="truncated",
                seed=self.random_seed,
            )

    def initialize(
        self,
        proposal_points,
        perturb_proposal=0.0,
        perturb_knots=0.0,
        random_t=True,
        proposal_weights=None,
        corr_len=0,
        corr_str=1.0,
        min_length=0,
        max_attempts=100_000,
        n_jobs=None,
        show_progress=False,
        filename=None,
        **kwargs,
    ):
        """
        Initialize archipelago populations from proposal points with intelligent sampling.

        Creates initial populations for all islands using the InitialPopulation class
        to generate diverse, feasible DAS layouts. Supports loading/saving populations
        for resuming optimization runs.

        Arguments:
            proposal_points: Initial layout proposals for population generation. Can be:

                - numpy.ndarray: Single set of points (N×2) or multiple sets (M×N×2)
                - list: List of point arrays for multiple proposal sets

                These points serve as starting points for generating the initial population through
                perturbation and interpolation.

            perturb_proposal: Standard deviation for Gaussian noise added to proposal
                points during population generation. Higher values increase diversity
                but may violate constraints. Defaults to 0.0.

            perturb_knots: Standard deviation for Gaussian noise added to knot
                points after interpolation. Defaults to 0.0.

            random_t: Whether to use random parameter values when interpolating along
                proposal point splines. If False, uses evenly spaced parameters.
                Defaults to True.

            proposal_weights: Relative weights for sampling from multiple proposal
                point sets. Must be non-negative and sum to positive value.
                If None, uses uniform weights. Defaults to None.

            corr_len: Correlation length for spatially correlated knot perturbations.
                Larger values create smoother perturbations along cable path.
                If 0, perturbations are uncorrelated. Defaults to 0.

            corr_str: Correlation strength (0-1) for knot perturbations. 1.0 means
                fully correlated, 0.0 means uncorrelated. Defaults to 1.0.

            min_length: Minimum acceptable cable length for generated layouts.
                Layouts shorter than this are rejected. Defaults to 0.

            max_attempts: Maximum attempts to generate each valid individual.
                Higher values improve success rate but increase computation time.
                Defaults to 100,000.

            n_jobs: Number of parallel jobs for population generation and fitness
                evaluation. If None, uses number of islands. Defaults to None.

            show_progress: Whether to display progress bars during initialization.
                Defaults to False.

            filename: Base filename for saving/loading population data (without
                extension). If provided and file exists, attempts to load previous
                population. Always saves final population. Defaults to None.

            **kwargs: Additional arguments passed to spline interpolation
                (scipy.interpolate.splprep).

        Raises:
            ValueError: If not enough valid individuals can be generated or if
                proposal weights are invalid.

        Examples:
            Basic initialization::

                >>> arch.initialize(proposal_points, perturb_proposal=10.0)

            Multi-proposal initialization with weights::

                >>> proposals = [straight_line, curved_path, complex_geometry]  # list of :class:`~numpy.ndarray` or similar structures
                >>> weights = [0.5, 0.3, 0.2]
                >>> arch.initialize(proposals, proposal_weights=weights,
                ...                 perturb_knots=5.0, show_progress=True)

            Resume from checkpoint::

                >>> arch.initialize(proposal_points, filename="optimization_run")
        """
        #TODO: allow loading without defining proposal points?
        
        n_jobs = n_jobs if n_jobs is not None else self.n_islands
        
        current_dim = self._get_problem_dim()
        if self.n_dim != current_dim:
            self.n_dim = current_dim
        islands_list = []
        file_valid = False
        if filename is not None and os.path.exists(f"{filename}.npz"):
            try:
                data = np.load(f"{filename}.npz", allow_pickle=True)
                individuals = data["individuals"]
                fitness_values_loaded = [fv.tolist() for fv in data["fitness_values"]]
                if (
                    data["n_islands"] == self.n_islands
                    and data["population_size"] == self.population_size
                ):
                    file_valid = True
                else:
                    pass
            except Exception:
                pass

        if not file_valid:
            pop_generator = InitialPopulation(
                problem=self.problem,
                proposal_points=proposal_points,
                perturb_proposal=perturb_proposal,
                perturb_knots=perturb_knots,
                random_t=random_t,
                proposal_weights=proposal_weights,
                corr_len=corr_len,
                corr_str=corr_str,
                min_length=min_length,
                max_attempts=max_attempts,
                random_seed=self.random_seed,
                n_jobs=n_jobs,
                show_progress=show_progress,
                **kwargs,
            )
            individuals = pop_generator.get_decision_vectors(
                self.population_size * self.n_islands
            )
            if n_jobs > 1:
                # Wrap the problem.fitness method if it's not picklable
                fitness_func = wrap_non_picklable_objects(self.problem.fitness)

                fitness_values_loaded = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(fitness_func)(ind_vec)
                    for ind_vec in tqdm(
                        individuals,
                        desc="Computing fitness",
                        disable=not show_progress,
                        leave=False,
                    )
                )
            else:
                fitness_values_loaded = [
                    self.problem.fitness(ind_vec)
                    for ind_vec in tqdm(
                        individuals,
                        desc="Computing fitness",
                        disable=not show_progress,
                        leave=False,
                    )
                ]
            fitness_values_loaded = np.stack(fitness_values_loaded, axis=0)
            if filename is not None:
                np.savez(
                    f"{filename}.npz",
                    individuals=np.array(individuals),  # :class:`~numpy.ndarray` of decision vectors
                    fitness_values=fitness_values_loaded,
                    n_islands=self.n_islands,
                    population_size=self.population_size,
                )
        if len(individuals) < self.n_islands * self.population_size:
            raise ValueError(
                f"Not enough individuals generated. Expected {self.n_islands * self.population_size}, got {len(individuals)}."
            )
        for i in range(self.n_islands):
            pop_seed = self.random_seed + i
            pop = pg.population(prob=self.problem, seed=pop_seed)
            start_idx = i * self.population_size
            end_idx = (i + 1) * self.population_size
            current_individuals_slice = individuals[start_idx:end_idx]
            current_fitness_slice = fitness_values_loaded[start_idx:end_idx]
            for k_idx in range(len(current_individuals_slice)):
                pop.push_back(
                    x=current_individuals_slice[k_idx], f=current_fitness_slice[k_idx]
                )

            if self.island_parallelization is not None:
                islands_list.append(
                    pg.island(
                        algo=self.algo[i], pop=pop, udi=self.island_parallelization
                    )
                )
            else:
                islands_list.append(pg.island(algo=self.algo[i], pop=pop))

        self.archipelago = pg.archipelago(
            t=pg.unconnected(),
            seed=self.random_seed
            if self.random_seed is not None
            else np.random.randint(1, 10000),
        )
        for island_obj in islands_list:
            self.archipelago.push_back(island_obj)

    #TODO: there are some errors whith population size 1 (raise proper error) and only one island (should be made to work)

    def optimize(
        self,
        n_generations=100,
        migrate_every=None,
        filename=None,
        show_progress=True,
        track_fitness_history=True,
        track_param_history=True,
    ):
        """
        Run evolutionary optimization for the specified number of generations.

        Executes the main optimization loop with optional migration between islands,
        progress tracking, and checkpointing. Supports resuming from saved states
        and provides comprehensive fitness/parameter history tracking.

        Arguments:
            n_generations: Total number of generations to run. Each generation
                involves one evolution step per island. Must be positive.
                Defaults to 100.

            migrate_every: Frequency of migration events between islands. If None,
                no migration occurs (islands evolve independently). If positive
                integer, migration happens every N generations. Migration can
                improve exploration and solution quality. Defaults to None.

            filename: Base filename for checkpoint saving/loading (without extension).
                If provided:

                - Attempts to load previous state at start
                - Saves state every 10 generations during optimization
                - Saves final state at completion

                Enables resuming interrupted optimizations. Defaults to None.

            show_progress: Whether to display progress bar with fitness information
                during optimization. Shows best fitness and migration events.
                Defaults to True.

            track_fitness_history: Whether to record fitness values throughout
                optimization. Required for fitness history analysis and plotting.
                Stored data can be retrieved with get_fitness_history().
                Defaults to True. Might add memory overhead for large runs.

            track_param_history: Whether to record decision variable values
                throughout optimization. Required for parameter evolution analysis.
                Stored data can be retrieved with get_param_history().
                Defaults to True. Might add memory overhead for large runs.

        Returns:
            int: Total number of generations completed, including any loaded
                from checkpoint files.

        Examples:
            Basic optimization::

                >>> total_gens = arch.optimize(n_generations=100)

            Optimization with migration and checkpointing::

                >>> total_gens = arch.optimize(
                ...     n_generations=500,
                ...     migrate_every=25,
                ...     filename="das_optimization",
                ...     show_progress=True
                ... )

            Resume interrupted optimization::

                >>> # This will load previous state and continue
                >>> total_gens = arch.optimize(
                ...     n_generations=1000,
                ...     filename="das_optimization"  # Same filename as before
                ... )

        Raises:
            ValueError: If archipelago not initialized (call initialize() first)

        """
        #TODO: save_every should be configurable
        
        # Initialize history tracking
        self._init_history(track_fitness_history, track_param_history)
        self.migrate_every = migrate_every

        # Load state from file if available
        start_generation = 0
        total_completed_generations = 0

        if filename is not None:
            total_completed_generations, loaded_successfully = self._load_state(
                filename
            )

            # If we loaded a file with more generations than requested, just return
            if loaded_successfully and total_completed_generations >= n_generations:
                logger.info(
                    f"Loaded state has {total_completed_generations} generations, "
                    f"which is >= {n_generations} requested. No additional optimization needed."
                )

            # Otherwise, start from where we left off
            if loaded_successfully:
                start_generation = total_completed_generations
                logger.info(
                    f"Continuing optimization from generation {start_generation}"
                )

        # Configure migration topology
        migrations_enabled = migrate_every is not None and migrate_every > 0
        current_migration_topology = self.migration_topology

        if migrations_enabled:
            if current_migration_topology is None:
                current_migration_topology = pg.ring(n=self.n_islands)
            elif isinstance(current_migration_topology, pg.unconnected):
                warnings.warn(
                    "Migration topology is unconnected. No migration will occur."
                )
        else:
            current_migration_topology = pg.unconnected()

        isolated_topology = pg.unconnected()
        self.archipelago.set_topology(isolated_topology)

        # Run evolution for the required number of generations
        for i in (
            pbar := tqdm(
                range(start_generation, n_generations),
                desc="Evolving islands",
                disable=not show_progress,
                leave=False,
            )
        ):
            current_gen_num = i + 1

            # Handle migration if enabled
            if migrations_enabled and current_gen_num % self.migrate_every == 0:
                self.archipelago.set_topology(current_migration_topology)
            else:
                self.archipelago.set_topology(isolated_topology)

            # Evolve one generation
            self.archipelago.evolve(1)
            self.archipelago.wait_check()

            # Record history and log progress
            self._record_history_and_log(current_gen_num, n_generations, pbar)

            # Periodic save (every 10 generations)
            if filename is not None and current_gen_num % 10 == 0:
                self._save_state(filename, current_gen_num)

        # Final save if we completed more generations than previously stored
        if filename is not None and n_generations > total_completed_generations:
            self._save_state(filename, n_generations)

    def _init_history(self, track_fitness_history, track_param_history):
        if not hasattr(self, "archipelago"):
            raise ValueError(
                "Archipelago not initialized. Call initialize() before optimize()."
            )
        if not hasattr(self, "_fitness_history") or self._fitness_history is None:
            self._fitness_history = [] if track_fitness_history else None
        if not hasattr(self, "_param_history") or self._param_history is None:
            self._param_history = [] if track_param_history else None

    def _record_history_and_log(self, current_gen_num, n_generations, pbar):
        if self.is_multi_objective:
            dv_gen, fit_gen = self._get_all_current_solutions()
            non_dominated_indices = pg.fast_non_dominated_sorting(
                fit_gen.reshape(-1, self.n_obj)
            )[0][0]
            summary_fit = fit_gen.reshape(-1, self.n_obj)[non_dominated_indices]
            summary_dvs = dv_gen.reshape(-1, self.n_dim)[non_dominated_indices]
            summary_fit = np.array(sorted(summary_fit, key=lambda x: tuple(x)))
            summary_dvs = np.array(sorted(summary_dvs, key=lambda x: tuple(x)))
            best_ind = pg.select_best_N_mo(fit_gen.reshape(-1, self.n_obj), 1)[0]
            best_fit = fit_gen.reshape(-1, self.n_obj)[best_ind]
            log_msg_fitness = (
                f"Gen {current_gen_num}/{n_generations} | Current best: ("
                + " ,".join(f"{-f:.4g}" for f in best_fit)
                + ") | Pareto Front: "
                + " | ".join(
                    [
                        "(" + ", ".join(f"{-fval:.4g}" for fval in fi) + ")"
                        for fi in summary_fit
                    ]
                )
            )
            logger.info(log_msg_fitness)
            pbar.set_postfix({"Best Overall": " ".join(f"{-f:.4g}" for f in best_fit)})
            if self._fitness_history is not None:
                self._fitness_history.append(summary_fit.copy())
            if self._param_history is not None:
                self._param_history.append(summary_dvs.copy())
        else:
            island_best_fit = np.array(self.archipelago.get_champions_f()).squeeze()
            island_best_dvs = np.array(self.archipelago.get_champions_x()).squeeze()
            best_ind = np.argmin(island_best_fit)
            best_fit = island_best_fit[best_ind].item()
            pbar.set_postfix({"Best Overall": f"{-best_fit:.4g}"})
            best_per_island_str = " | ".join(
                [
                    f"{-fval:.4g}" if np.isfinite(fval) else "N/A"
                    for fval in island_best_fit
                ]
            )
            log_msg_fitness = f"Current best: {-best_fit:.4g}"
            logger.info(
                f"Gen {current_gen_num}/{n_generations} | {log_msg_fitness} | Island bests: {best_per_island_str}"
            )
            if self._fitness_history is not None:
                self._fitness_history.append(island_best_fit.copy())
            if self._param_history is not None:
                self._param_history.append(island_best_dvs.copy())

    def _load_state(self, filename):
        start_generation = 0
        loaded_successfully = False
        
        # Load from single file format
        if os.path.exists(f"{filename}.npz"):
            try:
                # Load all data from single file
                data = np.load(f"{filename}.npz", allow_pickle=True)
                
                # Extract metadata
                metadata = data["metadata"].item()
                
                # Verify compatibility
                if (
                    metadata.get("n_islands") == self.n_islands
                    and metadata.get("population_size") == self.population_size
                    and metadata.get("n_obj") == self.n_obj
                ):
                    # Reconstruct archipelago
                    self._reconstruct_archipelago_from_data(
                        data["decision_vectors"], 
                        data["fitness_values"],
                        metadata
                    )
                    
                    # Load histories
                    if self._fitness_history is not None:
                        self._fitness_history = data.get("fitness_history", []).tolist()
                    if self._param_history is not None:
                        self._param_history = data.get("param_history", []).tolist()
                    
                    start_generation = metadata.get("completed_generations", 0)
                    loaded_successfully = True
                    logger.info(f"Successfully loaded state from {filename}.npz")
                    
                else:
                    logger.warning("Incompatible archipelago configuration in saved file")
                    
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
                
        return start_generation, loaded_successfully

    def _save_state(self, filename, completed_generations):
        try:
            # Extract all population data from archipelago
            decision_vectors = []
            fitness_values = []
            
            for island in self.archipelago:
                pop = island.get_population()
                decision_vectors.append(np.array(pop.get_x()))
                fitness_values.append(np.array(pop.get_f()))
            
            # Convert lists to numpy arrays
            decision_vectors = np.array(decision_vectors, dtype=object)
            fitness_values = np.array(fitness_values, dtype=object)
            
            # Prepare history data
            fitness_history = (
                np.array(self._fitness_history, dtype=object) 
                if self._fitness_history is not None 
                else np.array([])
            )
            param_history = (
                np.array(self._param_history, dtype=object)
                if self._param_history is not None
                else np.array([])
            )
            
            # Prepare metadata
            metadata = {
                "completed_generations": completed_generations,
                "n_islands": self.n_islands,
                "population_size": self.population_size,
                "n_obj": self.n_obj,
                "random_seed": getattr(self, 'random_seed', None),
                "algo_names": [str(algo.get_name()) for algo in self.algo] if hasattr(self, 'algo') else [],
                "migration_topology": str(type(self.migration_topology).__name__) if hasattr(self, 'migration_topology') and self.migration_topology else None,
                "version": "1.0"  # Version for future compatibility
            }
            
            # Save everything to a single file
            np.savez_compressed(
                f"{filename}.npz",
                decision_vectors=decision_vectors,
                fitness_values=fitness_values,
                fitness_history=fitness_history,
                param_history=param_history,
                metadata=metadata
            )
            
            logger.info(f"Successfully saved state to {filename}.npz")
                
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
            pass

    def _reconstruct_archipelago_from_data(self, decision_vectors, fitness_values, metadata):
        """
        Reconstruct the archipelago from saved population data.
        
        Arguments:
            decision_vectors: Array of decision vectors for each island
            fitness_values: Array of fitness values for each island  
            metadata: Dictionary containing archipelago configuration
        """
        try:
            # Create new archipelago with same configuration
            self.archipelago = pg.archipelago(
                t=pg.unconnected(),
                seed=self.random_seed if self.random_seed is not None else np.random.randint(1, 10000)
            )
            
            # Reconstruct each island
            for i in range(self.n_islands):
                # Create population with same seed as original
                pop_seed = self.random_seed + i if self.random_seed is not None else i + 1
                pop = pg.population(prob=self.problem, seed=pop_seed)
                
                # Add individuals back to population
                island_dvs = decision_vectors[i]
                island_fits = fitness_values[i]
                
                for dv, fit in zip(island_dvs, island_fits):
                    pop.push_back(x=dv, f=fit)
                
                # Create island with algorithm
                if self.island_parallelization is not None:
                    island = pg.island(
                        algo=self.algo[i], 
                        pop=pop, 
                        udi=self.island_parallelization
                    )
                else:
                    island = pg.island(algo=self.algo[i], pop=pop)
                
                self.archipelago.push_back(island)
                
        except Exception as e:
            logger.error(f"Failed to reconstruct archipelago: {e}")
            raise

    def _get_all_current_solutions(self, flatten=False):
        if not hasattr(self, "archipelago"):
            empty_f = (
                np.array([]).reshape(0, self.n_obj)
                if self.is_multi_objective
                else np.array([])
            )
            return np.array([]), empty_f
        dvs, fs = [], []
        for island in self.archipelago:
            pop = island.get_population()
            x, f = pop.get_x(), pop.get_f()
            dvs.append(np.array(x))
            fs.append(np.array(f))
        dvs = np.stack(dvs, axis=0)
        fs = np.stack(fs, axis=0)
        if flatten:
            dvs = dvs.reshape(-1, self.n_dim)
            fs = fs.reshape(-1, self.n_obj) if self.is_multi_objective else fs.flatten()
        return dvs, fs

    def get_current_layouts(self, flatten=True):
        """
        Return current layouts (decoded decision vectors) from the archipelago.
        """
        dvs, _ = self._get_all_current_solutions(flatten=flatten)
        if flatten:
            return [self.problem._dv2layout(dv) for dv in dvs]
        else:
            return [
                [self.problem._dv2layout(dv) for dv in island_dvs] for island_dvs in dvs
            ]

    def get_current_fitness(self, flatten=True):
        """
        Return current fitness values (maximized) from the archipelago.
        """
        _, current_fitness = self._get_all_current_solutions(flatten=flatten)
        return -1 * current_fitness

    def get_current_decision_vectors_and_fitness(self, flatten=True):
        """
        Return current decision vectors and fitness values from the archipelago.
        Fitness values are always maximized (user-facing).
        
        Returns:
            dvs: Decision vectors (2D array if flatten=True, else list of 2D arrays)
            fs: Fitness values (2D array if flatten=True, else list of 2D arrays)
        """
        dvs, fs = self._get_all_current_solutions(flatten=flatten)
        fs = -1 * np.array(fs)  # always maximized for user
        return dvs, fs

    def get_history_decision_vectors_and_fitness(self):
        """
        Return all decision vectors and fitness values from the optimization history.
        Fitness values are always maximized (user-facing).
        
        Returns:
            dvs: Decision vectors (2D array)
            fs: Fitness values (2D array)
        """
        dvs, fs = self._flatten_history_or_population()
        fs = -1 * np.array(fs)
        return dvs, fs

    def _flatten_history_or_population(self):
        """
        Flattens the history arrays (_param_history and _fitness_history) into 2D arrays.
        Returns (decision_vectors, fitness_values).
        """
        if (
            hasattr(self, "_param_history")
            and self._param_history is not None
            and len(self._param_history) > 0
        ):
            dvs = np.concatenate(
                [
                    np.atleast_2d(arr)
                    for arr in self._param_history
                    if arr is not None and len(arr) > 0
                ],
                axis=0,
            )
        else:
            dvs = np.array([])
        if (
            hasattr(self, "_fitness_history")
            and self._fitness_history is not None
            and len(self._fitness_history) > 0
        ):
            fs = np.concatenate(
                [
                    np.atleast_2d(arr)
                    for arr in self._fitness_history
                    if arr is not None and len(arr) > 0
                ],
                axis=0,
            )
        else:
            fs = np.array([])
        return dvs, fs

    def get_best(self, **kwargs):
        """
        Get the best solution(s) from the archipelago.
        For single-objective problems, returns the best solution.
        For multi-objective problems, returns the Pareto front.
        
        Arguments:
            **kwargs: Additional arguments passed to get_best_single or get_pareto_front.
        Returns:
            For single-objective: (layout, fitness) or layout or fitness based on kwargs.
            For multi-objective: (layouts, fitnesses) or layouts or fitnesses based on kwargs.
        """
        if self.is_multi_objective:
            return self.get_best_multi(**kwargs)
        else:
            return self.get_best_single(**kwargs)

    def get_best_single(
        self, from_history=False, return_layout=True, return_fitness=True
    ):
        """
        Get the best solution (maximized fitness) for single-objective problems.
        
        Arguments:
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return the decoded layout. Defaults to True.
            return_fitness: Whether to return the fitness value. Defaults to True.
        Returns:
            Depending on return_layout and return_fitness:
            - (layout, fitness) if both True
            - layout if return_layout is True
            - fitness if return_fitness is True
            - None if both are False
        """
        if self.is_multi_objective:
            raise ValueError("Use get_best_multi for multi-objective problems.")
        dvs, fs = (
            self.get_history_decision_vectors_and_fitness()
            if from_history
            else self.get_current_decision_vectors_and_fitness(flatten=True)
        )
        if len(dvs) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fs = np.array(fs).flatten()
        dvs = np.array(dvs)
        idx = np.argmax(fs)  # maximized
        layout = self.problem._dv2layout(dvs[idx]) if return_layout else None
        fitness = fs[idx] if return_fitness else None
        if return_layout and return_fitness:
            return layout, fitness
        elif return_layout:
            return layout
        elif return_fitness:
            return fitness
        else:
            return None

    def get_n_best_single(
        self,
        N=1,
        similarity_tol=None,
        from_history=False,
        return_layout=True,
        return_fitness=True,
    ):
        """
        Get N best (maximized) solutions for single-objective problems, optionally filtering by similarity.
        
        Arguments:
            N: Number of top solutions to return. Defaults to 1.
            similarity_tol: Minimum Euclidean distance between decision vectors to be considered unique.
                If None, no similarity filtering is applied. Defaults to None.
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return decoded layouts. Defaults to True.
            return_fitness: Whether to return fitness values. Defaults to True.
            
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False
        """
        if self.is_multi_objective:
            raise ValueError("Use get_n_best_multi for multi-objective problems.")
        dvs, fs = (
            self.get_history_decision_vectors_and_fitness()
            if from_history
            else self.get_current_decision_vectors_and_fitness(flatten=True)
        )
        if len(dvs) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fs = np.array(fs).flatten()
        dvs = np.array(dvs)
        sort_idx = np.argsort(-fs)  # maximized
        dvs = dvs[sort_idx]
        fs = fs[sort_idx]
        # Similarity filtering
        if similarity_tol is not None and self.n_dim > 0:
            unique_dvs, unique_fs = [], []
            for dv, f in zip(dvs, fs):
                if len(unique_dvs) >= N:
                    break
                if all(
                    np.linalg.norm(np.array(dv) - np.array(udv)) >= similarity_tol
                    for udv in unique_dvs
                ):
                    unique_dvs.append(dv)
                    unique_fs.append(f)
            dvs = np.array(unique_dvs)
            fs = np.array(unique_fs)
        else:
            dvs = dvs[:N]
            fs = fs[:N]
        layouts = [self.problem._dv2layout(dv) for dv in dvs] if return_layout else None
        if return_layout and return_fitness:
            return layouts, fs
        elif return_layout:
            return layouts
        elif return_fitness:
            return fs
        else:
            return None

    def get_n_spread_single(
        self, N=5, from_history=False, return_layout=True, return_fitness=True
    ):
        """
        Get N solutions spread as much as possible across the fitness range (single-objective).
        Uses linear spacing between min and max fitness.
        
        Arguments:
            N: Number of solutions to return. Defaults to 5.
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return decoded layouts. Defaults to True.
            return_fitness: Whether to return fitness values. Defaults to True.
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False
        """
        if self.is_multi_objective:
            raise ValueError("Use get_n_spread_multi for multi-objective problems.")
        dvs, fs = (
            self.get_history_decision_vectors_and_fitness()
            if from_history
            else self.get_current_decision_vectors_and_fitness(flatten=True)
        )
        if len(dvs) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fs = np.array(fs).flatten()
        dvs = np.array(dvs)
        if len(fs) <= N:
            idxs = np.arange(len(fs))
        else:
            min_f, max_f = np.min(fs), np.max(fs)
            targets = np.linspace(min_f, max_f, N)
            idxs = [np.argmin(np.abs(fs - t)) for t in targets]
            idxs = np.unique(idxs)
        layouts = (
            [self.problem._dv2layout(dvs[i]) for i in idxs] if return_layout else None
        )
        fitnesses = fs[idxs] if return_fitness else None
        if return_layout and return_fitness:
            return layouts, fitnesses
        elif return_layout:
            return layouts
        elif return_fitness:
            return fitnesses
        else:
            return None

    def get_pareto_front(
        self, from_history=False, return_layout=True, return_fitness=True
    ):
        """
        Get the non-dominated (Pareto) front for multi-objective problems.
        Fitness values are always maximized (user-facing).
        
        Arguments:
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return decoded layouts. Defaults to True.
            return_fitness: Whether to return fitness values. Defaults to True.
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False
        """
        if not self.is_multi_objective:
            raise ValueError("Use get_best_single for single-objective problems.")
        dvs, fs = (
            self.get_history_decision_vectors_and_fitness()
            if from_history
            else self.get_current_decision_vectors_and_fitness(flatten=True)
        )
        if len(dvs) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fs = np.array(fs)
        dvs = np.array(dvs)
        if fs.ndim == 1:
            fs = fs.reshape(-1, self.n_obj)
        nd_idx = pg.fast_non_dominated_sorting(-fs)[0][0]  # sort for maximized
        dvs_nd = dvs[nd_idx]
        fs_nd = fs[nd_idx]
        layouts = (
            [self.problem._dv2layout(dv) for dv in dvs_nd] if return_layout else None
        )
        fitnesses = fs_nd if return_fitness else None
        if return_layout and return_fitness:
            return layouts, fitnesses
        elif return_layout:
            return layouts
        elif return_fitness:
            return fitnesses
        else:
            return None

    def get_best_multi(
        self,
        from_history=False,
        return_layout=True,
        return_fitness=True,
        method="halfway",
    ):
        """
        Get the best solution on the Pareto front according to an aggregate function.

        Fitness values are always maximized (user-facing).

        Arguments:
            from_history: Whether to use historical data instead of current population
            return_layout: Whether to return layout objects
            return_fitness: Whether to return fitness values
            method: Aggregation method for selecting best solution:

                - "sum": maximizes sum of objectives
                - "minmax": maximizes the maximum objective
                - "compromise": maximally distant from line between endpoints
                - "halfway": closest to midpoint between best and worst for each objective
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False
        """
        layouts, fitnesses = self.get_pareto_front(
            from_history=from_history, return_layout=True, return_fitness=True
        )
        if not layouts or len(fitnesses) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fitnesses = np.array(fitnesses)
        if method == "sum":
            idx = np.argmax(np.sum(fitnesses, axis=1))
        elif method == "minmax":
            idx = np.argmax(np.max(fitnesses, axis=1))
        elif method == "compromise":
            # Find endpoints (best for each objective)
            best_obj_idxs = [
                np.argmax(fitnesses[:, i]) for i in range(fitnesses.shape[1])
            ]
            endpoints = [fitnesses[idx] for idx in best_obj_idxs]
            # If endpoints are (almost) identical, pick point furthest from this point
            if np.allclose(endpoints[0], endpoints[1], atol=1e-12):
                dists = [np.linalg.norm(f - endpoints[0]) for f in fitnesses]
                idx = np.argmax(dists)
            else:
                # Perpendicular distance from each point to the line between endpoints
                p1, p2 = endpoints
                x1, y1 = p1
                x2, y2 = p2
                denom = np.hypot(y2 - y1, x2 - x1)
                if denom < 1e-12:
                    dists = [np.linalg.norm(f - p1) for f in fitnesses]
                    idx = np.argmax(dists)
                else:
                    # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - y2 * x1
                    dists = [np.abs(A * f[0] + B * f[1] + C) / denom for f in fitnesses]
                    idx = np.argmax(dists)
        elif method == "halfway":
            # Find best and worst for each objective
            best = np.max(fitnesses, axis=0)
            worst = np.min(fitnesses, axis=0)
            halfway = (best + worst) / 2.0
            dists = [np.linalg.norm(f - halfway) for f in fitnesses]
            idx = np.argmin(dists)
        else:
            raise ValueError("Unknown method.")
        layout = layouts[idx] if return_layout else None
        fitness = fitnesses[idx] if return_fitness else None
        if return_layout and return_fitness:
            return layout, fitness
        elif return_layout:
            return layout
        elif return_fitness:
            return fitness
        else:
            return None

    def get_n_best_multi(
        self,
        N=5,
        similarity_tol=None,
        from_history=False,
        return_layout=True,
        return_fitness=True,
    ):
        """
        Get N best (maximized aggregate) solutions from the Pareto front, optionally filtering by similarity.
        Fitness values are always maximized (user-facing).
        
        Arguments:
            N: Number of top solutions to return. Defaults to 5.
            similarity_tol: Minimum Euclidean distance between decision vectors to be considered unique.
                If None, no similarity filtering is applied. Defaults to None.
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return decoded layouts. Defaults to True.
            return_fitness: Whether to return fitness values. Defaults to True.
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False    
        """
        layouts, fitnesses = self.get_pareto_front(
            from_history=from_history, return_layout=True, return_fitness=True
        )
        if not layouts or len(fitnesses) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fitnesses = np.array(fitnesses)
        sort_idx = np.argsort(-np.sum(fitnesses, axis=1))
        layouts = np.array(layouts)[sort_idx]
        fitnesses = fitnesses[sort_idx]
        # Similarity filtering
        if similarity_tol is not None and self.n_dim > 0:
            unique_layouts, unique_fits = [], []
            for layout, f in zip(layouts, fitnesses):
                if len(unique_layouts) >= N:
                    break
                dv = self.problem._layout2dv(layout)
                if all(
                    np.linalg.norm(
                        np.array(dv) - np.array(self.problem._layout2dv(layout_))
                    )
                    >= similarity_tol
                    for layout_ in unique_layouts
                ):
                    unique_layouts.append(layout)
                    unique_fits.append(f)
            layouts = unique_layouts
            fitnesses = unique_fits
        else:
            layouts = layouts[:N]
            fitnesses = fitnesses[:N]
        if return_layout and return_fitness:
            return layouts, fitnesses
        elif return_layout:
            return layouts
        elif return_fitness:
            return fitnesses
        else:
            return None

    def get_n_spread_multi(
        self, N=5, from_history=False, return_layout=True, return_fitness=True
    ):
        """
        Get N solutions spread as much as possible across the Pareto front (multi-objective).
        Always includes endmembers (solutions that maximize each objective) and fills the rest
        using a greedy max-min approach in fitness space.
        Fitness values are always maximized (user-facing).
        
        Arguments:
            N: Number of solutions to return. Defaults to 5.
            from_history: Whether to consider all historical solutions (True) or
                only current population (False). Defaults to False.
            return_layout: Whether to return decoded layouts. Defaults to True.
            return_fitness: Whether to return fitness values. Defaults to True.
        Returns:
            Depending on return_layout and return_fitness:
            - (layouts, fitnesses) if both True
            - layouts if return_layout is True
            - fitnesses if return_fitness is True
            - None if both are False
        """
        layouts, fitnesses = self.get_pareto_front(
            from_history=from_history, return_layout=True, return_fitness=True
        )
        if not layouts or len(fitnesses) == 0:
            return ([], []) if (return_layout and return_fitness) else []

        fitnesses = np.array(fitnesses)
        selected_idx = []

        # If we have fewer solutions than requested, return all of them
        if len(fitnesses) <= N:
            selected_idx = list(range(len(fitnesses)))
        else:
            # First select endmembers (best solution for each objective)
            for obj_idx in range(self.n_obj):
                idx = np.argmax(fitnesses[:, obj_idx])
                if idx not in selected_idx:
                    selected_idx.append(idx)

            # If we need more points, use greedy max-min approach to fill remaining slots
            if len(selected_idx) < N:
                selected = [fitnesses[i] for i in selected_idx]
                while len(selected_idx) < N:
                    dists = []
                    for i, f in enumerate(fitnesses):
                        if i in selected_idx:
                            dists.append(-np.inf)
                        else:
                            min_dist = np.min([np.linalg.norm(f - s) for s in selected])
                            dists.append(min_dist)
                    next_idx = np.argmax(dists)
                    selected.append(fitnesses[next_idx])
                    selected_idx.append(next_idx)

        # Order by values, first by first objective, then by second (lexicographical order)
        selected_idx_sorted = sorted(selected_idx, key=lambda i: tuple(fitnesses[i]))
        layouts_out = (
            [layouts[i] for i in selected_idx_sorted] if return_layout else None
        )
        fitnesses_out = (
            [fitnesses[i] for i in selected_idx_sorted] if return_fitness else None
        )

        if return_layout and return_fitness:
            return layouts_out, fitnesses_out
        elif return_layout:
            return layouts_out
        elif return_fitness:
            return fitnesses_out
        else:
            return None

    def get_endmembers(
        self,
        method="halfway",
        from_history=False,
        return_layout=True,
        return_fitness=True,
    ):
        """
        Get the endmembers of the Pareto front according to an aggregate function.

        Fitness values are always maximized (user-facing).

        Arguments:
            method: Aggregation method for selecting endmembers:

                - "sum": maximizes sum of objectives
                - "minmax": maximizes the maximum objective
                - "compromise": maximally distant from line between endpoints
                - "halfway": closest to midpoint between best and worst for each objective

            from_history: Whether to use historical data instead of current population
            return_layout: Whether to return layout objects
            return_fitness: Whether to return fitness values

        Returns:
            Dictionary of endmember solutions with objective names as keys
        """
        layouts, fitnesses = self.get_pareto_front(
            from_history=from_history, return_layout=True, return_fitness=True
        )
        if not layouts or len(fitnesses) == 0:
            return ([], []) if (return_layout and return_fitness) else []
        fitnesses = np.array(fitnesses)
        if method == "sum":
            idx = np.argmax(np.sum(fitnesses, axis=1))
        elif method == "minmax":
            idx = np.argmax(np.max(fitnesses, axis=1))
        elif method == "compromise":
            # Find endpoints (best for each objective)
            best_obj_idxs = [
                np.argmax(fitnesses[:, i]) for i in range(fitnesses.shape[1])
            ]
            endpoints = [fitnesses[idx] for idx in best_obj_idxs]
            # If endpoints are (almost) identical, pick point furthest from this point
            if np.allclose(endpoints[0], endpoints[1], atol=1e-12):
                dists = [np.linalg.norm(f - endpoints[0]) for f in fitnesses]
                idx = np.argmax(dists)
            else:
                # Perpendicular distance from each point to the line between endpoints
                p1, p2 = endpoints
                x1, y1 = p1
                x2, y2 = p2
                denom = np.hypot(y2 - y1, x2 - x1)
                if denom < 1e-12:
                    dists = [np.linalg.norm(f - p1) for f in fitnesses]
                    idx = np.argmax(dists)
                else:
                    # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - y2 * x1
                    dists = [np.abs(A * f[0] + B * f[1] + C) / denom for f in fitnesses]
                    idx = np.argmax(dists)
        elif method == "halfway":
            # Find best and worst for each objective
            best = np.max(fitnesses, axis=0)
            worst = np.min(fitnesses, axis=0)
            halfway = (best + worst) / 2.0
            dists = [np.linalg.norm(f - halfway) for f in fitnesses]
            idx = np.argmin(dists)
        else:
            raise ValueError("Unknown method.")

        results = {}
        if hasattr(self.problem, "criteria_names"):
            obj_names = list(self.problem.criteria_names)
        else:
            obj_names = [f"Objective_{i}" for i in range(fitnesses.shape[1])]

        # Endmembers for each objective (best for that objective)
        for i, name in enumerate(obj_names):
            idx_obj = np.argmax(fitnesses[:, i])
            layout_obj = layouts[idx_obj] if return_layout else None
            fitness_obj = fitnesses[idx_obj] if return_fitness else None
            if return_layout and return_fitness:
                results[name] = {"layout": layout_obj, "fitness": fitness_obj}
            elif return_layout:
                results[name] = layout_obj
            elif return_fitness:
                results[name] = fitness_obj

        # Endmember for aggregate method
        agg_name = "compromise"
        layout_agg = layouts[idx] if return_layout else None
        fitness_agg = fitnesses[idx] if return_fitness else None
        if return_layout and return_fitness:
            results[agg_name] = {"layout": layout_agg, "fitness": fitness_agg}
        elif return_layout:
            results[agg_name] = layout_agg
        elif return_fitness:
            results[agg_name] = fitness_agg

        return results

    def get_fitness_history(self):
        """
        Return the fitness history (maximized) as a numpy array.
        
        Returns:
            Numpy array of fitness history. Each entry corresponds to a generation.
            For single-objective: shape (n_generations, n_islands)
            For multi-objective: shape (n_generations,), each entry is an array of ND solutions.
            Empty generations are represented as empty arrays.
        """
        
        if self._fitness_history is None or not self._fitness_history:
            return np.array([])
        maximized_history = [
            item if isinstance(item, np.ndarray) and item.size > 0 else item
            for item in self._fitness_history
        ]
        return -1 * np.array(maximized_history, dtype=object)

    def get_param_history(self):
        """
        Return the parameter (decision vector) history as a numpy array.

        Returns:
            Numpy array of parameter history. Each entry corresponds to a generation.
            Each entry is an array of decision vectors for the best individuals per island.
            Empty generations are represented as empty arrays.
        """
        if self._param_history is None or not self._param_history:
            return np.array([])
        return np.array(self._param_history, dtype=object)

    def plot_fitness_history(self):
        """
        Plot the fitness history for all generations.
        For multi-objective: plots all ND solutions and best per objective.
        For single-objective: plots best per island and overall.
        """
        fitness_history_maximized = self.get_fitness_history()
        if (
            not isinstance(fitness_history_maximized, np.ndarray)
            or fitness_history_maximized.size == 0
        ):
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_title, y_label = "Fitness History", "Fitness"
        num_generations = len(fitness_history_maximized)
        ax.ax2 = None
        if self.is_multi_objective:
            all_nd_solutions = [
                gen for gen in fitness_history_maximized if gen.size > 0
            ]
            if not all_nd_solutions:
                plt.close(fig)
                return
            colors = plt.cm.tab10.colors
            alpha_all = 0.2
            lw_best = 2
            num_gens = len(fitness_history_maximized)
            all_obj_points = [[] for _ in range(self.n_obj)]
            best_obj_per_gen = [[] for _ in range(self.n_obj)]

            # Collect all fitness values for normalization
            all_fitness_values = []
            for gen in fitness_history_maximized:
                if gen.size > 0:
                    all_fitness_values.append(gen)

            if all_fitness_values:
                all_fitness = np.vstack(all_fitness_values)
                # Calculate min and max for each objective for normalization
                obj_min = np.min(all_fitness, axis=0)
                obj_max = np.max(all_fitness, axis=0)
                obj_range = obj_max - obj_min
                # Avoid division by zero
                obj_range = np.where(obj_range == 0, 1, obj_range)
            else:
                obj_min = np.zeros(self.n_obj)
                obj_range = np.ones(self.n_obj)

            for gen_idx, gen in enumerate(fitness_history_maximized):
                if gen.size > 0:
                    # Normalize generation data
                    gen_normalized = (gen - obj_min) / obj_range
                    for obj_idx in range(self.n_obj):
                        all_obj_points[obj_idx].extend(
                            [(gen_idx, val) for val in gen_normalized[:, obj_idx]]
                        )
                        best_obj_per_gen[obj_idx].append(
                            np.max(gen_normalized[:, obj_idx])
                        )
                else:
                    for obj_idx in range(self.n_obj):
                        best_obj_per_gen[obj_idx].append(np.nan)

            for obj_idx in range(self.n_obj):
                if all_obj_points[obj_idx]:
                    x_vals, y_vals = zip(*all_obj_points[obj_idx])
                    ax.scatter(
                        x_vals,
                        y_vals,
                        facecolor=colors[obj_idx % len(colors)],
                        marker=".",
                        edgecolor="none",
                        alpha=alpha_all,
                        label=None,
                        s=10,
                        zorder=-11,
                    )
                label = (
                    self.problem.criteria_names[obj_idx]
                    if hasattr(self.problem, "criteria_names")
                    else f"Objective {obj_idx}"
                )
                ax.plot(
                    range(num_gens),
                    best_obj_per_gen[obj_idx],
                    lw=lw_best,
                    color=colors[obj_idx % len(colors)],
                    label=f"{label} best",
                )
            num_nd_sols = [
                gen.shape[0] if gen.size > 0 else 0 for gen in fitness_history_maximized
            ]
            ax.ax2 = ax.twinx()
            ax.ax2.plot(
                num_nd_sols,
                color="green",
                ls="--",
                label="Num. ND Solutions",
                alpha=0.7,
            )
            ax.ax2.set_ylabel("Num. ND Solutions", color="green")
            ax.ax2.tick_params(axis="y", labelcolor="green")
            plot_title += " (All ND Solutions, Best Highlighted)"
            y_label += " (Normalized & Maximized)"
        else:
            if (
                fitness_history_maximized.ndim != 2
                or fitness_history_maximized.shape[1] == 0
            ):
                plt.close(fig)
                return
            best_overall = np.max(fitness_history_maximized, axis=1)
            for i in range(fitness_history_maximized.shape[1]):
                ax.plot(
                    fitness_history_maximized[:, i],
                    color="lightblue",
                    alpha=0.8,
                    label="Island Champions" if i == 0 else None,
                )
            ax.plot(best_overall, lw=2, label="Best Overall Champion", color="tab:blue")
            y_label += " (Maximized)"
        if (
            hasattr(self, "migrate_every")
            and self.migrate_every
            and num_generations > 0
        ):
            mig_gens = np.arange(
                self.migrate_every, num_generations + 1, self.migrate_every
            )
            mig_indices = mig_gens[mig_gens <= num_generations]
            if mig_indices.size > 0:
                ymin, ymax = ax.get_ylim()
                ax.scatter(
                    mig_indices,
                    [ymin] * len(mig_indices),
                    marker=7,
                    linewidth=0,
                    color="red",
                    s=60,
                    zorder=5,
                    label="Migration Event",
                )
                ax.set_ylim(ymin, ymax)
        handles, labels = ax.get_legend_handles_labels()
        if ax.ax2:
            h2, l2 = ax.ax2.get_legend_handles_labels()
            handles.extend(h2)
            labels.extend(l2)
        unique_legend = dict(zip(labels, handles))
        ax.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc="lower right",
            frameon=True,
            facecolor="white",
        )
        ax.set_xlim(0, num_generations - 1 if num_generations > 1 else 1)
        ax.grid(True, ls="--", alpha=0.7)
        ax.set_xlabel("Generations")
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    def get_file_info(self, filename):
        """
        Returns a nicely formatted string with the number of islands, population size, and number of objectives.
        """
        if os.path.exists(f"{filename}.npz"):
            try:
                # Load all data from single file
                data = np.load(f"{filename}.npz", allow_pickle=True)
                
                # Extract metadata
                metadata = data["metadata"].item()
                
                n_islands = metadata.get("n_islands", "Unknown")
                population_size = metadata.get("population_size", "Unknown") 
                n_obj = metadata.get("n_obj", "Unknown")
                completed_generations = metadata.get("completed_generations", "Unknown")
                version = metadata.get("version", "1.0")
                algo_names = metadata.get("algo_names", [])
                
                info_str = (
                    f"Save file '{filename}.npz' (format v{version}) contains:\n"
                    f"- Number of islands: {n_islands}\n"
                    f"- Population size: {population_size}\n"
                    f"- Number of objectives: {n_obj}\n"
                    f"- Completed generations: {completed_generations}"
                )
                
                if algo_names:
                    info_str += f"\n- Algorithms: {', '.join(algo_names)}"
                    
                print(info_str)
                return info_str
                
            except Exception as e:
                return f"Error reading save file: {e}"
        else:
            return "No save file found."


class ParallelTqdm(Parallel):
    """
    Joblib Parallel executor with integrated tqdm progress bar.

    Extends joblib.Parallel to provide real-time progress tracking during
    parallel execution. Useful for monitoring long-running parallel tasks
    such as population generation or fitness evaluation.
    """

    def __init__(
        self,
        n_jobs: int = 0,
        *,
        total_tasks: int = None,
        show_joblib_header: bool = False,
        parallel_kwargs: dict = None,
        tqdm_kwargs: dict = None,
    ):
        if parallel_kwargs is None:
            parallel_kwargs = {}
        parallel_kwargs["verbose"] = 1 if show_joblib_header else 0
        parallel_kwargs["n_jobs"] = n_jobs if int(n_jobs) > 0 else n_jobs - 1
        super().__init__(**parallel_kwargs)
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        if "iterable" in tqdm_kwargs:
            raise TypeError(
                "keyword argument 'iterable' is not supported in 'tqdm_kwargs'."
            )
        if "total" in tqdm_kwargs:
            total_from_tqdm = tqdm_kwargs.pop("total")
            if total_tasks is None:
                total_tasks = total_from_tqdm
            elif total_tasks != total_from_tqdm:
                raise ValueError(
                    "keyword argument 'total' for tqdm_kwargs is specified and different from 'total_tasks'"
                )
        self.tqdm_kwargs = dict(unit="tasks") | tqdm_kwargs
        self.total_tasks = total_tasks
        self.progress_bar = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            return super().__call__(iterable)
        finally:
            if self.progress_bar is not None:
                self.progress_bar.close()

    def dispatch_one_batch(self, iterator):
        if self.progress_bar is None:
            self.progress_bar = tqdm_module.tqdm(
                total=self.total_tasks,
                **self.tqdm_kwargs,
            )
        return super().dispatch_one_batch(iterator)

    def print_progress(self):
        if self.total_tasks is None and self._original_iterator is None:
            self.progress_bar.total = self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.refresh()
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


class InitialPopulation:
    """
    Intelligent initial population generator for DAS layout optimization.

    This class can generate diverse, feasible initial populations for evolutionary
    optimization by starting from user-provided proposal points and applying
    controlled perturbations. It ensures all generated layouts satisfy constraints
    while trying to maintaining diversity for effective optimization.

    The generator supports multiple proposal point sets with weighted sampling,
    spatial correlations in perturbations, and parallel generation for efficiency.
    Failed generations trigger emergency procedures to ensure population completeness.

    Arguments:
        problem: DASOptimizationProblem instance defining constraints and feasibility
            criteria for generated layouts.

        proposal_points: Base layouts for population generation. Can be:

            - numpy.ndarray: Single set of points (N×2) or multiple sets (M×N×2)
            - list: List of point arrays for different layout types

            These serve as seeds that are perturbed and interpolated to create
            the initial population. More diverse proposals lead to better exploration.

        perturb_proposal: Standard deviation for Gaussian perturbation of proposal
            points before spline interpolation. Higher values increase diversity
            but may violate constraints. Defaults to 0.0.

        perturb_knots: Standard deviation for Gaussian perturbation of final
            knot points after interpolation. Applied after constraint satisfaction.
            Defaults to 0.0.

        random_t: Whether to use random parameterization when sampling along
            proposal splines. If False, uses evenly spaced parameter values.
            Random sampling increases layout diversity. Defaults to True.

        proposal_weights: Relative sampling weights for multiple proposal sets.
            Must be non-negative and sum to positive value. If None, uses
            uniform weights. Allows biasing toward preferred layout types.
            Defaults to None.

        corr_len: Correlation length for spatially correlated knot perturbations.
            Controls smoothness of perturbations along cable path. Larger values
            create smoother deformations. If 0, perturbations are uncorrelated.
            Defaults to 0.

        corr_str: Correlation strength (0-1) for knot perturbations. 1.0 means
            fully correlated perturbations, 0.0 means independent. Controls
            balance between smooth and random deformations. Defaults to 1.0.

        min_length: Minimum acceptable cable length for generated layouts.
            Layouts shorter than this are rejected during generation. Defaults to 0.

        max_attempts: Maximum attempts to generate each valid individual.
            Higher values improve success rate but increase computation time.
            Should be large enough to handle constraint complexity.
            Defaults to 100,000.

        max_emergency_attempts: Maximum emergency attempts using randomized
            proposals when normal generation fails. Provides fallback to
            ensure population completeness. Defaults to 1,000.

        random_seed: Random seed for reproducible population generation.
            If None, uses random initialization. Defaults to None.

        n_jobs: Number of parallel workers for population generation.
            Higher values speed up generation but require more memory.
            Defaults to 1.

        show_progress: Whether to display progress bars during generation.
            Useful for monitoring long generation processes. Defaults to False.

        **kwargs: Additional arguments passed to scipy.interpolate.splprep
            for spline interpolation. Common options include smoothing parameters.

    Examples:
        Basic population generation::

            >>> pop_gen = InitialPopulation(
            ...     problem=problem,
            ...     proposal_points=proposal_layouts,
            ...     perturb_proposal=10.0,
            ...     max_attempts=50000
            ... )
            >>> layouts = pop_gen.get_layouts(200)

        Multi-proposal generation with correlation::

            >>> proposals = [straight_line, curved_path, spiral_layout]
            >>> weights = [0.5, 0.3, 0.2]
            >>> pop_gen = InitialPopulation(
            ...     problem=problem,
            ...     proposal_points=proposals,
            ...     proposal_weights=weights,
            ...     perturb_knots=5.0,
            ...     corr_len=50.0,
            ...     corr_str=0.8,
            ...     n_jobs=4,
            ...     show_progress=True
            ... )
            >>> decision_vectors = pop_gen.get_layouts(500)

    Raises:
        ValueError: If proposal points are invalid, weights don't sum to positive
            value, or not enough valid individuals can be generated.
        RuntimeError: If population generation fails after all attempts.
    """

    def __init__(
        self,
        problem,
        proposal_points,
        perturb_proposal=0.0,  # stddev for proposal points perturbation
        perturb_knots=0.0,  # stddev for knot points perturbation
        random_t=True,
        proposal_weights=None,
        corr_len=0,  # correlation length for knot perturbation
        corr_str=1.0,  # correlation strength for knot perturbation
        min_length=0,
        max_attempts=100_000,
        max_emergency_attempts=1000,
        random_seed=None,
        n_jobs=1,
        show_progress=False,
        **kwargs,
    ):
        self.problem = problem
        self.proposal_points_list = self._prepare_proposal_points(proposal_points)

        self.perturb_proposal = perturb_proposal
        self.perturb_knots = perturb_knots

        self.corr_len = corr_len
        self.corr_str = corr_str

        self.spline_kwargs = kwargs
        self.spline_kwargs.setdefault("k", 1)

        self.max_attempts = max_attempts
        self.max_emergency_attempts = max_emergency_attempts
        self.min_length = min_length

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)

        self.n_jobs = n_jobs
        self.show_progress = show_progress

        self.random_t = random_t

        # --- Handle proposal_weights ---
        if proposal_weights is not None:
            proposal_weights = np.asarray(proposal_weights, dtype=float)
            if proposal_weights.shape[0] != len(self.proposal_points_list):
                raise ValueError(
                    "proposal_weights must have same length as proposal_points"
                )
            if np.any(proposal_weights < 0):
                raise ValueError("proposal_weights must be non-negative")
            if np.sum(proposal_weights) == 0:
                raise ValueError("proposal_weights must sum to a positive value")
            self.proposal_weights = proposal_weights / np.sum(proposal_weights)
        else:
            self.proposal_weights = None

        # Validate proposal points if not using random_t
        if not self.random_t:
            self._validate_proposal_points()

    def _validate_proposal_points(self):
        """Validate that proposal points don't exceed the number of knots."""
        for i, proposal in enumerate(self.proposal_points_list):
            if len(proposal) > self.problem.N_knots:
                raise ValueError(
                    f"Proposal points set {i} has {len(proposal)} points, "
                    f"but problem only has {self.problem.N_knots} knots"
                )

    def _fill_proposal_to_knots(self, proposal):
        """Fill proposal points to match number of knots by keeping all proposal points and adding random ones."""
        if len(proposal) == self.problem.N_knots:
            return proposal

        if len(proposal) > self.problem.N_knots:
            if self.problem.N_knots < 2:
                raise ValueError("Need at least 2 knots")
            if self.problem.N_knots == 2:
                return np.array([proposal[0], proposal[-1]])
            middle_indices = self.random_state.choice(
                range(1, len(proposal) - 1),
                size=self.problem.N_knots - 2,
                replace=False,
            )
            middle_indices = np.sort(middle_indices)
            selected_indices = np.concatenate(
                [[0], middle_indices, [len(proposal) - 1]]
            )
            return proposal[selected_indices]

        if len(proposal) < 2:
            raise ValueError("Need at least 2 proposal points to create spline")

        spline, _ = make_splprep([proposal[:, 0], proposal[:, 1]], **self.spline_kwargs)
        n_missing = self.problem.N_knots - len(proposal)
        random_t_values = self.random_state.rand(n_missing)
        additional_points = np.array(spline(random_t_values)).T
        all_points = np.vstack([proposal, additional_points])
        all_t_values = np.concatenate(
            [np.linspace(0, 1, len(proposal)), random_t_values]
        )
        sort_indices = np.argsort(all_t_values)
        return all_points[sort_indices]

    def _generate_correlated_noise(self, n_points, dim, scale, knot_samples=None):
        """
        Generate correlated Gaussian noise for knot points.
        Correlation is based on distance along the cable (assume linear connections).
        If knot_samples is provided, use cumulative distance along the cable for correlation.
        """
        if self.corr_len is None or self.corr_len <= 0:
            return self.random_state.normal(loc=0, scale=scale, size=(n_points, dim))
        if knot_samples is not None and len(knot_samples) == n_points:
            # Compute cumulative distance along the cable
            dists = np.zeros(n_points)
            dists[1:] = np.cumsum(
                np.linalg.norm(knot_samples[1:] - knot_samples[:-1], axis=1)
            )
        else:
            # Fallback: use index as proxy for distance
            dists = np.arange(n_points)

        # Compute pairwise distance matrix
        dist_matrix = np.abs(dists[:, None] - dists[None, :])
        corr = np.exp(-dist_matrix / self.corr_len)
        corr = self.corr_str * corr + (1 - self.corr_str) * np.eye(n_points)
        L = np.linalg.cholesky(corr + 1e-10 * np.eye(n_points))
        z = self.random_state.normal(size=(n_points, dim))
        noise = L @ z
        noise = noise * scale
        return noise

    def _generate_samples(self, n_samples, return_layouts=True):
        n_proposals = len(self.proposal_points_list)
        weights = (
            self.proposal_weights
            if self.proposal_weights is not None
            else np.ones(n_proposals) / n_proposals
        )
        raw_counts = np.array(weights * n_samples)
        counts = np.floor(raw_counts).astype(int)
        remainder = n_samples - np.sum(counts)
        if remainder > 0:
            frac = raw_counts - counts
            for idx in np.argsort(-frac)[:remainder]:
                counts[idx] += 1
        all_seeds = []
        for count in counts:
            all_seeds.append(self.random_state.randint(0, 2**32, size=count))
        jobs = []
        for proposal_idx, seeds in enumerate(all_seeds):
            for seed in seeds:
                jobs.append((proposal_idx, seed))

        def job_wrapper(proposal_idx, seed):
            local_state = np.random.RandomState(seed)
            orig_state = self.random_state
            self.random_state = local_state
            try:
                result = self._generate_individual_for_proposal(proposal_idx)
            finally:
                self.random_state = orig_state
            if result is None:
                raise RuntimeError(
                    f"Failed to generate a valid sample for proposal point {proposal_idx} "
                    f"after {self.max_attempts} attempts. "
                    f"Consider relaxing constraints, increasing max_attempts, or checking proposal points."
                )
            return result

        if self.n_jobs == 1:
            results = []
            iterator = jobs
            if self.show_progress:
                iterator = tqdm_module.tqdm(
                    jobs, desc="Generating initial population", leave=False
                )
            for proposal_idx, seed in iterator:
                try:
                    result = job_wrapper(proposal_idx, seed)
                except RuntimeError as e:
                    raise RuntimeError(f"Initial population generation failed: {e}")
                results.append(result)
        elif self.show_progress:
            try:
                results = ParallelTqdm(
                    n_jobs=self.n_jobs,
                    total_tasks=len(jobs),
                    tqdm_kwargs={
                        "desc": "Generating initial population",
                        "leave": False,
                    },
                )(
                    [
                        delayed(wrap_non_picklable_objects(job_wrapper))(
                            proposal_idx, seed
                        )
                        for proposal_idx, seed in jobs
                    ]
                )
            except RuntimeError as e:
                raise RuntimeError(f"Initial population generation failed: {e}")
        else:
            try:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(wrap_non_picklable_objects(job_wrapper))(proposal_idx, seed)
                    for proposal_idx, seed in jobs
                )
            except RuntimeError as e:
                raise RuntimeError(f"Initial population generation failed: {e}")

        filtered_results = [r for r in results if r is not None]
        # return in random order
        self.random_state.shuffle(filtered_results)

        if return_layouts:
            return [result[0] for result in filtered_results]
        else:
            return [result[1] for result in filtered_results]

    def get_layouts(self, n_samples):
        """
        Generate a set of diverse, feasible DAS layouts.

        Arguments:
            n_samples: Number of layouts to generate.
        
        Returns:
            List of generated DASLayout objects.
        """
        
        return self._generate_samples(n_samples, return_layouts=True)

    def get_decision_vectors(self, n_samples):
        """
        Generate a set of diverse, feasible decision vectors for DAS layouts.
        Arguments:
            n_samples: Number of decision vectors to generate.
        Returns:
            List of generated decision vectors (numpy arrays).
        """
        return self._generate_samples(n_samples, return_layouts=False)

    def _prepare_proposal_points(self, proposal_points):
        if isinstance(proposal_points, np.ndarray):
            if proposal_points.ndim == 2:
                proposal_points_list = [proposal_points]
            elif proposal_points.ndim == 3:
                proposal_points_list = [
                    proposal_points[i] for i in range(proposal_points.shape[0])
                ]
            else:
                raise ValueError("Control points array must have 2 or 3 dimensions")
        elif isinstance(proposal_points, list):
            proposal_points_list = [np.array(cp) for cp in proposal_points]
        else:
            raise TypeError("Control points must be numpy array or list of arrays")

        if not proposal_points_list:
            raise ValueError("Control points list cannot be empty")

        for i, cp in enumerate(proposal_points_list):
            if cp.ndim != 2 or cp.shape[1] != 2:
                raise ValueError(
                    f"Control points set {i} must have shape (N_points, 2)"
                )

        return proposal_points_list

    def _generate_individual_for_proposal(self, proposal_idx):
        proposal = self.proposal_points_list[proposal_idx]

        def try_generate_knot_samples(proposal):
            # Perturb proposal points if requested
            if self.perturb_proposal > 0:
                perturbed_proposal = proposal + self.random_state.normal(
                    loc=0, scale=self.perturb_proposal, size=proposal.shape
                )
            else:
                perturbed_proposal = proposal

            if self.random_t:
                spline, _ = make_splprep(
                    [perturbed_proposal[:, 0], perturbed_proposal[:, 1]],
                    **self.spline_kwargs,
                )
                t = np.sort(self.random_state.rand(self.problem.N_knots))
                knot_samples = np.array(spline(t)).T
            else:
                knot_samples = self._fill_proposal_to_knots(perturbed_proposal)

            # Perturb knots if requested
            if self.perturb_knots > 0:
                if self.corr_len and self.corr_len > 0:
                    noise = self._generate_correlated_noise(
                        knot_samples.shape[0],
                        knot_samples.shape[1],
                        self.perturb_knots,
                        knot_samples=knot_samples,
                    )
                else:
                    noise = self.random_state.normal(
                        loc=0, scale=self.perturb_knots, size=knot_samples.shape
                    )
                knot_samples += noise

            if self.problem.spatial_constraints is not None:
                for i in range(len(knot_samples)):
                    point = Point(knot_samples[i])
                    if not self.problem.allowed_area.contains(point):
                        nearest_allowed = nearest_points(
                            point, self.problem.allowed_area
                        )[1]
                        knot_samples[i] = (nearest_allowed.x, nearest_allowed.y)

            knot_samples[:, 0] = np.clip(
                knot_samples[:, 0],
                self.problem.bounds[0, 0],
                self.problem.bounds[0, 1],
            )
            knot_samples[:, 1] = np.clip(
                knot_samples[:, 1],
                self.problem.bounds[1, 0],
                self.problem.bounds[1, 1],
            )

            cable_length = np.linalg.norm(knot_samples[1:] - knot_samples[:-1], axis=1)
            if np.any(cable_length > self.problem.cable_length):
                return None

            try:
                layout = self.problem._knots2layout(knot_samples)
            except ValueError:
                return None

            if np.any(layout.cable_length > self.problem.cable_length):
                return None

            if layout.cable_length < self.min_length:
                return None

            if self.problem._check_feasibility(layout):
                return (layout, self.problem._variable_knots2dv(knot_samples))

            return None

        # First, try with the given proposal up to max_attempts
        for _ in range(self.max_attempts):
            result = try_generate_knot_samples(proposal)
            if result is not None:
                return result

        logger.warning(
            f"Failed to generate a valid sample for proposal point {proposal_idx} "
            f"after {self.max_attempts} attempts. Trying with randomised proposals."
        )

        # If not successful, try max_emergency_attempts with randomised proposals
        for _ in range(self.max_emergency_attempts):
            idx = self.random_state.randint(0, len(self.proposal_points_list))
            random_proposal = self.proposal_points_list[idx]
            result = try_generate_knot_samples(random_proposal)
            if result is not None:
                return result
        logger.error(
            f"Failed to generate a valid sample for proposal point {proposal_idx} "
            f"after {self.max_emergency_attempts} emergency attempts."
        )

        return None


def _create_allowed_area_geometry(bounds, spatial_constraints):
    """
    Create allowed area geometry from bounds and spatial constraints.

    Constructs a Shapely geometry representing the valid region for DAS layout
    placement by subtracting forbidden areas from the bounding box. Applies
    small buffers to handle numerical precision issues.

    Arguments:
        bounds: Rectangular search area as [[x_min, x_max], [y_min, y_max]]
        spatial_constraints: Shapely Polygon/MultiPolygon of forbidden areas,
            or None if no constraints exist

    Returns:
        shapely.geometry: Geometry representing allowed placement area
    """
    # Create bounding box with small buffer for numerical stability
    bounding_box = box(bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 1]).buffer(
        GEOMETRY_BUFFER, cap_style=2, join_style=2
    )

    if spatial_constraints is None:
        return bounding_box

    # Subtract forbidden areas from the bounding box
    allowed = bounding_box.difference(spatial_constraints)
    simplified_allowed = allowed.simplify(GEOMETRY_BUFFER, preserve_topology=True)

    return simplified_allowed


def _validate_single_point_format(point_data):
    """
    Validate format of a single fixed point coordinate pair.

    Arguments:
        point_data: Point data as list, tuple, or array

    Returns:
        tuple: Validated (x, y) coordinates as floats

    Raises:
        TypeError: If point_data is not list/tuple/array
        ValueError: If point doesn't have exactly 2 numeric coordinates
    """
    if not isinstance(point_data, (list, tuple, np.ndarray)):
        raise TypeError(
            f"Fixed point data must be list/tuple/array, got: {type(point_data)}"
        )
    point_list = list(point_data)
    expected_len = 2

    if len(point_list) != expected_len:
        raise ValueError(
            f"Fixed point must have length {expected_len} (x, y), got: {len(point_list)}"
        )
    if not all(isinstance(val, (int, float, np.number)) for val in point_list):
        raise ValueError(f"Fixed point coordinates must be numeric, got: {point_list}")
    return tuple(float(v) for v in point_list)


def _validate_fixed_points(fixed_points):
    """
    Validate and normalize fixed points input to consistent format.

    Accepts various input formats for fixed points and converts them to
    a standardized format for internal use. Supports lists, dictionaries,
    and numpy arrays with appropriate validation.

    Arguments:
        fixed_points: Fixed points in various formats:

            - None: No fixed points
            - list/tuple: Points prepended to variable knots
            - dict: {final_index: (x, y)} for precise placement
            - numpy.ndarray: Array of points with shape (N, 2)

    Returns:
        Validated fixed points in list or dict format, or None.
        Dict results are sorted by index for consistent ordering.

    Raises:
        ValueError: If array shape is invalid or dict keys are invalid
        TypeError: If fixed_points type is unsupported
    """
    if fixed_points is None:
        return None
    if isinstance(fixed_points, np.ndarray):
        if fixed_points.ndim != 2 or fixed_points.shape[1] != 2:
            raise ValueError("Fixed points numpy array must have shape (N, 2).")
        # Convert numpy array to list of tuples for consistent handling
        fixed_points = [tuple(row) for row in fixed_points]

    if isinstance(fixed_points, dict):
        validated_dict = {}
        for key, value in fixed_points.items():
            if not isinstance(key, int) or key < 0:
                raise ValueError(
                    "Fixed point dict keys must be non-negative integers (final index), got: {key}"
                )
            validated_dict[key] = _validate_single_point_format(value)
        # Return sorted by index for predictable insertion order
        return dict(sorted(validated_dict.items()))
    elif isinstance(fixed_points, (list, tuple)):
        return [_validate_single_point_format(point) for point in fixed_points]
    else:
        raise TypeError(
            "fixed_points must be a dictionary {final_index: point_data}, list/tuple [point_data,...], or numpy array."
        )

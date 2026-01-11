import itertools
import logging
from typing import Any, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from geopandas import GeoDataFrame
from scipy.sparse import csr_matrix, eye, lil_matrix, linalg
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.plotting import plot_polygon
from shapely.vectorized import contains

from ..layout import DASLayout

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

try:
    from ttcrpy.rgrid import Grid2d
    _TTCRPY_AVAILABLE = True
except ImportError:
    _TTCRPY_AVAILABLE = False
    Grid2d = None


__all__ = ["RaySensitivity", "get_gaussian_prior"]

# --- Type Aliases ---
DesignInput = Union[DASLayout, GeoDataFrame]

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
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)


# Ensure logger has a handler at module import
_ensure_logger_handler(logger)


def get_gaussian_prior(
    smoothing_length: float, 
    nx: int, 
    ny: int, 
    grid_spacing: Tuple[float, float]
) -> csr_matrix:
    """
    Create a Gaussian prior covariance matrix for a 2D grid using vectorized operations.

    This function generates a sparse covariance matrix that encodes Gaussian spatial
    correlation between grid cells. The covariance decreases exponentially with
    distance according to the specified smoothing length scale.

    Arguments:
        smoothing_length: Characteristic length scale for Gaussian smoothing in meters.
            Controls the spatial correlation range - larger values create smoother
            spatial fields with longer-range correlations.
            
        nx: Number of grid cells in the x-direction (horizontal).
        
        ny: Number of grid cells in the y-direction (vertical).
        
        grid_spacing: Tuple containing grid spacing in (x, y) directions in meters.
            Format: ``(dx, dy)`` where dx and dy are the cell dimensions.

    Returns:
        Sparse CSC (Compressed Sparse Column) matrix of shape (N, N) where N = nx * ny.
        Matrix element (i, j) represents the covariance between grid cells i and j.
        Diagonal elements are normalized to 1.0, off-diagonal elements decay
        exponentially with spatial separation.

    Examples:
        Create prior for a 50x50 grid with 100m spacing::
        
            >>> grid_spacing = (100.0, 100.0)  # 100m x 100m cells
            >>> smoothing_length = 500.0       # 500m correlation length
            >>> prior_cov = get_gaussian_prior(smoothing_length, 50, 50, grid_spacing)
            >>> print(f"Prior matrix shape: {prior_cov.shape}")
            >>> print(f"Matrix density: {prior_cov.nnz / (50*50)**2:.3f}")
        
        Use with different x/y spacing::
        
            >>> # Rectangular cells: 200m x 100m
            >>> grid_spacing = (200.0, 100.0)
            >>> prior_cov = get_gaussian_prior(300.0, 25, 50, grid_spacing)
    """
    # Calculate kernel size (4 times smoothing length)
    s = 4 * int(np.ceil(smoothing_length))

    # Create 1D Gaussian kernels for x and y
    x_indices = np.arange(-s, s + 1)
    y_indices = np.arange(-s, s + 1)

    # Calculate Gaussian weights for x and y directions
    dist_x = np.exp(-0.5 * (x_indices**2) * grid_spacing[0] ** 2 / smoothing_length**2)
    dist_y = np.exp(-0.5 * (y_indices**2) * grid_spacing[1] ** 2 / smoothing_length**2)

    # Create 2D kernel via outer product
    K = np.outer(dist_x, dist_y)

    # Initialize sparse matrix in LIL format (efficient for incremental construction)
    N = nx * ny
    C = lil_matrix((N, N))

    # Build the covariance matrix more efficiently
    for i in range(N):
        # Convert flat index to 2D indices
        idx, idy = divmod(i, ny)

        # Calculate kernel bounds for this grid point
        x_min, x_max = max(0, idx - s), min(nx, idx + s + 1)
        y_min, y_max = max(0, idy - s), min(ny, idy + s + 1)

        # Extract relevant part of the kernel
        k_x_min, k_x_max = s - (idx - x_min), s + (x_max - idx)
        k_y_min, k_y_max = s - (idy - y_min), s + (y_max - idy)
        kernel_slice = K[k_x_min:k_x_max, k_y_min:k_y_max]

        # Normalize
        scale = kernel_slice.sum()
        if scale > 0:
            kernel_slice = kernel_slice / scale

            # Set the covariance values
            for ii in range(x_min, x_max):
                for jj in range(y_min, y_max):
                    j = ny * ii + jj
                    k_x, k_y = ii - x_min, jj - y_min
                    C[i, j] = kernel_slice[k_x, k_y]

    return C.tocsc()


class RaySensitivity:
    r"""
    Ray-based sensitivity criterion for optimal DAS layout design.

    This class evaluates DAS layouts using straight-ray path assumptions to
    compute sensitivity-based optimality criteria. The method constructs a
    linearized sensitivity matrix relating model parameters (for example,
    seismic velocity) to data observations, then computes A-optimality,
    D-optimality, or RER measures.

    Type hints and documentation use Sphinx roles for common external types
    such as :class:`~numpy.ndarray`, :class:`pandas.DataFrame`, and
    :class:`geopandas.GeoDataFrame`. Internal references (for example
    :class:`~dased.layout.DASLayout`) use fully qualified references.

    The criterion supports both source-to-receiver and inter-channel (ambient noise)
    ray configurations, with Signal-to-Noise Ratio (SNR) filtering based on:
    
    - Geometric spreading (distance decay)
    - Wave type directional sensitivity (Rayleigh vs Love waves)  
    - Channel-specific signal strength and coupling factors
    - User-defined SNR thresholds

    Arguments:
        x_range: Spatial extent in x-direction as (min, max) tuple in meters.
            Defines the horizontal bounds of the model parametrization grid.
            
        y_range: Spatial extent in y-direction as (min, max) tuple in meters.
            Defines the vertical bounds of the model parametrization grid.
            
        data_type: Wave type for directional sensitivity calculations. Supported types:
        
            - ``"rayleigh"``: Rayleigh waves with :math:`\cos^2(\theta)` directional scaling
            - ``"love"``: Love waves with :math:`|\sin(2\theta)|` directional scaling
            - ``"3C"``: Three-component (no directional scaling)
            
            Where :math:`\theta` is the angle between ray direction and channel orientation.
            
        n_points: Grid resolution for model parametrization. Can be:
        
            - ``int``: Creates square grid with n_points :math:`\times` n_points cells
            - ``tuple``: Specifies (nx, ny) for rectangular grids
            
            Higher resolution increases computational cost but improves spatial detail.
            Defaults to 70.
            
        reference_distance: Reference distance in meters for SNR normalization.
            Distance at which SNR = 1.0 for signal strength calculation.
            Used in geometric spreading: SNR :math:`\propto` (distance/reference_distance):math:`^{-1}`.
            Defaults to 1000.0 meters.
            
        snr_threshold: Minimum SNR threshold for ray inclusion. Defaults to 1.5.
            
        sources: Optional[:class:`~numpy.ndarray`]
            External source locations for source-to-receiver ray paths. If
            provided, should be a 2D array with shape ``(M, >=2)`` containing
            ``(x, y)`` coordinates, optionally ``(x, y, z)``. If ``None``,
            uses inter-channel rays for ambient noise analysis.

            Format: ``[[x1, y1], [x2, y2], ..., [xM, yM]]``. Units should match
            ``x_range`` and ``y_range`` (typically meters). Defaults to ``None``.
            
        roi: Region of Interest for constraining the model space. Options:
        
            - ``None``: Use entire grid (default)
            - ``list/tuple``: Bounding box as [xmin, xmax, ymin, ymax]
            - :class:`shapely.geometry.Polygon`/:class:`shapely.geometry.MultiPolygon`: Shapely geometry objects
            
            Only grid cells within the ROI contribute to the optimization criterion.
            Useful for focusing on area(s) of interest. Defaults to None.
            
        criterion: Optimality criterion for layout evaluation. Supported options:
        
            - ``"A"``: A-optimality (trace of covariance matrix)
            - ``"D"``: D-optimality (determinant of covariance matrix)  
            - ``"RER"``: RER Criterion
            
            A-optimality minimizes average parameter uncertainty, D-optimality
            minimizes uncertainty volume, RER optimises number of eigenvalues above threshold.
            Defaults to "D".
            
        criterion_kwargs: Additional parameters for criterion calculation.
            Dictionary passed to the optimality computation method. Specific
            options depend on the chosen criterion.
            Defaults to None (empty dictionary).

    Raises:
        ImportError: If `ttcrpy` package is not installed (required for ray tracing).
        ValueError: If data_type or criterion are not in supported lists, or if
            sources array has invalid dimensions.

    Examples:
        Basic D-optimality criterion for Rayleigh waves::
        
            >>> criterion = RaySensitivity(
            ...     x_range=(0, 5000),     # 5 km extent
            ...     y_range=(0, 3000),     # 3 km extent  
            ...     data_type="rayleigh",
            ...     n_points=80,           # 80x80 grid
            ...     snr_threshold=1.5
            ... )
            
        Source-receiver configuration with A-optimality::
        
            >>> import numpy as np
            >>> sources = np.array([[100, 100], [900, 100], [500, 500]])  # :class:`~numpy.ndarray`
            >>> criterion = RaySensitivity(
            ...     x_range=(0, 1000),
            ...     y_range=(0, 600), 
            ...     data_type="love",
            ...     sources=sources,
            ...     criterion="A"
            ... )
            
        Using ROI to focus on specific area::
        
            >>> from shapely.geometry import Polygon
            >>> roi_poly = Polygon([(200, 200), (800, 200), (800, 400), (200, 400)])  # :class:`shapely.geometry.Polygon`
            >>> criterion = RaySensitivity(
            ...     x_range=(0, 1000),
            ...     y_range=(0, 600),
            ...     data_type="rayleigh", 
            ...     roi=roi_poly,
            ...     criterion="RER"
            ... )
    """
    #TODO: theres a lot to potentially add here 
    # prior model for non straight ray paths
    # allow to incorporate prior knowledge in model and data space to avoid regularization

    _VALID_DATA_TYPES = {"rayleigh", "love", "3c"}
    _VALID_CRITERIA = {"A", "D", "RER"}

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        data_type: str,
        n_points: Union[int, Tuple[int, int]] = 70,
        reference_distance: float = 1000.0,
        snr_threshold: float = 1.5,
        sources: Optional[np.ndarray] = None,
        roi: Optional[Any] = None,
        criterion: str = "D",
        criterion_kwargs: Optional[dict] = None,
    ):
        if not _TTCRPY_AVAILABLE:
            raise ImportError(
                "The ttcrpy package is required for ray-path calculations. The vtk package is needed as well which is not automatically installed with ttcrpy. Please install both packages to use the RaySensitivity criterion."
            )

        if data_type.lower() not in self._VALID_DATA_TYPES:
            raise ValueError(f"Invalid data_type '{data_type}'. Allowed: {self._VALID_DATA_TYPES}")
        if criterion not in self._VALID_CRITERIA:
            raise ValueError(f"Invalid criterion '{criterion}'. Allowed: {self._VALID_CRITERIA}")

        self.data_type = data_type.lower()
        self.x_range = x_range
        self.y_range = y_range
        self.n_points = n_points
        self.reference_distance = reference_distance
        self.snr_threshold = snr_threshold
        self.roi = roi
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs if criterion_kwargs is not None else {}

        # Validate and store source configuration
        if sources is not None:
            self.sources = np.asarray(sources)
            if self.sources.ndim != 2 or self.sources.shape[1] < 2:
                raise ValueError("Sources must be a 2D array with shape (M, >=2).")
        else:
            self.sources = None

        # Precompute grid and ROI mask/indices
        if isinstance(self.n_points, (int, float)):
            self.nx, self.ny = int(self.n_points), int(self.n_points)
        else:
            self.nx, self.ny = map(int, self.n_points)
        self._original_grid_shape: Optional[Tuple[int, int]] = (self.nx, self.ny)
        n_cells = self.nx * self.ny

        # Create grid coordinate arrays
        x_nodes = np.linspace(self.x_range[0], self.x_range[1], self.nx + 1)
        y_nodes = np.linspace(self.y_range[0], self.y_range[1], self.ny + 1)
        x_centers = 0.5 * (x_nodes[1:] + x_nodes[:-1])
        y_centers = 0.5 * (y_nodes[1:] + y_nodes[:-1])

        # Initialize region of interest
        if self.roi is None:
            self._roi_mask: Optional[np.ndarray] = np.ones(
                (self.nx, self.ny), dtype=bool
            )
            self._roi_indices: Optional[np.ndarray] = np.arange(n_cells)
        else:
            self._roi_mask: Optional[np.ndarray] = self._get_roi_mask(
                x_centers, y_centers
            )
            self._roi_indices: Optional[np.ndarray] = np.where(
                self._roi_mask.flatten()
            )[0]

        self.n_model_space = (
            len(self._roi_indices) if self._roi_indices is not None else n_cells
        )

        # Initialize sensitivity matrix cache
        self._sensitivity_matrix: Optional[csr_matrix] = None

        self.__name__ = f"RaySensitivity_{self.data_type}_{self.criterion}"

    def _snr_scaling(self, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculate directional SNR scaling factor for wave type sensitivity.
        
        Arguments:
            phi: Ray propagation azimuth angles in radians
            theta: Channel orientation angles in radians
            
        Returns:
            Directional scaling factors (0-1 for Rayleigh, 0-1 for Love)
        """
        angle_diff = theta - phi
        if self.data_type == "rayleigh":
            # Rayleigh waves have maximum sensitivity perpendicular to propagation
            return np.cos(angle_diff) ** 2
        elif self.data_type == "love":
            # Love waves have maximum sensitivity for horizontal motion
            # Drop factor of 0.5 since this is accounted for in reference distance
            return np.abs(np.sin(2 * angle_diff))
        elif self.data_type == "3c":
            # 3C sensitivity (isotropic)
            return np.ones_like(phi)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def _snr_scaling_pair(
        self, phi: np.ndarray, theta1: np.ndarray, theta2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate combined directional SNR scaling for inter-channel ray pairs.
        
        Arguments:
            phi: Ray propagation azimuth angles in radians
            theta1: First channel orientation angles in radians 
            theta2: Second channel orientation angles in radians
            
        Returns:
            Combined directional scaling factors for channel pairs
        """
        scale1 = self._snr_scaling(phi, theta1)
        scale2 = self._snr_scaling(phi, theta2)
        return scale1 * scale2

    def _get_channel_properties(
        self, gdf: GeoDataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract channel positions, orientations, and SNR multipliers from GeoDataFrame.
        
        Arguments:
            gdf: GeoDataFrame with channel information including Point geometries
                and direction vectors (u_x, u_y)
        
        Returns:
            Tuple containing:
            - positions: Array of (x, y) coordinates, shape (N, 2)
            - directions: Array of (u_x, u_y) unit vectors, shape (N, 2) 
            - multipliers: Array of combined SNR multipliers, shape (N,)
            
        Raises:
            ValueError: If geometry is invalid or required columns are missing
        """
        if not gdf.geometry.is_valid.all() or not all(
            isinstance(geom, Point) for geom in gdf.geometry
        ):
            raise ValueError("Geometry column must contain only valid Point objects.")
        if len(gdf) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        positions = np.array([[geom.x, geom.y] for geom in gdf.geometry])

        try:
            directions = gdf[["u_x", "u_y"]].to_numpy(dtype=float)
        except (KeyError, ValueError) as e:
            raise ValueError(
                "Columns 'u_x' and 'u_y' must exist and be numeric."
            ) from e

        signal_strength = (
            pd.to_numeric(gdf.get("signal_strength", 1.0), errors="coerce")
            .fillna(1.0)
            .values
        )
        signal_strength[signal_strength < 0] = 0.0

        k_col_specific = f"K_{self.data_type}"
        k_col_generic = "K"
        if k_col_specific in gdf.columns:
            k_factors = (
                pd.to_numeric(gdf[k_col_specific], errors="coerce").fillna(1.0).values
            )
        elif k_col_generic in gdf.columns:
            k_factors = (
                pd.to_numeric(gdf[k_col_generic], errors="coerce").fillna(1.0).values
            )
        else:
            k_factors = np.ones(len(gdf))
        k_factors[k_factors < 0] = 0.0

        multipliers = signal_strength * k_factors
        return positions, directions, multipliers

    def _filter_rays_by_snr(
        self,
        p1s: np.ndarray,
        p2s: np.ndarray,
        d1s: Optional[np.ndarray] = None,
        d2s: Optional[np.ndarray] = None,
        m1s: Optional[np.ndarray] = None,
        m2s: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters rays (source-channel or inter-channel) based on SNR threshold."""
        ray_vecs = p2s - p1s
        dists = np.linalg.norm(ray_vecs, axis=1)

        mask = dists > 1e-9
        if not np.any(mask):
            return np.empty((0, 2)), np.empty((0, 2))

        # Apply mask
        p1s_filt, p2s_filt = p1s[mask], p2s[mask]
        ray_vecs_filt = ray_vecs[mask]
        dists_filt = dists[mask]

        # Calculate SNR components
        dist_snr = (dists_filt / self.reference_distance) ** (-1)
        phi = np.arctan2(ray_vecs_filt[:, 1], ray_vecs_filt[:, 0])  # Ray azimuth

        if d1s is not None and d2s is not None and m1s is not None and m2s is not None:
            # Inter-channel mode
            d1s_filt, d2s_filt = d1s[mask], d2s[mask]
            m1s_filt, m2s_filt = m1s[mask], m2s[mask]
            theta1 = np.arctan2(d1s_filt[:, 1], d1s_filt[:, 0])
            theta2 = np.arctan2(d2s_filt[:, 1], d2s_filt[:, 0])
            dir_scaling = self._snr_scaling_pair(phi, theta1, theta2)
            multiplier_scaling = m1s_filt * m2s_filt
        elif d1s is not None and m1s is not None:
            # Source-to-channel mode (d1s=channel_dir, m1s=channel_mult)
            d1s_filt = d1s[mask]
            m1s_filt = m1s[mask]
            theta = np.arctan2(d1s_filt[:, 1], d1s_filt[:, 0])
            dir_scaling = self._snr_scaling(phi, theta)
            multiplier_scaling = m1s_filt
        else:
            raise ValueError(
                "Invalid combination of direction/multiplier inputs for SNR filtering."
            )

        snr = dist_snr * dir_scaling * multiplier_scaling
        valid_mask = snr >= self.snr_threshold

        return p1s_filt[valid_mask], p2s_filt[valid_mask]

    def _get_roi_mask(self, x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
        """Gets a boolean mask indicating grid cells within the ROI."""
        self.nx, self.ny = len(x_centers), len(y_centers)
        if self.roi is None:
            return np.ones((self.nx, self.ny), dtype=bool)

        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

        if isinstance(self.roi, (tuple, list)) and len(self.roi) == 4:
            x_min, x_max, y_min, y_max = self.roi
            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        elif isinstance(self.roi, (Polygon, MultiPolygon)):
            points = np.column_stack([X.flatten(), Y.flatten()])
            mask = contains(self.roi, points[:, 0], points[:, 1]).reshape(X.shape)
        else:
            logger.warning(f"Unknown ROI type {type(self.roi)}. Including all cells.")
            mask = np.ones_like(X, dtype=bool)
        return mask

    def _apply_roi_to_sensitivity(
        self, L: csr_matrix, x_nodes: np.ndarray, y_nodes: np.ndarray
    ) -> csr_matrix:
        """Applies the ROI mask to the sensitivity matrix."""
        self.nx, self.ny = len(x_nodes) - 1, len(y_nodes) - 1
        n_cells = self.nx * self.ny

        if L.shape[1] != n_cells:
            logger.error(
                f"Sensitivity matrix columns ({L.shape[1]}) != grid cells ({n_cells}). Skipping ROI masking."
            )
            return L

        return L[:, self._roi_indices]

    def _get_filtered_rays(self, design: DesignInput) -> Tuple[np.ndarray, np.ndarray]:
        """Generates and filters rays based on SNR."""
        gdf = design.get_gdf() if isinstance(design, DASLayout) else design
        
        # remove channels outside the x_range and y_range
        gdf = gdf.cx[self.x_range[0] : self.x_range[1], self.y_range[0] : self.y_range[1]]
        
        positions, directions, multipliers = self._get_channel_properties(gdf)
        n_channels = len(positions)

        if self.sources is not None:
            # Source-to-channel mode
            source_coords = self.sources[:, :2]  # Use only x, y
            # Repeat sources and channels for pairwise comparison
            all_sources = np.repeat(source_coords, n_channels, axis=0)
            all_channels_pos = np.tile(positions, (len(source_coords), 1))
            all_channels_dir = np.tile(directions, (len(source_coords), 1))
            all_channels_mult = np.tile(multipliers, len(source_coords))

            sources_filt, receivers_filt = self._filter_rays_by_snr(
                all_sources,
                all_channels_pos,
                d1s=all_channels_dir,
                m1s=all_channels_mult,
            )
        else:
            # Inter-channel mode
            if n_channels < 2:
                return np.empty((0, 2)), np.empty((0, 2))

            indices = list(itertools.combinations(range(n_channels), 2))
            idx1 = [idx[0] for idx in indices]
            idx2 = [idx[1] for idx in indices]

            p1s, p2s = positions[idx1], positions[idx2]
            d1s, d2s = directions[idx1], directions[idx2]
            m1s, m2s = multipliers[idx1], multipliers[idx2]

            sources_filt, receivers_filt = self._filter_rays_by_snr(
                p1s, p2s, d1s=d1s, d2s=d2s, m1s=m1s, m2s=m2s
            )

        return sources_filt, receivers_filt

    def _build_sensitivity_matrix(self, design: DesignInput) -> csr_matrix:
        """Builds the sensitivity matrix L, applying SNR filtering and ROI masking."""
        sources_filt, receivers_filt = self._get_filtered_rays(design)

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        self.nx, self.ny = (
            (int(self.n_points), int(self.n_points))
            if isinstance(self.n_points, (int, float))
            else map(int, self.n_points)
        )
        n_cells = self.nx * self.ny

        # Initialize ROI properties even if no rays pass filter
        x_nodes = np.linspace(x_min, x_max, self.nx + 1)
        y_nodes = np.linspace(y_min, y_max, self.ny + 1)
        self._original_grid_shape = (self.nx, self.ny)
        if self.roi is None:
            self._roi_mask = np.ones((self.nx, self.ny), dtype=bool)
            self._roi_indices = np.arange(n_cells)
        else:
            # Calculate mask and indices early
            x_centers = 0.5 * (x_nodes[1:] + x_nodes[:-1])
            y_centers = 0.5 * (y_nodes[1:] + y_nodes[:-1])
            self._roi_mask = self._get_roi_mask(x_centers, y_centers)
            self._roi_indices = np.where(self._roi_mask.flatten())[0]

        if len(sources_filt) == 0:
            logger.warning(
                f"No rays meet the SNR threshold for data_type='{self.data_type}'."
            )
            self._sensitivity_matrix = csr_matrix(
                (
                    0,
                    len(self._roi_indices)
                    if self._roi_indices is not None
                    else n_cells,
                )
            )
            return self._sensitivity_matrix

        try:
            L_full = Grid2d.data_kernel_straight_rays(
                sources_filt, receivers_filt, x_nodes, y_nodes
            )
            L_masked = self._apply_roi_to_sensitivity(L_full, x_nodes, y_nodes)
            self._sensitivity_matrix = L_masked
            return L_masked
        except Exception as e:
            logger.info(f"Error building sensitivity matrix: {e}. Return empty matrix.")
            self._sensitivity_matrix = csr_matrix(
                (
                    0,
                    len(self._roi_indices)
                    if self._roi_indices is not None
                    else n_cells,
                )
            )
            return self._sensitivity_matrix

    def _power_method(self, A: Union[np.ndarray, csr_matrix], n: int = 20) -> float:
        """Estimates the largest eigenvalue using the power method."""
        if A.shape[0] == 0:
            return 0.0
        x = np.random.rand(A.shape[0]).astype(np.float64)
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-12:
            x = np.ones(A.shape[0], dtype=np.float64) / np.sqrt(A.shape[0])
        else:
            x /= norm_x

        for _ in range(n):
            Ax = A.dot(x)
            norm_Ax = np.linalg.norm(Ax)
            if norm_Ax < 1e-12:  # Matrix is likely zero or maps x to zero
                return 0.0
            x = Ax / norm_Ax

        # Rayleigh quotient: x^T A x / x^T x (where x^T x = 1)
        eig_max = np.dot(x, A.dot(x))
        return float(eig_max)  # Return standard float

    def _abc_optimality(self, L: csr_matrix) -> float:
        """Calculates the A-, D-, or RER-optimality criterion using torch."""

        normalize = self.criterion_kwargs.get("normalize", True)
        threshold = self.criterion_kwargs.get("threshold", 1e-10)
        penalty   = self.criterion_kwargs.get("penalty", None)
        default_penalty = 0.0

        # No data → penalty or zero
        if L is None or L.shape[0] == 0 or L.shape[1] == 0:
            return default_penalty
        # Fisher Information Matrix
        F = L.T @ L
        # Convert to dense torch tensor (double precision)
        A_np = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
        A = torch.from_numpy(A_np).double()

        if self.criterion == "A":
            # A-optimality: trace(F)
            tr = torch.trace(A)
            if normalize and tr > 0:
                # max eigenvalue via torch
                eigs = torch.linalg.eigvalsh(A)
                max_eig = eigs.max()
                return float((tr / max_eig).item()) if max_eig > 0 else 0.0
            return float(tr.item())

        elif self.criterion == "D":
            # D-optimality: penalized log-det
            eigs = torch.linalg.eigvalsh(A)
            n = eigs.numel()
            if n <= 1:
                return default_penalty

            if penalty is None:
                pen = torch.log(torch.tensor(threshold, dtype=torch.double))
            else:
                pen = torch.tensor(penalty, dtype=torch.double)
            
            valid = eigs >= threshold
            num_bad = n - int(valid.sum().item())
            good = eigs[valid]

            if good.numel() == 0:
                return float(num_bad * pen.item())

            if normalize:
                max_eig = good.max()
                if max_eig > 0:
                    good = good / max_eig
                else:
                    return float(num_bad * pen.item() + good.numel() * pen.item())

                log_det = torch.sum(torch.log(good)) + num_bad * pen

                denom = pen * n
                
                return float((1.0 - torch.abs(log_det / denom)).item())

            else:
                log_det = torch.sum(torch.log(good)) + num_bad * pen
        
                return float(log_det.item())

        elif self.criterion == "RER":
            # RER-optimality: normalized rank
            eigs = torch.linalg.eigvalsh(A)
            max_eig = eigs.max()
            if max_eig > 0:
                eigs = eigs / max_eig
            count = int((eigs >= threshold).sum().item())
            return float(count) / float(self.n_model_space or 1)

        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

    def __call__(self, design: DesignInput) -> float:
        """
        Calculate the optimality criterion value for a given DAS layout design.

        This method evaluates the specified design by constructing a sensitivity 
        matrix based on ray paths and computing the selected optimality criterion
        (A-optimality, D-optimality, or RER).

        Arguments:
            design: DAS layout design to evaluate. Can be either:
            
                - :class:`~dased.layout.DASLayout`: Layout object with channel positions and properties
                - ``GeoDataFrame``: Direct GeoDataFrame representation with Point geometries
                
                Must contain columns 'u_x' and 'u_y' for channel orientations.
                Optional columns include 'signal_strength', 'K', and data-type specific
                'K_rayleigh' or 'K_love' for SNR calculations.

        Returns:
            Optimality criterion value as a float. Interpretation depends on criterion:
            
            - **A-optimality**: Higher values indicate better designs (minimized average uncertainty)
            - **D-optimality**: Higher values indicate better designs (maximized information content)
            - **RER**: Higher values indicate better RER Criterion
            
            Returns 0.0 if no valid rays meet the SNR threshold.

        Raises:
            TypeError: If design is not a DASLayout or GeoDataFrame
            ValueError: If design lacks required geometry or orientation columns

        Examples:
            Evaluate a linear array for ambient noise tomography::
            
                >>> import numpy as np
                >>> from dased.layout import DASLayout
                >>> 
                >>> # Create simple linear layout
                >>> knots = np.array([[0, 0], [2000, 0]])  # :class:`~numpy.ndarray`
                >>> layout = DASLayout(knots, spacing=50.0)
                >>> 
                >>> # Evaluate with A-optimality
                >>> score = criterion(layout)  # assuming criterion is configured
                >>> print(f"A-optimality score: {score:.3f}")
                
            Compare different layout configurations::
            
                >>> # L-shaped layout
                >>> knots_L = np.array([[0, 0], [1000, 0], [1000, 1000]])  # :class:`~numpy.ndarray`
                >>> layout_L = DASLayout(knots_L, spacing=50.0)
                >>> 
                >>> # Circular layout  
                >>> theta = np.linspace(0, 2*np.pi, 21)[:-1]  # 20 points
                >>> knots_circle = 500 * np.column_stack([np.cos(theta), np.sin(theta)])  # :class:`~numpy.ndarray`
                >>> layout_circle = DASLayout(knots_circle, spacing=50.0)
                >>> 
                >>> score_L = criterion(layout_L)
                >>> score_circle = criterion(layout_circle)
                >>> print(f"L-shape: {score_L:.3f}, Circle: {score_circle:.3f}")
        """
        #TODO: linear layout is a bad example if no sources are provided
        #TODO: test if code exaples run
        
        L = self._build_sensitivity_matrix(design)
        return self._abc_optimality(L)

    def _calculate_sensitivity_grid(self, L: csr_matrix) -> np.ndarray:
        """Calculates the sensitivity values per grid cell for plotting."""
        if self._original_grid_shape is None or self._roi_indices is None:
            logger.warning(
                "Grid properties not initialized. Cannot calculate sensitivity grid."
            )
            return np.zeros((1, 1))  # Return minimal grid

        grid_size_x, grid_size_y = self._original_grid_shape
        sensitivity_grid = np.zeros((grid_size_x, grid_size_y))

        if L is not None and L.shape[0] > 0 and L.shape[1] > 0:
            try:
                # Summing sparse matrix columns; prefer todense if feasible
                if L.shape[1] < 10000:  # Heuristic
                    roi_sensitivities = np.array(L.todense().sum(axis=0)).flatten()
                else:
                    roi_sensitivities = np.array(L.sum(axis=0)).flatten()

                if len(roi_sensitivities) == len(self._roi_indices):
                    flat_grid = np.zeros(grid_size_x * grid_size_y)
                    flat_grid[self._roi_indices] = roi_sensitivities
                    sensitivity_grid = flat_grid.reshape((grid_size_x, grid_size_y))
                else:
                    logger.warning(
                        "Mismatch between sensitivity values and ROI indices."
                    )
            except MemoryError:
                logger.warning(
                    "MemoryError calculating sensitivity grid. Using sparse sum."
                )
                roi_sensitivities = np.array(L.sum(axis=0)).flatten()
                if len(roi_sensitivities) == len(self._roi_indices):
                    flat_grid = np.zeros(grid_size_x * grid_size_y)
                    flat_grid[self._roi_indices] = roi_sensitivities
                    sensitivity_grid = flat_grid.reshape((grid_size_x, grid_size_y))
                else:
                    logger.warning(
                        "Mismatch between sensitivity values and ROI indices."
                    )
            except Exception as e:
                logger.error(f"Error calculating sensitivity grid: {e}")

        return sensitivity_grid

    def _plot_roi_boundary(self, ax: matplotlib.axes.Axes) -> None:
        """Plots the ROI boundary on the axes."""
        if self.roi is None:
            return

        roi_label = "ROI Boundary"
        plot_kwargs = dict(edgecolor="grey", linestyle="--", linewidth=1.5, zorder=4)

        if isinstance(self.roi, (tuple, list)) and len(self.roi) == 4:
            x_min_roi, x_max_roi, y_min_roi, y_max_roi = self.roi
            rect = plt.Rectangle(
                (x_min_roi, y_min_roi),
                x_max_roi - x_min_roi,
                y_max_roi - y_min_roi,
                fill=False,
                label=roi_label,
                **plot_kwargs,
            )
            ax.add_patch(rect)
        elif isinstance(self.roi, (Polygon, MultiPolygon)):
            plot_polygon(
                self.roi,
                ax,
                add_points=False,
                facecolor="none",
                label=roi_label,
                **plot_kwargs,
            )
        elif hasattr(self.roi, "log_prob") and self._roi_mask is not None:
            # Plot contour of the mask for distribution-based ROI
            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            self.nx, self.ny = self._original_grid_shape
            x_coords = np.linspace(x_min, x_max, self.nx + 1)
            y_coords = np.linspace(y_min, y_max, self.ny + 1)
            x_centers = 0.5 * (x_coords[1:] + x_coords[:-1])
            y_centers = 0.5 * (y_coords[1:] + y_coords[:-1])
            # Contour needs Z(y, x) if X, Y are meshgrid(x, y)
            ax.contour(
                x_centers,
                y_centers,
                self._roi_mask.T,
                levels=[0.5],
                colors=plot_kwargs["edgecolor"],
                linestyles=plot_kwargs["linestyle"],
                linewidths=plot_kwargs["linewidth"],
                zorder=plot_kwargs["zorder"],
            )
            # Add dummy plot for legend entry
            ax.plot(
                [],
                [],
                color=plot_kwargs["edgecolor"],
                linestyle=plot_kwargs["linestyle"],
                linewidth=plot_kwargs["linewidth"],
                label=roi_label,
            )
        else:
            logger.debug("Cannot plot ROI boundary for this ROI type.")

    def _generate_stats_text(self, L: csr_matrix) -> str:
        """Generates the statistics text block."""
        optimality_value = self._abc_optimality(L)
        roi_cells = len(self._roi_indices) if self._roi_indices is not None else 0
        total_cells = (
            np.prod(self._original_grid_shape) if self._original_grid_shape else 0
        )
        roi_perc = (100 * roi_cells / total_cells) if total_cells > 0 else 0.0
        ray_count = L.shape[0] if L is not None else 0

        stats = [
            f"Data Type: {self.data_type}",
            f"Criterion: {self.criterion}-optimality",
            f"Value: {optimality_value:.4f}",
            f"SNR Threshold: {self.snr_threshold}",
            f"Ray Count: {ray_count}",
            f"ROI Coverage: {roi_cells}/{total_cells} cells ({roi_perc:.1f}%)",
        ]

        if self.roi is not None:
            roi_type = "Unknown"
            if isinstance(self.roi, (tuple, list)):
                roi_type = "Bounds"
            elif isinstance(self.roi, (Polygon, MultiPolygon)):
                roi_type = "Polygon"
            elif hasattr(self.roi, "log_prob"):
                roi_type = "Distribution"
            stats.append(f"ROI Type: {roi_type}")

        params_str = ", ".join(f"{k}={v}" for k, v in self.criterion_kwargs.items())
        if params_str:
            stats.append(f"Parameters: {params_str}")

        return "\n".join(stats)

    def plot(
        self,
        design: DASLayout,
        title: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        show_stats: bool = True,
        plot_sources: bool = True,
        plot_roi_boundary: bool = True,
        log_scale: bool = False,
        **kwargs,
    ) -> Tuple[Optional[plt.Figure], Optional[matplotlib.axes.Axes]]:
        """
        Plots the sensitivity map and related info.

        Args:
            design: DASLayout object.
            title: Plot title.
            ax: Axes to plot on. If None, creates a new figure/axes.
            show_stats: Display statistics text box.
            plot_sources: Plot source markers if sources were provided.
            plot_roi_boundary: Plot the ROI boundary if ROI was provided.
            log_scale: If True, apply log1p transformation to sensitivity values.
            **kwargs: Additional arguments passed to `ax.pcolormesh`.

        Returns:
            tuple: (fig, ax) The figure and axes objects, or (None, None) on failure.
        """
        if not isinstance(design, (DASLayout, GeoDataFrame)):
            raise TypeError("Plotting requires a DASLayout object or GeoDataFrame.")

        # Ensure sensitivity matrix is built for the current design
        L = self._build_sensitivity_matrix(design)

        if self._original_grid_shape is None:
            logger.warning("Grid not initialized. Cannot plot.")
            return None, None

        sensitivity_grid = self._calculate_sensitivity_grid(L)

        # Setup Figure and Axes
        if ax is None:
            fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="white")
            ax = fig.add_subplot(111)
        else:
            fig = None

        # Plot Sensitivity Map
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("shading", "auto")
        kwargs.setdefault("vmin", 0)

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        grid_size_x, grid_size_y = self._original_grid_shape
        x_coords = np.linspace(x_min, x_max, grid_size_x + 1)
        y_coords = np.linspace(y_min, y_max, grid_size_y + 1)

        # pcolormesh expects Z[y, x] if X,Y are meshgrid(x,y, indexing='xy')
        # Or Z[x, y] if X,Y are meshgrid(x,y, indexing='ij')
        # Our sensitivity_grid is [self.nx, self.ny], so use indexing='ij'
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
        
        # Apply log transformation if requested
        plot_data = np.log1p(sensitivity_grid) if log_scale else sensitivity_grid
        mesh = ax.pcolormesh(x_grid, y_grid, plot_data, **kwargs)

        # Return early if ax was provided
        if fig is None:
            return ax

        # Plot ROI Boundary
        if plot_roi_boundary:
            self._plot_roi_boundary(ax)

        # Plot DAS Layout
        design.plot(ax=ax, color="k", plot_style="line", label="DAS Cable", zorder=3)

        # Plot Sources
        if self.sources is not None and plot_sources:
            ax.scatter(
                self.sources[:, 0],
                self.sources[:, 1],
                color="red",
                marker="*",
                s=150,
                edgecolor="k",
                linewidth=0.5,
                label="Sources",
                zorder=5,
            )

        # Add Colorbar
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.8)
        cbar.set_label("Ray Count per Cell", fontsize=12)

        # Set Title
        ray_mode = "Source-to-channel" if self.sources is not None else "Inter-channel"
        default_title = f"{ray_mode} Ray Sensitivity ({self.criterion}-optimality, {self.data_type})"
        ax.set_title(title or default_title, fontsize=14)

        # Add Statistics Text Box
        if show_stats:
            stats_text = self._generate_stats_text(L)
            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
            )

        # Finalize Plot
        ax.set_xlabel("X coordinate", fontsize=12)
        ax.set_ylabel("Y coordinate", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Create Legend (handling potential duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for h, label in zip(handles, labels):
            if label not in unique_labels and not label.startswith("_"):  # Avoid private labels
                unique_labels[label] = h
        if unique_labels:
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc="upper right",
                fontsize=9,
                facecolor="white",
                framealpha=0.8,
            )

        plt.tight_layout()

        return fig, ax

    def get_eigenvalue_spectrum(self, design: DASLayout, normalise=True) -> np.ndarray:
        """
        Computes the normalized eigenvalue spectrum of the sensitivity matrix for the given design.

        Args:
            design: DASLayout object.

        Returns:
            np.ndarray: Normalized eigenvalues in descending order.
        """

        L = self._build_sensitivity_matrix(design)
        if self._original_grid_shape is None or L.shape[0] == 0 or L.shape[1] == 0:
            logger.warning("Grid not initialized or sensitivity matrix empty.")
            return np.array([])

        F = L.T @ L

        try:
            A_np = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
            A = torch.from_numpy(A_np).double()
            eigenvalues = torch.linalg.eigvalsh(A).numpy()
        except Exception as e:
            logger.warning(f"Failed to compute eigenvalues: {e}")
            return np.array([])

        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        max_eig = np.nanmax(eigenvalues)
        if max_eig > 0 and normalise:
            eigenvalues /= max_eig  # Normalize
        return eigenvalues

    def plot_eigenvalue_spectrum(
        self,
        design: DASLayout,
        title: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs,
    ) -> Tuple[Optional[plt.Figure], Optional[matplotlib.axes.Axes]]:
        """
        Plots the eigenvalue spectrum of the sensitivity matrix.

        Args:
            design: DASLayout object.
            title: Plot title.
            ax: Axes to plot on. If None, creates a new figure/axes.
            **kwargs: Additional arguments passed to `ax.plot`.

        Returns:
            tuple: (fig, ax) The figure and axes objects, or (None, None) on failure.
        """
        if not isinstance(design, DASLayout):
            raise TypeError("Plotting requires a DASLayout object.")

        eigenvalues = self.get_eigenvalue_spectrum(design)
        if eigenvalues.size == 0:
            logger.warning("No eigenvalues to plot.")
            return None, None

        # Setup Figure and Axes
        if ax is None:
            fig = plt.figure(figsize=(8, 6), dpi=100, facecolor="white")
            ax = fig.add_subplot(111)
        else:
            fig = None

        # Plot Eigenvalue Spectrum
        ax.plot(eigenvalues, linestyle="-", **kwargs)

        # Set Title
        default_title = "Eigenvalue Spectrum"
        ax.set_title(title or default_title, fontsize=14)
        ax.set_yscale("log")

        # Finalize Plot
        ax.set_xlabel("Eigenvalue Index", fontsize=12)
        ax.set_ylabel("Normalized Eigenvalue", fontsize=12)
        ax.set_xlim(0, len(eigenvalues) - 1)
        ax.set_ylim(self.criterion_kwargs.get("threshold", 1e-10), 1.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_aspect("auto")

        # Create Legend (handling potential duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for h, label in zip(handles, labels):
            if label not in unique_labels and not label.startswith("_"):
                unique_labels[label] = h
        if unique_labels:
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                loc="upper right",
                fontsize=9,
                facecolor="white",
                framealpha=0.8,
            )
        plt.tight_layout()
        # Return early if ax was provided
        if fig is None:
            return ax
        return fig, ax

    def _compute_checkerboard_metrics(
        self,
        velocity_true: np.ndarray,
        velocity_est: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute resolution metrics for checkerboard test evaluation.

        Arguments:
            velocity_true: True velocity model (2D array).
            velocity_est: Estimated velocity model (2D array).
            roi_mask: Optional boolean mask for ROI (True = inside ROI).

        Returns:
            Dictionary with amplitude recovery (AR), correlation coefficient (CC),
            and root mean square error (RMSE).
        """
        # Flatten and apply ROI mask if provided
        if roi_mask is not None:
            mask = roi_mask.flatten()
            v_true = velocity_true.flatten()[mask]
            v_est = velocity_est.flatten()[mask]
        else:
            v_true = velocity_true.flatten()
            v_est = velocity_est.flatten()

        # Remove NaN values
        valid = ~(np.isnan(v_true) | np.isnan(v_est))
        v_true = v_true[valid]
        v_est = v_est[valid]

        if len(v_true) == 0:
            return {"AR": np.nan, "CC": np.nan, "RMSE": np.nan}

        # Amplitude Recovery: ratio of standard deviations
        std_true = np.std(v_true)
        std_est = np.std(v_est)
        ar = std_est / std_true if std_true > 0 else np.nan

        # Correlation Coefficient
        cc = np.corrcoef(v_true, v_est)[0, 1] if len(v_true) > 1 else np.nan

        # Root Mean Square Error
        rmse = np.sqrt(np.mean((v_est - v_true) ** 2))

        return {"AR": ar, "CC": cc, "RMSE": rmse}

    def inverse(
        self,
        design,
        m_true,
        m_prior,
        sigma_d,
        correlation_length,
        regularization_weight,
    ):
        """
        Performs a simple linear inversion to estimate the slowness model.
        
        Arguments:
            design: DASLayout or GeoDataFrame for the layout.
            m_true: True slowness model (2D array).
            m_prior: Prior slowness model (2D array).
            sigma_d: Data standard deviation (float).
            correlation_length: Prior correlation length (float).
            regularization_weight: Prior regularization weight (float).
            
        Returns:
            Estimated slowness model (2D array).
        """
        
        x_nodes = np.linspace(self.x_range[0], self.x_range[1], self.nx + 1)
        y_nodes = np.linspace(self.y_range[0], self.y_range[1], self.ny + 1)

        sources_filt, receivers_filt = self._get_filtered_rays(design)

        L = Grid2d.data_kernel_straight_rays(
            sources_filt, receivers_filt, x_nodes, y_nodes
        )

        d_true = L @ m_true.flatten()

        # sigma_d = 0.01  # Data standard deviation.
        np.random.seed(0)
        d_obs = d_true + sigma_d * np.random.randn(len(d_true))

        Cd_inv = 1 / sigma_d**2 * eye(len(d_obs))

        # correlation_length = 200.0  # lambda
        # regularization_weight = 2.5e-5  # sigma_M

        grid_spacing = (
            (self.x_range[1] - self.x_range[0]) / self.nx,
            (self.y_range[1] - self.y_range[0]) / self.ny,
        )

        Cm = get_gaussian_prior(correlation_length, self.nx, self.ny, grid_spacing)

        Cm *= regularization_weight**2
        Cm_inv = linalg.inv(Cm)

        H = L.T * Cd_inv * L + Cm_inv
        Cm_post = linalg.inv(H)

        m_est = (Cm_post * (L.T * Cd_inv * d_obs + Cm_inv * m_prior.flatten())).reshape(
            m_true.shape
        )

        return m_est

    def plot_checkerboard(
        self,
        design,
        background_velocity: float,  # = 2000.0,
        perturbation: float,  # = 0.05,
        vmin: float,  # = 1800.0,
        vmax: float,  # = 2200.0,
        sigma_d: float,  # = 0.001,
        correlation_length: float,  # = 200.0,
        regularization_weight: float,  # = 2.5e-5,
        ax: Optional[matplotlib.axes.Axes] = None,
        checkerboard_size: int = 4,
        mask_outside_roi: bool = True,
        show_metrics: bool = True,
        **kwargs,
    ):
        """
        Plots the true and estimated checkerboard velocity pattern using the provided design.

        Args:
            design: DASLayout or GeoDataFrame for the layout.
            ax: Optional matplotlib axes to plot on. If provided, only the estimated velocity is plotted.
            checkerboard_size: Size of each block in the checkerboard.
            background_velocity: Background velocity (m/s).
            perturbation: Relative perturbation (e.g., 0.05 for 5%).
            vmin: Minimum velocity for color scale.
            vmax: Maximum velocity for color scale.
            sigma_d: Data standard deviation.
            correlation_length: Prior correlation length.
            regularization_weight: Prior regularization weight.
            mask_outside_roi: If True, set values outside ROI to NaN.
            show_metrics: If True, display resolution metrics (AR, CC, RMSE) on the plot.
            **kwargs: Additional arguments for pcolormesh.

        Returns:
            The matplotlib axes with the plot.
        """
        nx = (
            self.nx
            if hasattr(self, "nx")
            else (self.n_points if isinstance(self.n_points, int) else self.n_points[0])
        )
        ny = (
            self.ny
            if hasattr(self, "ny")
            else (self.n_points if isinstance(self.n_points, int) else self.n_points[1])
        )
        x_range = self.x_range
        y_range = self.y_range

        # Build true checkerboard model (slowness)
        m_true = np.ones((nx, ny))
        for i in range(nx):
            for j in range(ny):
                if (i // checkerboard_size + j // checkerboard_size) % 2 == 0:
                    m_true[i, j] = 1 / (background_velocity * (1 - perturbation))
                else:
                    m_true[i, j] = 1 / (background_velocity * (1 + perturbation))

        # Use background as prior
        m_prior = np.ones((nx, ny)) / background_velocity

        # Estimate model using inverse
        m_est = self.inverse(
            design,
            m_true,
            m_prior,
            sigma_d,
            correlation_length,
            regularization_weight,
        )

        velocity_true = 1.0 / m_true
        velocity_est = 1.0 / m_est

        x_nodes = np.linspace(x_range[0], x_range[1], nx + 1)
        y_nodes = np.linspace(y_range[0], y_range[1], ny + 1)
        X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")

        kwargs.setdefault("cmap", "RdBu_r")
        kwargs.setdefault("shading", "auto")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        # Mask outside ROI if requested
        if mask_outside_roi and self.roi is not None:
            # Compute ROI mask
            x_centers = 0.5 * (x_nodes[1:] + x_nodes[:-1])
            y_centers = 0.5 * (y_nodes[1:] + y_nodes[:-1])
            roi_mask = self._get_roi_mask(x_centers, y_centers)
            # Ensure mask shape matches velocity arrays
            if roi_mask.shape == velocity_true.shape:
                velocity_true = np.where(roi_mask, velocity_true, np.nan)
                velocity_est = np.where(roi_mask, velocity_est, np.nan)

        # Compute ROI mask for metrics (before masking velocities with NaN)
        roi_mask_for_metrics = None
        if self.roi is not None:
            x_centers = 0.5 * (x_nodes[1:] + x_nodes[:-1])
            y_centers = 0.5 * (y_nodes[1:] + y_nodes[:-1])
            roi_mask_for_metrics = self._get_roi_mask(x_centers, y_centers)

        # Compute metrics using original (non-NaN-masked) velocity arrays
        metrics = self._compute_checkerboard_metrics(
            1.0 / m_true, 1.0 / m_est, roi_mask_for_metrics
        )

        if ax is not None:
            # Only plot the estimated velocity on the provided ax
            ax.pcolormesh(X, Y, velocity_est, **kwargs)
            if show_metrics:
                metrics_text = f"AR: {metrics['AR']:.2f}  CC: {metrics['CC']:.2f}  RMSE: {metrics['RMSE']:.1f} m/s"
                ax.text(
                    0.03,
                    0.03,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=7,
                    va="bottom",
                    ha="left",
                    # bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
                )
            return ax

        design.plot(ax=ax, color="k", plot_style="line", label="DAS Cable", zorder=3)
        # Otherwise, plot both true and estimated velocities side by side
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100, facecolor="white")

        # Plot true velocity
        mesh0 = axs[0].pcolormesh(X, Y, velocity_true, **kwargs)
        design.plot(
            ax=axs[0], color="k", plot_style="line", label="DAS Cable", zorder=3
        )
        axs[0].set_title("True Checkerboard Velocity")
        axs[0].set_xlabel("X coordinate")
        axs[0].set_ylabel("Y coordinate")
        axs[0].set_aspect("equal")
        plt.colorbar(mesh0, ax=axs[0], label="Velocity (m/s)", shrink=0.6)

        # Plot estimated velocity
        mesh1 = axs[1].pcolormesh(X, Y, velocity_est, **kwargs)
        design.plot(
            ax=axs[1], color="k", plot_style="line", label="DAS Cable", zorder=3
        )
        axs[1].set_title("Estimated Velocity (Inversion)")
        axs[1].set_xlabel("X coordinate")
        axs[1].set_ylabel("Y coordinate")
        axs[1].set_aspect("equal")
        plt.colorbar(mesh1, ax=axs[1], label="Velocity (m/s)", shrink=0.6)

        # Add metrics text box to estimated velocity subplot
        if show_metrics:
            metrics_text = f"AR: {metrics['AR']:.2f}\nCC: {metrics['CC']:.2f}\nRMSE: {metrics['RMSE']:.1f} m/s"
            axs[1].text(
                0.02,
                0.02,
                metrics_text,
                transform=axs[1].transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
            )

        plt.tight_layout()
        return fig, axs

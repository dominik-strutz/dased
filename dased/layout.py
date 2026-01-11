"""
This module provides classes for managing DAS layouts in both local
Cartesian and geographic coordinate systems.

.. warning:: :class:`~dased.layout.DASLayoutGeographic` is experimental and
    has not yet been tested thoroughly.
"""

import logging
import warnings
from copy import deepcopy

from typing import Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pyproj
import xarray as xr
from scipy.interpolate import splev, splprep
from scipy.spatial.distance import pdist
from shapely.geometry import LineString, Point

try:
    import contextily as ctx

    _CONTEXTILY_AVAILABLE = True
except ImportError:
    _CONTEXTILY_AVAILABLE = False
    ctx = None


__all__ = ["DASLayout", "DASLayoutGeographic"]

# --- Constants ---
DEFAULT_PENALTY = np.inf  # Penalty for invalid design variables in fitness evaluation
GEOMETRY_BUFFER = 1e-9  # Small buffer for Shapely geometric operations
EPSILON = 1e-3  # Small step for numerical gradient calculation
MIN_NORM_THRESHOLD = 1e-9  # Threshold for vector normalization

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


class DASLayout:
    r"""
    Represents a DAS cable layout defined by knot points, channel spacing and the choice of spline interpolation.
    Manages the positioning and properties of DAS channels along a cable path defined by knot points.

    Arguments:
        knots: A 2D array (N x 2) of local Cartesian coordinates defining the cable
            path knot points in meters. Minimum 2 points required.

        spacing: Spacing between channels along the cable in meters. Must be positive.

        elevation: Elevation handling for the cable path. Can be:

            - ``float``: Constant elevation applied to all channels
            - :class:`xarray.DataArray`: Gridded elevation data from which elevation is interpolated
            - ``callable``: Function f(x, y) returning elevation at given coordinates

            Defaults to 0.

        field_properties: Dictionary containing additional field properties to interpolate along cable.
            Keys represent property names, values can be:

            - Constants (float/int): Applied uniformly to all channels
            - :class:`xarray.DataArray`: Gridded data for spatial interpolation
            - ``callable``: Function f(x, y) for custom property calculation

            Defaults to an empty dictionary (no additional properties).

        signal_decay: Signal decay rate in dB/km for attenuation modeling.
            Used for modeling signal loss along the cable length.

            If None or ≤0, signal strength remains constant.

            Defaults to 0 (no decay modeling).

        wrap_around: Whether to connect the last knot to the first to form a closed loop.
            Useful for creating circular or ring-shaped DAS arrays.
            Defaults to False.

        **kwargs: Additional arguments passed to ``scipy.interpolate.splprep`` for spline fitting.
            Common options:

            - ``k``: int, smoothing order (1=linear, 2=quadratic, 3=cubic)
            - ``s``: float, smoothing factor (0=exact interpolation)
            - ``t``: array-like, parameter values for knot otherwise calculated automatically from chord length

    Examples:
        Basic straight-line layout::

            >>> import numpy as np
            >>> # Example knots as a NumPy array (2 x N coordinates in meters) :class:`~numpy.ndarray`
            >>> knots = np.array([[0, 0], [1000, 0]])  # :class:`~numpy.ndarray`
            >>> layout = DASLayout(knots, spacing=10.0)
            >>> print(layout)
    """

    def __init__(
        self,
        knots: np.ndarray = None,
        spacing: float = None,
        elevation: Union[
            float, xr.DataArray, Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = 0,
        field_properties: Optional[
            Dict[str, Union[float, int, xr.DataArray, Callable]]
        ] = {},
        signal_decay: Optional[float] = 0.0,
        wrap_around: bool = False,
        anchors: np.ndarray = None,  # backward compatibility
        **kwargs,
    ):
        # Handle backward compatibility for anchors parameter
        if anchors is not None:
            if knots is not None:
                raise ValueError(
                    "Cannot specify both 'knots' and 'anchors' parameters. Use 'knots'."
                )
            import warnings

            logger.warning(
                "The 'anchors' parameter is deprecated. Use 'knots' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            knots = anchors

        if knots is None:
            raise TypeError("Missing required argument: 'knots'")
        if spacing is None:
            raise TypeError("Missing required argument: 'spacing'")

        # Validate spacing
        if spacing <= 0:
            raise ValueError("Spacing must be a positive value.")
        self._channel_spacing = float(spacing)

        # Validate and store knots
        self.knots = np.asarray(knots)
        if self.knots.ndim != 2 or self.knots.shape[1] != 2:
            raise ValueError(
                "Knots must be a 2D array (N x 2) of local Cartesian coordinates."
            )

        # Initialize field properties
        self._field_properties = field_properties

        # Handle elevation parameter
        if "elevation" not in self._field_properties:
            self._field_properties["elevation"] = elevation 
        
        # Store signal decay parameter
        self._signal_decay = (
            float(signal_decay)
            if signal_decay is not None and signal_decay > 0
            else None
        )

        # Handle wrap-around option
        if wrap_around:
            self.knots = np.vstack([self.knots, self.knots[0]])

        # Calculate or validate parameterization
        t = kwargs.pop("t", None)  # Extract t from kwargs if present
        if t is None:
            t = self._calculate_parameterization(self.knots)

        if len(t) != self.knots.shape[0]:
            raise ValueError(
                f"Parameter t length ({len(t)}) must match knots length ({self.knots.shape[0]})"
            )

        # Initialize the layout using spline interpolation
        self._initialize_from_standard_spline(self.knots, spacing, t, **kwargs)

    def _calculate_parameterization(self, knots):
        """
        Calculate parameterization of knot points along the curve.

        Uses cumulative chord length between consecutive knot points,
        normalized to the range [0, 1]. This provides a natural parameterization
        for spline interpolation.

        Arguments:
            knots: A 2D array (N x 2) of knot point coordinates.

        Returns:
            :class:`~numpy.ndarray`: Normalized parameterization values in the range [0, 1].
            For N knot points, returns N parameter values.
        """
        n_points = knots.shape[0]

        if n_points < 2:
            return np.array([0.0] if n_points == 1 else [])

        t = np.zeros(n_points)
        distances = np.linalg.norm(np.diff(knots, axis=0), axis=1)

        # Handle case where all points are identical or nearly identical
        if np.all(distances < MIN_NORM_THRESHOLD):
            return np.linspace(0, 1, n_points)

        t[1:] = np.cumsum(distances)
        total_length = t[-1]

        if total_length > MIN_NORM_THRESHOLD:
            t = t / total_length  # Normalize to [0, 1]
        else:
            # Fallback for edge case
            t = np.linspace(0, 1, n_points)

        # Final validation to ensure strictly increasing values
        if not np.all(np.diff(t) >= 0):
            t = np.linspace(0, 1, n_points)

        return t

    # ---- Core interpolation methods ----

    def _initialize_from_standard_spline(self, knots, spacing, t, **kwargs):
        """
        Initialize channel layout using spline interpolation.

        Creates a smooth spline through the knot points and places channels
        at approximately equal intervals along the spline path.

        Arguments:
            knots: Knot point coordinates (N x 2)
            spacing: Desired spacing between channels
            t: Parameter values for knot points
            **kwargs: Additional arguments for ``scipy.interpolate.splprep``
        """
        kwargs.setdefault("k", 1)  # Default to linear interpolation
        kwargs.setdefault("s", 0)  # Default to no smoothing

        # Validate inputs before calling splprep
        if knots.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 knot points for spline interpolation, got {knots.shape[0]}"
            )

        # Check for duplicate or nearly identical points
        if knots.shape[0] > 1:
            distances = np.linalg.norm(np.diff(knots, axis=0), axis=1)
            if np.any(distances < MIN_NORM_THRESHOLD):
                # Remove duplicate points while preserving order
                unique_indices = [0]  # Always keep first point
                for i in range(1, knots.shape[0]):
                    if (
                        np.linalg.norm(knots[i] - knots[unique_indices[-1]])
                        >= MIN_NORM_THRESHOLD
                    ):
                        unique_indices.append(i)

                if len(unique_indices) < 2:
                    raise ValueError(
                        "After removing duplicates, fewer than 2 unique knot points remain"
                    )

                knots = knots[unique_indices]
                t = t[unique_indices]
                # Recalculate t to ensure it's properly normalized
                t = self._calculate_parameterization(knots)

        # Additional validation for parameter array t
        if len(t) != knots.shape[0]:
            raise ValueError(
                f"Parameter t length ({len(t)}) must match knots length ({knots.shape[0]})"
            )

        if not np.allclose(t[0], 0.0) or not np.allclose(t[-1], 1.0):
            # Re-normalize t to [0, 1] if needed
            t = (
                (t - t[0]) / (t[-1] - t[0])
                if t[-1] != t[0]
                else np.linspace(0, 1, len(t))
            )

        # Ensure t is strictly increasing
        if not np.all(np.diff(t) >= 0):
            t = np.linspace(0, 1, len(t))

        try:
            tck, _ = splprep(knots.T, u=t, **kwargs)
        except ValueError as e:
            raise ValueError(
                f"scipy.interpolate.splprep failed with inputs: knots.shape={knots.shape}, "
                f"t.shape={t.shape}, t_range=[{t[0]:.6f}, {t[-1]:.6f}], "
                f"min_knot_distance={np.min(np.linalg.norm(np.diff(knots, axis=0), axis=1)) if knots.shape[0] > 1 else 'N/A'}. "
                f"Original error: {e}"
            )

        # Define spline evaluation functions
        def spline_func(param):
            return np.array(splev(param, tck)).T

        def spline_deriv_func(param):
            return np.array(splev(param, tck, der=1)).T

        # Step along the spline to place channels
        t_start, t_end = t[0], t[-1]
        elevation_data = self.field_properties.get("elevation", 0)
        channel_parameters, channel_locations_2d, cable_length = (
            self._step_along_spline(
                spline_func, spacing, t_start, t_end, elevation_data
            )
        )

        # Calculate 2D direction vectors from spline derivatives
        channel_directions_2d = self._calculate_directions_2d(
            channel_locations_2d, channel_parameters, spline_deriv_func
        )

        # Convert to 3D coordinates including elevation and gradients
        channel_locations_3d, channel_directions_3d = self._add_elevation_and_gradient(
            channel_locations_2d,
            channel_directions_2d,
        )

        # Set instance attributes
        self._channel_locations = channel_locations_3d
        self._channel_directions = channel_directions_3d
        self._n_channels = self._channel_locations.shape[0]
        self._cable_length = cable_length
        self._knot_locations = knots

    @property
    def channel_locations(self):
        """
        Coordinates of channel locations along the cable. Channels are placed at equal arc-length intervals along the spline-interpolated cable path.

        Returns:
            :class:`~numpy.ndarray`: Array of shape (N, 3) where N is the number of channels.
            Each row contains [x, y, z] coordinates in local Cartesian system.

            - x, y: Local horizontal coordinates
            - z: Elevation above reference datum
        """
        return self._channel_locations

    @property
    def channel_directions(self):
        """
        Direction unit vectors for each channel along the cable, representing local cable orientation.

        Returns:
            :class:`~numpy.ndarray`: Array of shape (N, 3) where N is the number of channels.
            Each row contains [u_x, u_y, u_z] unit direction vector.

            - u_x, u_y: Horizontal direction components
            - u_z: Vertical direction component (elevation gradient)
        """
        return self._channel_directions

    @property
    def channel_spacing(self):
        """
        Spacing between channels along the cable.

        Set during initialization and used throughout the spline stepping process
        to determine channel placement intervals. Actual spacing may vary slightly.

        Final number of channels depends on total cable length divided by this spacing.

        Returns:
            ``float``: Target channel spacing. Always positive.
        """
        return self._channel_spacing

    @property
    def n_channels(self):
        """
        Number of channels placed along the cable.

        Depends on the total cable length and the specified channel spacing.
        Channels are placed at (near) equal arc-length intervals starting from the beginning of the cable path.

        Returns:
            ``int``: Number of channels. Always non-negative.
            Returns 0 if no valid cable path could be created.
        """
        return self._n_channels

    @property
    def cable_length(self):
        """
        Total length of the cable path.

        Calculated as the cumulative arc length along the spline-interpolated
        path between knot points. Accounts for both horizontal distance and
        elevation changes.

        Returns:
            ``float``: Total cable length. Always non-negative.
        """
        return self._cable_length

    @property
    def knot_locations(self):
        """
        Knot point coordinates used to define the cable path.

        These are the processed knot points used for spline interpolation,
        which may differ from the input knots if wrap_around was enabled
        (adds duplicate of first point at end).

        Returns:
            :class:`~numpy.ndarray`: Array of shape (M, 2) where M is the number of knot points.
            Each row contains [x, y] coordinates in local Cartesian system.
        """
        return self._knot_locations

    @property
    def anchor_locations(self):
        """
        Backward compatibility alias for knot_locations.

        .. deprecated::
            Use `knot_locations` instead.
        """
        import warnings

        logger.warning(
            "anchor_locations is deprecated. Use knot_locations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.knot_locations

    @property
    def field_properties(self):
        """
        Dictionary of field properties associated with the cable layout.

        Contains all field properties that can be interpolated at channel locations,
        including elevation data and any additional properties provided during
        initialization.

        Returns:
            ``dict``: Dictionary mapping property names (str) to their data.
            Always contains 'elevation' key. Additional keys depend on
            field_properties parameter passed to constructor.

        Values can be:
            - Constants (float/int): Applied uniformly to all channels
            - :class:`xarray.DataArray`: Gridded data for spatial interpolation
            - callable: Function f(x, y) for custom property calculation
        """
        return self._field_properties

    @property
    def signal_decay(self):
        """
        Signal decay rate for attenuation modeling in dB/km.

        Used to model signal strength reduction along the cable length.
        When provided, signal strength decreases exponentially with distance
        from the cable start point.

        Returns:
            ``float``: Signal decay rate in dB/km.
        """
        return self._signal_decay

    def _calculate_directions_2d(
        self, channel_locations_2d, channel_parameters, spline_deriv_func
    ):
        """
        Calculate normalized 2D direction vectors from spline derivatives.

        Arguments:
            channel_locations_2d: 2D channel coordinates (N x 2)
            channel_parameters: Parameter values for channels (N,)
            spline_deriv_func: Function to evaluate spline derivatives

        Returns:
            :class:`~numpy.ndarray`: Normalized 2D direction vectors (N x 2)
        """
        if channel_locations_2d.shape[0] == 0:
            return np.array([]).reshape(0, 2)

        channel_directions_2d = spline_deriv_func(channel_parameters)
        norms = np.linalg.norm(channel_directions_2d, axis=1, keepdims=True)
        valid_norms = norms > MIN_NORM_THRESHOLD

        # Normalize directions where possible, set to zero otherwise
        return np.where(valid_norms, channel_directions_2d / norms, 0.0)

    @staticmethod
    def _interpolate_xarray_map(x_coords, y_coords, map_da, warn_nan=True):
        """
        Interpolate values from xarray DataArray using linear and nearest neighbor methods.

        Performs linear interpolation first, then uses nearest neighbor interpolation
        as a fallback for any NaN values. This is useful for field data that may
        have gaps or boundaries.

        Arguments:
            x_coords: X coordinates for interpolation
            y_coords: Y coordinates for interpolation
            map_da: Source data array for interpolation
            warn_nan: Whether to warn about NaN interpolation. Defaults to True.

        Returns:
            :class:`~numpy.ndarray`: Interpolated values at the specified coordinates
        """
        if not isinstance(map_da, xr.DataArray):
            raise ValueError("Map data must be an xarray.DataArray for interpolation.")

        # Find coordinate names (flexible naming)
        x_coord_name = next(
            (name for name in ["x", "longitude", "lon"] if name in map_da.coords), None
        )
        y_coord_name = next(
            (name for name in ["y", "latitude", "lat"] if name in map_da.coords), None
        )

        if not x_coord_name or not y_coord_name:
            raise ValueError(
                f"Map data must have compatible coordinates. Found: {list(map_da.coords.keys())}"
            )

        try:
            x_da = xr.DataArray(x_coords, dims="points")
            y_da = xr.DataArray(y_coords, dims="points")

            # Primary: linear interpolation
            interp_coords = {x_coord_name: x_da, y_coord_name: y_da}
            interpolated_values = map_da.interp(
                interp_coords, method="linear", kwargs={"fill_value": np.nan}
            ).values

            # Fallback: nearest neighbor for NaN values
            nan_mask = np.isnan(interpolated_values)
            if np.any(nan_mask):
                num_nan_initial = np.sum(nan_mask)

                if warn_nan:
                    logger.info(
                        f"{num_nan_initial} points resulted in NaN during linear interpolation. "
                        f"Using nearest neighbor for these points."
                    )

                nearest_values = map_da.interp(
                    {x_coord_name: x_da[nan_mask], y_coord_name: y_da[nan_mask]},
                    method="nearest",
                    kwargs={"fill_value": None},
                ).values
                interpolated_values[nan_mask] = nearest_values

                # Check if any values are still NaN after nearest neighbor
                final_nan_mask = np.isnan(interpolated_values)
                num_nan_final = np.sum(final_nan_mask)
                if num_nan_final > 0:
                    raise ValueError(
                        f"Interpolation failed for {num_nan_final} points after nearest neighbor fallback."
                    )

            return interpolated_values

        except Exception as e:
            raise ValueError(f"Failed to interpolate map data: {e}") from e

    def _get_property_value_at_points(
        self,
        property_name,
        x_coords,
        y_coords,
    ):
        """
        Get field property values at specified coordinates.

        Supports multiple data types for field properties:
        - Constants (int/float): Same value everywhere
        - xarray DataArrays: Gridded data with spatial interpolation
        - Callable functions: Custom functions f(x, y) -> values

        Arguments:
            property_name: Name of the property to retrieve
            x_coords: X coordinates for evaluation
            y_coords: Y coordinates for evaluation

        Returns:
            :class:`~numpy.ndarray`: Property values at the specified coordinates
        """
        property_data = self.field_properties.get(property_name)
        n_points = len(x_coords)

        if isinstance(property_data, (int, float)):
            return np.full(n_points, property_data)

        elif isinstance(property_data, xr.DataArray):
            return self._interpolate_xarray_map(
                x_coords,
                y_coords,
                property_data,
            )

        elif callable(property_data):
            values = property_data(x_coords, y_coords)
            values = np.asarray(values).reshape(-1)
            if values.shape[0] != n_points:
                raise ValueError(
                    f"Callable for property '{property_name}' returned wrong shape: "
                    f"expected {n_points}, got {values.shape[0]}"
                )
            return values

    @staticmethod
    def _cable_attenuation(distance_m, decay_rate_db_per_km):
        """
        Calculate signal attenuation factor based on distance and decay rate.

        Converts signal decay in dB/km to linear amplitude factors, representing
        the relative signal strength at each position along the cable.

        Arguments:
            distance_m: Distances along cable.
            decay_rate_db_per_km: Signal decay rate in dB/km. If None or ≤0, no
                attenuation is applied.

        Returns:
            :class:`~numpy.ndarray`: Signal strength factors (1.0 = no attenuation, <1.0 = attenuated)

        Notes:
            Signal attenuation in dB: -20 * log10(amplitude_factor)
            Therefore: amplitude_factor = 10^(-attenuation_dB / 20)
        """
        distance_array = np.asarray(distance_m)

        if decay_rate_db_per_km is None or decay_rate_db_per_km <= 0:
            # No decay: signal strength factor is 1.0 everywhere
            return np.ones_like(distance_array, dtype=float)

        # Convert dB/km to dB/m and calculate total attenuation
        attenuation_rate_db_per_m = decay_rate_db_per_km / 1000.0
        attenuation_db = attenuation_rate_db_per_m * distance_array

        # Convert dB attenuation to linear amplitude factor
        amplitude_factor = 10 ** (-attenuation_db / 20.0)
        return amplitude_factor

    def _add_elevation_and_gradient(self, channel_locations_2d, channel_directions_2d):
        """
        Convert 2D locations to 3D by adding elevation and calculate 3D direction vectors.

        This method:
        1. Interpolates elevation values at channel locations
        2. Calculates elevation gradients along channel directions
        3. Constructs 3D direction vectors including vertical components

        Arguments:
            channel_locations_2d: 2D channel coordinates (N x 2)
            channel_directions_2d: 2D direction vectors (N x 2)

        Returns:
            ``tuple``: (locations_3d, directions_3d) both as (N x 3) arrays
        """
        if channel_locations_2d.shape[0] == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Get elevation values at channel locations
        z_coords = self._get_property_value_at_points(
            "elevation",
            channel_locations_2d[:, 0],
            channel_locations_2d[:, 1],
        )

        z_coords = np.nan_to_num(z_coords, nan=0.0)

        locations_3d = np.column_stack((channel_locations_2d, z_coords))

        # Calculate 3D direction vectors including elevation gradient
        if channel_directions_2d.shape[0] > 0:
            # Calculate elevation at slightly shifted positions to estimate gradient
            shifted_locations_2d = (
                channel_locations_2d + channel_directions_2d * EPSILON
            )
            z_coords_shifted = self._get_property_value_at_points(
                "elevation",
                shifted_locations_2d[:, 0],
                shifted_locations_2d[:, 1],
            )
            z_coords_shifted = np.nan_to_num(z_coords_shifted, nan=0.0)

            # Calculate elevation gradient component
            dz_component = (z_coords_shifted - z_coords) / EPSILON
            dir_vec_3d = np.column_stack((channel_directions_2d, dz_component))

            # Normalize 3D direction vectors
            norms = np.linalg.norm(dir_vec_3d, axis=1, keepdims=True)
            valid_norms = norms > MIN_NORM_THRESHOLD
            directions_3d = np.where(valid_norms, dir_vec_3d / norms, 0.0)
        else:
            directions_3d = np.zeros((locations_3d.shape[0], 3))

        return locations_3d, directions_3d

    def _step_along_spline(self, spline_func, spacing, t_start, t_end, elevation_data):
        """
        Place channel points along a spline at approximately equal arc-length intervals.

        This method accounts for both horizontal distance and elevation changes when
        calculating arc length, ensuring accurate spacing along the true 3D cable path.

        Arguments:
            spline_func: Function to evaluate spline positions
            spacing: Desired spacing between channels
            t_start: Starting parameter value
            t_end: Ending parameter value
            elevation_data: Elevation data for 3D arc length calculation

        Returns:
            ``tuple``: (parameter_values, locations_2d, total_length)

                - parameter_values: Parameter values for placed channels
                - locations_2d: 2D coordinates of channel locations
                - total_length: Total cable length
        """
        if spacing <= 0 or t_start > t_end or abs(t_start - t_end) < MIN_NORM_THRESHOLD:
            if abs(t_start - t_end) < MIN_NORM_THRESHOLD:  # Single point case
                loc_start = spline_func(np.array([t_start]))
                return np.array([t_start]), loc_start.reshape(1, -1), 0.0
            else:
                return np.array([]), np.empty((0, 2)), 0.0

        # Check if elevation varies spatially
        is_varying_elevation = not isinstance(elevation_data, (int, float, type(None)))
        elevation_func = (
            self._create_elevation_function(elevation_data)
            if is_varying_elevation
            else None
        )

        # Sample spline at high resolution for arc length approximation
        # Estimate cable length from knot distances to determine adaptive sampling
        knot_distances = np.linalg.norm(np.diff(self.knots, axis=0), axis=1)
        estimated_length = np.sum(knot_distances)
        
        # Adaptive sampling: aim for ~10-20 samples per channel spacing
        # with minimum of 100 and maximum of 50000 samples
        samples_per_spacing = 15
        num_samples = int(np.clip(
            (estimated_length / spacing) * samples_per_spacing,
            1000,
            100000
        ))
        
        t_samples = np.linspace(t_start, t_end, num_samples)
        points_2d = spline_func(t_samples)

        if points_2d.shape[0] < 2:
            return (
                t_samples,
                points_2d.reshape(1, -1) if points_2d.size > 0 else np.empty((0, 2)),
                0.0,
            )

        # Calculate 2D segment lengths
        segments_2d = np.diff(points_2d, axis=0)
        segment_lengths_sq_2d = np.sum(segments_2d**2, axis=1)

        # Add elevation component to length calculation if needed
        if is_varying_elevation and elevation_func is not None:
            z_vals = elevation_func(points_2d[:, 0], points_2d[:, 1])
            z_diffs = np.diff(z_vals)
            segment_lengths = np.sqrt(segment_lengths_sq_2d + z_diffs**2)
        else:
            segment_lengths = np.sqrt(segment_lengths_sq_2d)

        # Calculate cumulative arc lengths
        cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = cumulative_lengths[-1]

        if total_length < spacing / 2.0:
            logger.warning(
                f"Total spline length ({total_length:.2f}) is less than half the desired spacing."
            )
            return np.array([]), np.empty((0, 2)), total_length

        # Generate target distances at regular intervals
        target_distances = np.arange(spacing / 2.0, total_length, spacing)

        if len(target_distances) == 0:
            logger.warning("Placing one channel in the middle.")
            target_distances = np.array([total_length / 2.0])

        # Map target distances back to parameter values
        unique_indices = np.concatenate(
            ([True], np.diff(cumulative_lengths) > MIN_NORM_THRESHOLD)
        )
        if not np.all(unique_indices):
            logger.warning(
                "Duplicate lengths detected in spline sampling.", RuntimeWarning
            )

        t_pts = np.interp(
            target_distances,
            cumulative_lengths[unique_indices],
            t_samples[unique_indices],
        )
        locs_pts_2d = spline_func(t_pts)

        # Handle shape consistency
        if locs_pts_2d.ndim == 1 and t_pts.shape[0] > 0:
            locs_pts_2d = locs_pts_2d.reshape(t_pts.shape[0], -1)

        if locs_pts_2d.shape[0] > 0 and locs_pts_2d.shape[1] != 2:
            raise ValueError(
                f"Spline function returned unexpected shape: {locs_pts_2d.shape}"
            )

        return t_pts, locs_pts_2d, total_length

    def _create_elevation_function(self, elevation_data):
        """
        Create a function to calculate elevation at given coordinates.

        Wraps different elevation data types (xarray, callable) into a consistent
        function interface for use in arc length calculations.

        Arguments:
            elevation_data: Elevation data (:class:`xarray.DataArray` or callable)

        Returns:
            ``callable`` or :class:`None`: Function f(x, y) -> z values, or None if unsupported
        """
        if isinstance(elevation_data, xr.DataArray):

            def get_elevation(x, y):
                return self._interpolate_xarray_map(
                    np.asarray(x),
                    np.asarray(y),
                    elevation_data,
                    warn_nan=False,
                )

            return get_elevation

        elif callable(elevation_data):

            def get_elevation(x, y):
                try:
                    z = elevation_data(x, y)
                    return np.nan_to_num(np.asarray(z).reshape(-1), nan=0.0)
                except Exception as e:
                    logger.warning(f"Elevation callable failed: {e}", RuntimeWarning)
                    return np.zeros(len(np.asarray(x)))

            return get_elevation

        return None

    def get_array(self):
        """
        Return the layout data as a NumPy array.

        Returns:
            :class:`~numpy.ndarray`: Combined array of shape (N, 6) containing:
                [x, y, z, u_x, u_y, u_z] for each channel
        """
        return np.column_stack((self.channel_locations, self.channel_directions))

    def get_shapely(self):
        """
        Return the layout cable path as a Shapely LineString object.

        Returns:
            shapely.geometry.LineString: LineString representation of the cable path
                using 2D coordinates. Returns empty LineString if fewer than 2 channels.
        """
        if self.n_channels < 2:
            return LineString()
        return LineString(self.channel_locations[:, :2])

    def get_gdf(self):
        """
        Convert the layout to a GeoDataFrame with local coordinates.

        Creates a GeoDataFrame containing channel information with Point geometries
        and comprehensive attributes including channel positions and directions,
        distance along cable from start, signal strength factors, and interpolated
        field properties.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame with the following columns:

                - channel_id: Sequential channel identifiers (0, 1, 2, ...)
                - type: Receiver type (always "das")
                - u_x, u_y, u_z: Direction vector components
                - distance: Distance along cable from start
                - signal_strength: Signal strength factor (1.0 = full strength)
                - geometry: Point geometries in local coordinates
                - z: Elevation values
                - Additional columns for each field property
        Notes:
            The GeoDataFrame uses local Cartesian coordinates with CRS=None.
            For georeferenced output, use :meth:`~dased.layout.DASLayoutGeographic.get_gdf`.
        """
        if self.n_channels == 0:
            # Return empty GeoDataFrame with proper column structure
            return gpd.GeoDataFrame(
                columns=[
                    "channel_id",
                    "type",
                    "u_x",
                    "u_y",
                    "u_z",
                    "distance",
                    "signal_strength",
                    "geometry",
                    "z",
                ],
                crs=None,
            )

        # Calculate distances along cable
        channel_distances = np.zeros(self.n_channels)
        if self.n_channels > 1:
            segment_lengths = np.linalg.norm(
                np.diff(self.channel_locations, axis=0), axis=1
            )
            channel_distances[1:] = np.cumsum(segment_lengths)

        # Pre-calculate field property values
        property_values = {}
        local_x = self.channel_locations[:, 0]
        local_y = self.channel_locations[:, 1]
        for prop_name in self.field_properties:
            property_values[prop_name] = self._get_property_value_at_points(
                prop_name, local_x, local_y
            )

        # Calculate signal strength using attenuation model
        signal_strength = self._cable_attenuation(channel_distances, self.signal_decay)

        # Create Point geometries
        geometries = [Point(x, y) for x, y in zip(local_x, local_y)]

        # Build core data dictionary
        data = {
            "channel_id": np.arange(self.n_channels),
            "type": ["das"] * self.n_channels,
            "u_x": self.channel_directions[:, 0],
            "u_y": self.channel_directions[:, 1],
            "u_z": self.channel_directions[:, 2],
            "distance": channel_distances,
            "signal_strength": signal_strength,
        }

        # Add field property columns (avoid name conflicts)
        for prop_name in self.field_properties:
            if prop_name not in data:
                data[prop_name] = property_values[prop_name]
            else:
                logger.warning(
                    f"Field property name '{prop_name}' conflicts with a core column. Skipping.",
                    UserWarning,
                )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=None)
        gdf["z"] = self.channel_locations[:, 2]  # Add elevation coordinate

        return gdf

    # TODO: write function that converts DASLayout to DASLayoutGeographic

    def plot(
        self,
        ax=None,
        show_knots=False,
        knot_color="k",
        direction_scale=None,
        plot_style="channels",
        knot_kwargs=None,
        **kwargs,
    ):
        """
        Plot the DAS layout in a 2D top-down view using local coordinates.

        Arguments:
            ax: Matplotlib axes to plot on. If None, creates new figure and axes.
                Defaults to None.
            show_knots: Whether to plot knot points. Defaults to False.
            knot_color: Color for knot point markers. Defaults to "k" (black).
            direction_scale: Scaling factor for channel direction arrows. If None,
                defaults to 80% of channel spacing. Defaults to None.
            plot_style: Visualization style. Can be:

                - "channels": Plot as direction arrows showing channel orientations
                - "line": Plot as connected line showing cable path

                Defaults to "channels".
            knot_kwargs: Additional keyword arguments for knot point plotting
                (passed to ``ax.scatter``). Defaults to None.
            **kwargs: Additional keyword arguments passed to the primary plotting
                function (``ax.quiver`` for "channels", ``ax.plot`` for "line").

        Returns:
            ``tuple``: (``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            auto_format = True
        else:
            fig = ax.figure
            auto_format = False

        # Set up anchor plotting parameters
        knot_kwargs = knot_kwargs or {}
        knot_kwargs.setdefault("color", knot_color)
        knot_kwargs.setdefault("label", "Knots")
        knot_kwargs.setdefault("zorder", 3)

        # Calculate direction arrow scaling
        channel_scale = direction_scale or (
            self.channel_spacing * 0.8 if self.channel_spacing else 1.0
        )
        if channel_scale <= 0:
            channel_scale = 1.0

        # Plot channels
        if self.n_channels > 0:
            if plot_style == "channels":
                kwargs.setdefault("color", "tab:blue")
                kwargs.setdefault("label", "Channels (Local Coords)")
                self._plot_channels(ax, scale=channel_scale, **kwargs)
            elif plot_style == "line":
                kwargs.setdefault("color", "tab:blue")
                kwargs.setdefault("label", "DAS Cable (Local Coords)")
                kwargs.setdefault("linewidth", 2)
                self._plot_line(ax, **kwargs)
            else:
                raise ValueError(
                    f"Invalid plot_style: {plot_style}. Choose 'channels' or 'line'."
                )
        else:
            logger.warning("No channels to plot.", UserWarning)

        # Plot knots if requested
        if (
            show_knots
            and hasattr(self, "knot_locations")
            and self.knot_locations is not None
            and self.knot_locations.shape[0] > 0
        ):
            self._plot_knots(ax, **knot_kwargs)

        # Format axes if we created them
        if auto_format:
            self._format_axes(ax)

        return fig, ax

    def _plot_channels(self, ax, scale, **kwargs):
        """Plot channels as quiver arrows."""
        x = self.channel_locations[:, 0]
        y = self.channel_locations[:, 1]
        dx = self.channel_directions[:, 0] * scale
        dy = self.channel_directions[:, 1] * scale

        kwargs.setdefault("width", 0.008)
        kwargs.setdefault("headaxislength", 0)
        kwargs.setdefault("headlength", 0)
        kwargs.setdefault("headwidth", 0)
        kwargs.setdefault("pivot", "middle")
        kwargs.setdefault("scale", 1)
        kwargs.setdefault("scale_units", "xy")
        ax.quiver(x, y, dx, dy, **kwargs)

    def _plot_line(self, ax, **kwargs):
        """Plot the spline curve (not just channel locations)."""
        ax.plot(self.channel_locations[:, 0], self.channel_locations[:, 1], **kwargs)

    def _plot_knots(self, ax, **kwargs):
        """Plot knot points."""
        anchor_locs_2d = self.knot_locations[:, :2]
        kwargs.setdefault("marker", "P")
        kwargs.setdefault("linewidth", 0)
        kwargs.setdefault("s", 70)
        ax.scatter(anchor_locs_2d[:, 0], anchor_locs_2d[:, 1], **kwargs)

    def _format_axes(self, ax):
        """Format the plot axes."""
        ax.set_xlabel("Local X coordinate")
        ax.set_ylabel("Local Y coordinate")
        ax.set_title("DAS Layout (Local Coordinates)")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Set plot limits
        points = []
        if self.n_channels > 0:
            points.append(self.channel_locations[:, :2])
        if (
            hasattr(self, "knot_locations")
            and self.knot_locations is not None
            and self.knot_locations.shape[0] > 0
        ):
            points.append(self.knot_locations[:, :2])

        if points:
            points = np.vstack(points)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            x_range = max(x_max - x_min, 1e-6)
            y_range = max(y_max - y_min, 1e-6)
            buffer_x = x_range * 0.1
            buffer_y = y_range * 0.1

            ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
            ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        ax.set_aspect("equal", adjustable="box")

        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique_labels = {}
            for handle, label in zip(handles, labels):
                # Updated variable names for clarity
                if label not in unique_labels:
                    unique_labels[label] = handle
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                facecolor="white",
                framealpha=0.9,
                loc="best",
            )

    def aperture(self):
        """
        Calculate the maximum distance between any two channels in the layout.

        This represents the overall "aperture" or span of the DAS array,
        which is important for seismic analysis capabilities.

        Returns:
            float: Maximum distance between any pair of channels.
                Returns 0.0 if there are fewer than 2 channels.
        """
        if self.n_channels < 2:
            return 0.0
        distances = pdist(self.channel_locations)
        return np.max(distances) if distances.size > 0 else 0.0

    def __str__(self):
        """Return a human-readable string representation of the layout."""
        desc = f"DASLayout (Local Coords, {self.n_channels} channels"

        if hasattr(self, "knot_locations") and self.knot_locations is not None:
            desc += f", {len(self.knot_locations)} anchors"

        if self.channel_spacing is not None:
            desc += f", spacing≈{self.channel_spacing:.2f}m"

        if self.cable_length is not None:
            desc += f", length={self.cable_length:.2f}m"
        else:
            desc += ", length=unknown"

        if self.signal_decay is not None:
            desc += f", decay={self.signal_decay:.2f} dB/km"
        else:
            desc += ", decay=None"

        desc += f", {len(self.field_properties)} field properties)"
        return desc

    def _repr_html_(self):
        """
        HTML representation for Jupyter notebooks.

        Returns:
            str: Formatted HTML representation of the layout
        """

        def get_data_desc(data_attr):
            """Describe the type and nature of field property data."""
            if isinstance(data_attr, xr.DataArray):
                coords = ", ".join(list(data_attr.coords.keys()))
                return f"Gridded (xarray: {coords})"
            elif isinstance(data_attr, (int, float)):
                return f"Constant ({data_attr})"
            elif callable(data_attr):
                if (
                    hasattr(data_attr, "__qualname__")
                    and "DASLayoutGeographic" in data_attr.__qualname__
                ):
                    return "Wrapped Geographic Callable/xarray"
                else:
                    return "Callable function (local coords)"
            elif data_attr is None:
                return "None"
            else:
                return f"Unknown ({type(data_attr).__name__})"

        # Build field properties section
        props_html = ""
        if self.field_properties:
            props_html += "<li><b>Field Properties:</b><ul>"
            for name, data in self.field_properties.items():
                props_html += f"<li><i>{name}:</i> {get_data_desc(data)}</li>"
            props_html += "</ul></li>"
        else:
            props_html += "<li><b>Field Properties:</b> None</li>"

        # Signal decay information
        decay_str = (
            f"{self.signal_decay:.2f} dB/km"
            if self.signal_decay is not None
            else "None"
        )
        decay_html = f"<li><b>Signal Decay:</b> {decay_str}</li>"

        # Layout summary information
        anchor_count = (
            len(self.knot_locations)
            if hasattr(self, "knot_locations") and self.knot_locations is not None
            else 0
        )

        spacing_str = (
            f"{self.channel_spacing:.2f}m"
            if self.channel_spacing is not None
            else "N/A"
        )

        length_str = (
            f"{self.cable_length:.2f}m" if self.cable_length is not None else "Unknown"
        )

        html = f"""
        <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; 
                    margin: 5px; font-family: sans-serif">
            <strong>DASLayout (Local Coordinates)</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li><b>Channels:</b> {self.n_channels} (3D positioned)</li>
                <li><b>Cable Length:</b> {length_str}</li>
                <li><b>Knots:</b> {anchor_count}</li>
                <li><b>Channel Spacing:</b> {spacing_str}</li>
                {props_html}
                {decay_html}
            </ul>
        </div>
        """
        return html

    def __repr__(self):
        return self.__str__()
    
    def translate(self, offset):
        """
        Translate the entire layout by a specified offset in local coordinates.

        This method shifts all channel and knot locations by the given (dx, dy)
        offset, effectively moving the entire layout without altering its shape
        or orientation.

        Arguments:
            offset: Tuple or array-like of (dx, dy) translation in meters.

        Raises:
            ValueError: If offset is not a tuple or array-like of length 2.
        """
        offset = np.asarray(offset)
        if offset.shape != (2,):
            raise ValueError("Offset must be a tuple or array-like of length 2 (dx, dy).")

        # Translate channel locations
        if self.n_channels > 0:
            self.channel_locations[:, 0] += offset[0]
            self.channel_locations[:, 1] += offset[1]

        # Translate knot locations if they exist
        if hasattr(self, "knot_locations") and self.knot_locations is not None:
            self.knot_locations[:, 0] += offset[0]
            self.knot_locations[:, 1] += offset[1]

        # Note: Reference point and CRS remain unchanged as this is a local translation

    def copy(self):
        """
        Create a deep copy of the DASLayout instance.

        Returns:
            DASLayout: A new instance that is a deep copy of the current layout.
        """
        return deepcopy(self)



# TODO: make sure this relies as much as possible on DASLayout and is appropriately tested and documented
class DASLayoutGeographic(DASLayout):
    r"""
    Georeferenced DAS layout using local Cartesian coordinates with CRS transformation.

    This class extends DASLayout to support georeferenced DAS layouts by:
    1. Performing channel calculations in a local Cartesian coordinate system
    2. Using a reference point and CRS to provide geographic context
    3. Handling coordinate transformation during data export

    The local coordinate system origin (0,0) corresponds to the reference point
    in the specified CRS. This approach provides:

    - Computational efficiency (avoids complex geodesic calculations)
    - Numerical stability (avoids precision issues with large geographic coordinates)
    - Flexibility (supports both geographic and projected reference systems)

    Arguments:
        knots: 2D array (N x 2) of local Cartesian coordinates (x, y) in meters,
            relative to an origin (0,0).
        reference_point: Coordinates (longitude, latitude) or (x, y) of the local
            origin (0,0) in the system defined by ``crs``.
        crs: The Coordinate Reference System of the ``reference_point`` and the
            target system for georeferencing.
        spacing: Desired spacing between channels along the cable in meters.
        elevation: Elevation data relative to the datum. If gridded or callable,
            it should expect local Cartesian coordinates (x, y). Defaults to 0.
        field_properties: Dictionary mapping property names to their data:

            - Constants (float/int): Applied uniformly
            - ``xr.DataArray``: Gridded data for spatial interpolation
            - ``callable``: Function f(x,y) in local coordinate system

            Properties should be defined in the local coordinate system.
            Defaults to None.
        signal_decay: Signal decay rate in dB/km. If None or ≤0, signal strength
            remains constant (1.0). Defaults to None.
        wrap_around: Whether to connect the last knot to the first to form a closed loop.
            Defaults to False.
        t: Custom parameterization values for knot points controlling spline behavior.
            If None, calculated automatically from cumulative chord length.
            Defaults to None.
        **kwargs: Additional arguments passed to ``scipy.interpolate.splprep`` for spline fitting.

    Raises:
        ValueError: If spacing is not positive, if knots are not a valid 2D array (N x 2),
            if CRS is invalid, or if reference_point is not a valid coordinate pair.

    Coordinate System Handling:
        - **Geographic CRS** (e.g., EPSG:4326): The reference point represents
            (longitude, latitude) and defines the center of a local Azimuthal
            Equidistant projection for internal calculations.
        - **Projected CRS** (e.g., UTM): The reference point represents (x, y)
            coordinates within that projection, serving as an offset for the
            local coordinate system.
    """

    def __init__(
        self,
        knots=None,
        reference_point=None,
        crs=None,
        spacing=None,
        elevation=0,
        field_properties={},
        signal_decay=None,
        wrap_around=False,
        t=None,
        anchors=None,  # backward compatibility
        **kwargs,
    ):
        # Handle backward compatibility for anchors parameter
        if anchors is not None:
            if knots is not None:
                raise ValueError(
                    "Cannot specify both 'knots' and 'anchors' parameters. Use 'knots'."
                )

            logger.warning(
                "The 'anchors' parameter is deprecated. Use 'knots' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            knots = anchors

        # Validate required parameters
        if knots is None:
            raise TypeError("Missing required argument: 'knots'")
        if reference_point is None:
            raise TypeError("Missing required argument: 'reference_point'")
        if crs is None:
            raise TypeError("Missing required argument: 'crs'")
        if spacing is None:
            raise TypeError("Missing required argument: 'spacing'")

        # Validate and store reference point
        self.reference_point = np.asarray(reference_point)
        if self.reference_point.shape != (2,):
            raise ValueError(
                "reference_point must be a tuple or array of length 2 (lon, lat or x, y)."
            )

        # Validate and store CRS
        try:
            self.crs = pyproj.CRS(crs)
        except Exception as e:
            raise ValueError(f"Invalid CRS provided: {crs}. Error: {e}")

        # Store the original local knots for reference
        self.knots_local_input = np.asarray(knots)

        # Validate knots (local Cartesian)
        if self.knots_local_input.ndim != 2 or self.knots_local_input.shape[1] != 2:
            raise ValueError(
                "Knots must be a 2D array (N x 2) of local Cartesian coordinates."
            )
        if self.knots_local_input.shape[0] == 0:
            raise ValueError("Knots array cannot be empty.")

        # Initialize parent class with local coordinates and properties
        super().__init__(
            knots=self.knots_local_input,
            spacing=spacing,
            elevation=elevation,  # Expects local coords if functional/gridded
            field_properties=field_properties,  # Expects local coords if functional/gridded
            signal_decay=signal_decay,
            wrap_around=wrap_around,
            t=t,
            **kwargs,
        )

    def get_gdf(self, output_crs=None):
        """
        Return a GeoDataFrame with channel locations transformed to the specified CRS.

        The transformation process depends on whether the primary CRS (``self.crs``)
        is geographic or projected:

        - If ``self.crs`` is geographic (e.g., EPSG:4326), the ``reference_point``
            (lon, lat) defines the center of a local Azimuthal Equidistant (AEQD)
            projection. Local Cartesian coordinates are transformed from this
            local AEQD system to the ``output_crs``.
        - If ``self.crs`` is projected (e.g., UTM), the ``reference_point`` (x, y)
            defines the origin offset within that projected system. Local Cartesian
            coordinates are translated by this offset within ``self.crs``, and then
            potentially re-projected to ``output_crs``.

        Arguments:
            output_crs: The target CRS for the output GeoDataFrame. If None, the CRS
                provided during initialization (``self.crs``) is used. Defaults to None.

        Returns:
            :class:`geopandas.GeoDataFrame`: GeoDataFrame with channel points and attributes,
                with geometry in the target CRS.
        """

        # Get GDF in local Cartesian coordinates from parent
        gdf_local = super().get_gdf()

        # Determine the final target CRS
        target_crs = pyproj.CRS(output_crs) if output_crs else self.crs

        if gdf_local.empty:
            # Return an empty GDF with the target CRS and consistent columns
            empty_cols = list(gdf_local.columns)  # Get columns from parent call
            if "geometry" not in empty_cols:
                empty_cols.append("geometry")
            if "z" not in empty_cols:
                empty_cols.append("z")

            empty_gdf = gpd.GeoDataFrame(
                columns=empty_cols, crs=target_crs, geometry=[]
            )
            empty_gdf["geometry"] = gpd.GeoSeries(dtype="geometry", crs=target_crs)
            # Ensure essential columns exist
            for col in ["channel_id", "type", "geometry", "z", "signal_strength"]:
                if col not in empty_gdf.columns:
                    empty_gdf[col] = []
            return empty_gdf

        # Extract local coordinates
        x_local = gdf_local.geometry.x.values
        y_local = gdf_local.geometry.y.values
        # Get Z coordinate if it exists, otherwise default to zeros
        z_values = (
            gdf_local["z"].values
            if "z" in gdf_local.columns
            else np.zeros(len(gdf_local))
        )

        # Perform transformation based on the nature of self.crs
        if self.crs.is_geographic:
            # Case 1: Reference point is Lon/Lat, defining a local tangent plane (AEQD)
            ref_lon, ref_lat = self.reference_point
            # Define the local Cartesian system using AEQD centered at the reference point
            local_crs = pyproj.CRS(
                proj="aeqd", lat_0=ref_lat, lon_0=ref_lon, datum="WGS84", units="m"
            )

            # Create transformer from local AEQD to the target CRS
            transformer = pyproj.Transformer.from_crs(
                local_crs, target_crs, always_xy=True
            )
            x_transformed, y_transformed = transformer.transform(x_local, y_local)

            # Create new geometries in the target CRS
            geometries = gpd.points_from_xy(x_transformed, y_transformed, z=z_values)

        else:
            # Case 2: Reference point is X/Y in a projected CRS, defining an offset
            ref_x, ref_y = self.reference_point
            # Translate local coordinates by the reference point offset
            x_translated = x_local + ref_x
            y_translated = y_local + ref_y

            # Create geometries initially in self.crs
            geometries_initial = gpd.points_from_xy(
                x_translated, y_translated, z=z_values
            )
            gdf_temp = gpd.GeoDataFrame(geometry=geometries_initial, crs=self.crs)

            # Re-project to target_crs if necessary
            if not self.crs.equals(target_crs):
                gdf_temp = gdf_temp.to_crs(target_crs)

            geometries = gdf_temp.geometry

        # Create the final GeoDataFrame
        # Copy attributes from the local GDF, keep other columns
        gdf_attributes = gdf_local.drop(columns=["geometry", "z"], errors="ignore")
        # Create the new GeoDataFrame with attributes, new geometry, and target CRS
        gdf_output = gpd.GeoDataFrame(
            gdf_attributes, geometry=geometries, crs=target_crs
        )

        # Add back the z coordinate if geometry is PointZ (might be lost in reprojection)
        if geometries.has_z.all():
            gdf_output["z"] = geometries.z
        else:
            # If Z was lost or not present, add the original Z values back as a separate column
            gdf_output["z"] = z_values

        # Note: Direction vectors (u_x, u_y, u_z) remain in the local frame's orientation.
        # Transforming direction vectors requires complex handling of CRS distortions
        # and is not implemented here. They represent the local orientation at the
        # georeferenced point.

        return gdf_output

    def plot(
        self,
        ax=None,
        plot_crs=None,
        use_basemap=False,
        basemap_provider=None,
        basemap_kwargs=None,
        plot_style="line",
        show_knots=True,
        knot_color="k",
        knot_kwargs=None,
        **kwargs,
    ):
        """
        Plot the DAS layout in geographic coordinates, optionally with a basemap.

        Arguments:
            ax: Matplotlib axes to plot on. If None, a new figure and axes are created.
                Defaults to None.
            plot_crs: The CRS to use for plotting. If None, uses the layout's native CRS
                (``self.crs``). If ``use_basemap`` is True, data will be projected to Web
                Mercator (EPSG:3857). Defaults to None.
            use_basemap: Whether to add a contextual basemap using ``contextily``.
                Requires ``contextily`` to be installed. Defaults to False.
            basemap_provider: The basemap provider to use. See ``contextily``
                documentation for options. Defaults to ``contextily``'s default.
            basemap_kwargs: Additional keyword arguments passed to
                ``contextily.add_basemap``. Defaults to None.
            plot_style: How to represent the layout:

                - 'line': Plot the cable path as a line.
                - 'points': Plot individual channel locations as points.
                - 'channels': Plot channels with local direction arrows (use with caution
                    in geographic plots as directions are not transformed).

                Defaults to "line".
            show_knots: Whether to plot the knot points. Defaults to True.
            knot_color: Color for the knot points. Defaults to "k".
            knot_kwargs: Additional keyword arguments passed to ``ax.scatter`` for
                plotting knots. Defaults to None.
            **kwargs: Additional keyword arguments passed to the plotting function
                (``gdf.plot`` for line/points, ``ax.quiver`` for channels).

        Returns:
            ``tuple``: (``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``)

        Raises:
            ImportError: If ``use_basemap`` is True and ``contextily`` is not installed.
            ValueError: If an invalid ``plot_style`` is provided.
        """
        # Check if there's anything to plot
        has_channels = self.n_channels > 0
        has_knots = (
            show_knots
            and hasattr(self, "knot_locations")
            and self.knot_locations is not None
            and self.knot_locations.shape[0] > 0
        )

        if not has_channels and not has_knots:
            logger.warning("No channels or knots to plot.", UserWarning)
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.set_title("Empty DAS Layout")
            return fig, ax

        _basemap_kwargs = basemap_kwargs or {}
        _knot_kwargs = knot_kwargs or {}
        _knot_kwargs.setdefault("color", knot_color)
        _knot_kwargs.setdefault("label", "Knots")
        _knot_kwargs.setdefault("zorder", 5)  # Above basemap and layout
        _knot_kwargs.setdefault("marker", "P")
        _knot_kwargs.setdefault("s", 70)

        # Determine target CRS for plotting
        target_crs = self.crs  # Default to native CRS
        if plot_crs:
            target_crs = pyproj.CRS(plot_crs)

        # If using basemap, must plot in Web Mercator (EPSG:3857)
        if use_basemap:
            if not _CONTEXTILY_AVAILABLE:
                raise ImportError(
                    "Please install contextily to use basemaps: pip install contextily"
                )
            basemap_crs = pyproj.CRS("EPSG:3857")
            target_crs = basemap_crs  # Override target_crs for basemap compatibility

        # Get GeoDataFrame in the target plotting CRS only if needed for plotting channels/line
        gdf_plot = None
        if has_channels:
            try:
                gdf_plot = self.get_gdf(output_crs=target_crs)
                if gdf_plot.empty:
                    logger.warning("Layout is empty after transformation.", UserWarning)
                    has_channels = False  # Treat as if no channels if GDF is empty
            except Exception as e:
                logger.warning(
                    f"Could not transform layout to target CRS ({target_crs}): {e}",
                    RuntimeWarning,
                )
                # Fallback: try plotting in original CRS without basemap
                target_crs = self.crs
                try:
                    gdf_plot = self.get_gdf(output_crs=target_crs)
                    if gdf_plot.empty:
                        has_channels = False
                except Exception as e2:
                    logger.warning(
                        f"Could not transform layout to original CRS ({target_crs}) either: {e2}",
                        RuntimeWarning,
                    )
                    has_channels = False  # Cannot plot channels

                use_basemap = False  # Disable basemap on fallback
                logger.warning(
                    f"Falling back to plotting in original CRS ({target_crs}) without basemap.",
                    RuntimeWarning,
                )

        # Setup axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            auto_format = True
        else:
            fig = ax.figure
            auto_format = False

        # Plot layout (channels/line)
        if has_channels and gdf_plot is not None and not gdf_plot.empty:
            if plot_style == "line":
                kwargs.setdefault("label", "DAS Cable")
                kwargs.setdefault("linewidth", 2)
                # Need to convert points to line for plotting
                try:
                    line_geom = LineString(gdf_plot.geometry.tolist())
                    line_gdf = gpd.GeoDataFrame(
                        [1], geometry=[line_geom], crs=gdf_plot.crs
                    )
                    line_gdf.plot(ax=ax, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Could not create or plot LineString: {e}. Plotting points instead.",
                        UserWarning,
                    )
                    # Fallback to points if line creation fails (e.g., single point)
                    point_kwargs = kwargs.copy()
                    point_kwargs.setdefault("label", "channels")
                    point_kwargs.setdefault("marker", ".")
                    point_kwargs.setdefault("s", 10)
                    point_kwargs.pop("linewidth", None)  # Remove line specific arg
                    gdf_plot.plot(ax=ax, **point_kwargs)

            elif plot_style == "points":
                kwargs.setdefault("label", "channels")
                kwargs.setdefault("marker", ".")
                kwargs.setdefault("s", 10)
                gdf_plot.plot(ax=ax, **kwargs)

            elif plot_style == "channels":
                logger.warning(
                    "Plotting channels with direction arrows ('channels' style) in geographic coordinates."
                    " Arrows represent LOCAL direction and are NOT transformed by the CRS.",
                    UserWarning,
                )
                # Plot points first
                point_kwargs = kwargs.copy()
                point_kwargs.setdefault("label", "channels (Local Direction)")
                point_kwargs.setdefault("marker", ".")
                point_kwargs.setdefault("s", 5)
                point_kwargs.pop("width", None)  # Remove quiver args if present
                point_kwargs.pop("scale", None)
                point_kwargs.pop("scale_units", None)
                point_kwargs.pop("angles", None)
                point_kwargs.pop("pivot", None)
                gdf_plot.plot(ax=ax, **point_kwargs)

                # Overlay quiver with local directions
                x_plot = gdf_plot.geometry.x.values
                y_plot = gdf_plot.geometry.y.values
                dx_local = gdf_plot["u_x"].values
                dy_local = gdf_plot["u_y"].values

                # Scale arrows based on channel spacing or estimate from points
                if self.channel_spacing is not None and self.channel_spacing > 0:
                    avg_dist = self.channel_spacing
                elif len(x_plot) > 1:
                    avg_dist = np.mean(
                        np.sqrt(np.diff(x_plot) ** 2 + np.diff(y_plot) ** 2)
                    )
                else:
                    avg_dist = 1.0  # Fallback scale

                scale_factor = avg_dist * 0.5

                q_kwargs = kwargs.copy()
                q_kwargs.setdefault("color", point_kwargs.get("color", "blue"))
                q_kwargs.setdefault("width", 0.003)
                q_kwargs.setdefault("scale", 1)
                q_kwargs.setdefault("scale_units", "xy")
                q_kwargs.setdefault("angles", "xy")
                q_kwargs.setdefault("pivot", "middle")
                q_kwargs.pop("label", None)
                q_kwargs.pop("marker", None)
                q_kwargs.pop("s", None)
                q_kwargs.pop("linewidth", None)

                ax.quiver(
                    x_plot,
                    y_plot,
                    dx_local * scale_factor,
                    dy_local * scale_factor,
                    **q_kwargs,
                )

            else:
                raise ValueError(
                    f"Invalid plot_style: {plot_style}. Choose 'line', 'points', or 'channels'."
                )

        # Plot knots if requested
        if has_knots:
            try:
                # Transform knot points to target CRS
                x_local_knots = self.knot_locations[:, 0]
                y_local_knots = self.knot_locations[:, 1]

                if self.crs.is_geographic:
                    ref_lon, ref_lat = self.reference_point
                    local_crs_knots = pyproj.CRS(
                        proj="aeqd",
                        lat_0=ref_lat,
                        lon_0=ref_lon,
                        datum="WGS84",
                        units="m",
                    )
                    transformer_knots = pyproj.Transformer.from_crs(
                        local_crs_knots, target_crs, always_xy=True
                    )
                    x_knots_plot, y_knots_plot = transformer_knots.transform(
                        x_local_knots, y_local_knots
                    )
                else:  # Projected CRS
                    ref_x, ref_y = self.reference_point
                    x_translated_knots = x_local_knots + ref_x
                    y_translated_knots = y_local_knots + ref_y
                    # Re-project if needed
                    if not self.crs.equals(target_crs):
                        transformer_knots = pyproj.Transformer.from_crs(
                            self.crs, target_crs, always_xy=True
                        )
                        x_knots_plot, y_knots_plot = transformer_knots.transform(
                            x_translated_knots, y_translated_knots
                        )
                    else:
                        x_knots_plot, y_knots_plot = (
                            x_translated_knots,
                            y_translated_knots,
                        )

                ax.scatter(x_knots_plot, y_knots_plot, **_knot_kwargs)

            except Exception as e:
                logger.warning(f"Could not transform or plot knots: {e}", RuntimeWarning)

        # Add basemap if requested
        if use_basemap:
            plot_crs_str = target_crs.to_string() if target_crs else None
            if plot_crs_str:
                try:
                    ctx.add_basemap(
                        ax, crs=plot_crs_str, source=basemap_provider, **_basemap_kwargs
                    )
                except Exception as e:
                    logger.warning(f"Failed to add basemap: {e}", RuntimeWarning)
            else:
                logger.warning(
                    "Cannot add basemap without a valid target CRS.", RuntimeWarning
                )

        # Format axes
        if auto_format:
            self._format_axes_geographic(ax, target_crs)

        return fig, ax

    def _format_axes_geographic(self, ax, target_crs):
        """
        Format the plot axes for geographic coordinate plots.

        Arguments:
            ax: Matplotlib axes to format
            target_crs: Target CRS used for plotting
        """
        ax.set_title(
            f"DAS Layout (CRS: {target_crs.name if target_crs else 'Unknown'})"
        )

        # Set axis labels based on CRS type
        if target_crs and target_crs.is_geographic:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        elif target_crs:
            try:
                x_unit = target_crs.axis_info[0].unit_name
                y_unit = target_crs.axis_info[1].unit_name
                ax.set_xlabel(f"Easting ({x_unit})")
                ax.set_ylabel(f"Northing ({y_unit})")
            except:
                ax.set_xlabel("X coordinate")
                ax.set_ylabel("Y coordinate")
        else:
            ax.set_xlabel("X coordinate")
            ax.set_ylabel("Y coordinate")

        # Set appropriate aspect ratio based on CRS type and plot location
        try:
            # Get all plotted geometries to determine bounds
            all_geoms = [
                child for child in ax.get_children() if hasattr(child, "get_data")
            ]

            # Skip if no geometries
            if not all_geoms:
                ax.set_aspect("auto")
                return

            # Calculate bounds from all plotted data
            xs, ys = [], []
            for geom in all_geoms:
                try:
                    x, y = geom.get_data()
                    if len(x) > 0 and len(y) > 0:
                        xs.extend(x)
                        ys.extend(y)
                except:
                    pass

            # Set aspect based on CRS type and location
            if xs and ys:
                if target_crs and (target_crs.is_projected or abs(np.mean(ys)) < 60):
                    ax.set_aspect("equal", adjustable="box")
                elif not target_crs:
                    ax.set_aspect("equal", adjustable="box")
                else:
                    ax.set_aspect("auto")
            else:
                ax.set_aspect("auto")

        except Exception:
            # Fallback if aspect setting fails
            ax.set_aspect("auto")

        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique_labels = {}
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels[label] = handle
            ax.legend(
                unique_labels.values(),
                unique_labels.keys(),
                facecolor="white",
                framealpha=0.9,
                loc="best",
            )

        ax.grid(True, linestyle="--", alpha=0.6)

    def __str__(self):
        """Return a human-readable string representation of the georeferenced layout."""
        desc = f"DASLayoutGeographic ({self.n_channels} channels"
        ref_pt_str = f"({self.reference_point[0]:.4f}, {self.reference_point[1]:.4f})"
        crs_name = self.crs.name if self.crs else "N/A"
        crs_type = "Geographic" if self.crs.is_geographic else "Projected"
        desc += f", RefPt={ref_pt_str} in {crs_type} CRS"

        if hasattr(self, "knot_locations") and self.knot_locations is not None:
            desc += f", {len(self.knot_locations)} knots"

        if self.channel_spacing is not None:
            desc += f", spacing≈{self.channel_spacing:.2f}m"

        if self.cable_length is not None:
            desc += f", length={self.cable_length:.2f}m"
        else:
            desc += ", length=unknown"

        if self.signal_decay is not None:
            desc += f", decay={self.signal_decay:.2f} dB/km"
        else:
            desc += ", decay=None"

        desc += f", {len(self.field_properties)} field properties)"
        return desc

    def _repr_html_(self):
        """HTML representation for Jupyter notebooks."""

        def get_data_desc(data_attr):
            """Describe the type and nature of field property data."""
            if isinstance(data_attr, xr.DataArray):
                coords = ", ".join(list(data_attr.coords.keys()))
                return f"Gridded (xarray: {coords})"
            elif isinstance(data_attr, (int, float)):
                return f"Constant ({data_attr})"
            elif callable(data_attr):
                return "Callable function (local coords)"
            elif data_attr is None:
                return "None"
            else:
                return f"Unknown ({type(data_attr).__name__})"

        # Build field properties section
        props_html = ""
        if self.field_properties:
            props_html += "<li><b>Field Properties (local coords):</b><ul>"
            for name, data in self.field_properties.items():
                props_html += f"<li><i>{name}:</i> {get_data_desc(data)}</li>"
            props_html += "</ul></li>"
        else:
            props_html += "<li><b>Field Properties:</b> None</li>"

        # Reference point and CRS information
        ref_pt_str = f"({self.reference_point[0]:.4f}, {self.reference_point[1]:.4f})"
        crs_name = self.crs.name if self.crs else "N/A"
        crs_epsg = (
            f"EPSG:{self.crs.to_epsg()}" if self.crs and self.crs.to_epsg() else "N/A"
        )
        crs_type = "Geographic" if self.crs.is_geographic else "Projected"

        # Signal decay information
        decay_str = (
            f"{self.signal_decay:.2f} dB/km"
            if self.signal_decay is not None
            else "None"
        )
        decay_html = f"<li><b>Signal Decay:</b> {decay_str}</li>"

        # Layout summary information
        knot_count = (
            len(self.knot_locations)
            if hasattr(self, "knot_locations") and self.knot_locations is not None
            else 0
        )

        spacing_str = (
            f"{self.channel_spacing:.2f}m"
            if self.channel_spacing is not None
            else "N/A"
        )
        length_str = (
            f"{self.cable_length:.2f}m" if self.cable_length is not None else "Unknown"
        )

        html = f"""
        <div style="border: 1px solid #ccc; border-radius: 5px; padding: 10px; 
                    margin: 5px; font-family: sans-serif">
            <strong>DASLayoutGeographic</strong>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li><b>Channels:</b> {self.n_channels} (3D positioned)</li>
                <li><b>Reference Point (Local Origin):</b> {ref_pt_str}</li>
                <li><b>Coordinate System (CRS):</b> {crs_name} ({crs_epsg}, Type: {crs_type})</li>
                <li><b>Cable Length:</b> {length_str}</li>
                <li><b>Knots:</b> {knot_count}</li>
                <li><b>Channel Spacing:</b> {spacing_str}</li>
                {props_html}
                {decay_html}
            </ul>
            <small><i>Note: Internal calculations use local Cartesian coordinates. Direction vectors (u_x, u_y, u_z) represent local cable orientation.</i></small>
        </div>
        """
        return html

    def __repr__(self):
        """Technical representation."""
        return self.__str__()


# =============================================================================
# COMMENTED FUNCTIONS - LEGACY CODE
# =============================================================================
# The following functions are kept for reference and potential future use.
# They represent alternative implementations and helper functions that were
# part of the development process but are not currently used in the main
# codebase.
# =============================================================================


# def _plot_knots_helper(ax, knot_locations, **kwargs):
#     """Helper function to plot knot points (2D)."""
#     if knot_locations is not None and knot_locations.shape[0] > 0:
#         kwargs.setdefault("marker", "P")
#         kwargs.setdefault("linewidth", 0)
#         kwargs.setdefault("s", 70)
#         kwargs.setdefault("label", "Knots")
#         kwargs.setdefault("zorder", 3) # Ensure anchors are visible
#         ax.scatter(knot_locations[:, 0], knot_locations[:, 1], **kwargs)

# def _format_axes_local(ax, channel_locations, knot_locations):
#     """Helper function to format axes for local coordinate plots."""
#     ax.set_xlabel("Local X coordinate")
#     ax.set_ylabel("Local Y coordinate")
#     ax.set_title("DAS Layout (Local Coordinates)")
#     ax.grid(True, linestyle="--", alpha=0.6)

#     # Set plot limits
#     points = []
#     if channel_locations is not None and channel_locations.shape[0] > 0:
#         points.append(channel_locations[:, :2])
#     if knot_locations is not None and knot_locations.shape[0] > 0:
#         points.append(knot_locations[:, :2])

#     if points:
#         points = np.vstack(points)
#         x_min, y_min = np.min(points, axis=0)
#         x_max, y_max = np.max(points, axis=0)
#         x_range = max(x_max - x_min, 1e-6)
#         y_range = max(y_max - y_min, 1e-6)
#         buffer_x = x_range * 0.1
#         buffer_y = y_range * 0.1

#         ax.set_xlim(x_min - buffer_x, x_max + buffer_x)
#         ax.set_ylim(y_min - buffer_y, y_max + buffer_y)
#     else:
#         ax.set_xlim(-1, 1)
#         ax.set_ylim(-1, 1)

#     ax.set_aspect("equal", adjustable="box")

#     # Create legend with unique entries
#     handles, labels = ax.get_legend_handles_labels()
#     if labels:
#         unique_labels = {}
#         for h, l in zip(handles, labels):
#             if l not in unique_labels:
#                 unique_labels[l] = h
#         ax.legend(
#             unique_labels.values(),
#             unique_labels.keys(),
#             facecolor="white",
#             framealpha=0.9,
#             loc="best",
#         )

# def plot_das_layout_local(
#     layout,
#     ax=None,
#     show_knots=False,
#     knot_color="k",
#     direction_scale=None,
#     plot_style="channels",
#     knot_kwargs=None,
#     **kwargs,
# ):
#     """
#     Plot the DAS layout in its local 2D Cartesian coordinate system.

#     Args:
#         layout (DASLayout): The DASLayout object to plot.
#         ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None,
#             a new figure and axes are created. Defaults to None.
#         show_knots (bool, optional): Whether to plot the knot points.
#             Defaults to False.
#         knot_color (str, optional): Color for the knot points. Defaults to "k".
#         direction_scale (float, optional): Scaling factor for channel direction arrows.
#             If None, defaults to 80% of channel spacing. Defaults to None.
#         plot_style (str, optional): How to represent the layout: 'channels' (arrows),
#             'line'. Defaults to "channels".
#         knot_kwargs (dict, optional): Additional keyword arguments passed to
#             `ax.scatter` for plotting anchors. Defaults to None.
#         **kwargs: Additional keyword arguments passed to the primary plotting function
#             (`ax.quiver` for 'channels', `ax.plot` for 'line').

#     Returns:
#         tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         auto_format = True
#     else:
#         fig = ax.figure
#         auto_format = False

#     _knot_kwargs = knot_kwargs or {}
#     _knot_kwargs.setdefault("color", knot_color)

#     channel_scale = direction_scale or (
#         layout.channel_spacing * 0.8 if layout.channel_spacing else 1.0
#     )
#     if channel_scale <= 0:
#         channel_scale = 1.0

#     # Plot channels or Line
#     if layout.n_channels > 0:
#         x = layout.channel_locations[:, 0]
#         y = layout.channel_locations[:, 1]

#         if plot_style == "channels":
#             dx = layout.channel_directions[:, 0] * channel_scale
#             dy = layout.channel_directions[:, 1] * channel_scale
#             kwargs.setdefault("color", "tab:blue")
#             kwargs.setdefault("label", "channels (Local Coords)")
#             kwargs.setdefault("width", 0.008)
#             kwargs.setdefault("headaxislength", 0)
#             kwargs.setdefault("headlength", 0)
#             kwargs.setdefault("headwidth", 0)
#             kwargs.setdefault("pivot", "middle")
#             kwargs.setdefault("scale", 1)
#             kwargs.setdefault("scale_units", "xy")
#             ax.quiver(x, y, dx, dy, **kwargs)
#         elif plot_style == "line":
#             kwargs.setdefault("color", "tab:blue")
#             kwargs.setdefault("label", "DAS Cable (Local Coords)")
#             kwargs.setdefault("linewidth", 2)
#             ax.plot(x, y, **kwargs)
#         else:
#             raise ValueError(
#                 f"Invalid plot_style: {plot_style}. Choose 'channels' or 'line'."
#             )
#     else:
#         logger.warning("No channels to plot.", UserWarning)

#     # Plot Anchors
#     if (
#         show_knots
#         and hasattr(layout, "knot_locations")
#         and layout.knot_locations is not None
#     ):
#         _plot_knots_helper(ax, layout.knot_locations[:, :2], **_knot_kwargs)

#     # Format Axes
#     if auto_format:
#         anchor_locs = layout.knot_locations if hasattr(layout, "knot_locations") else None
#         _format_axes_local(ax, layout.channel_locations, anchor_locs)

#     return fig, ax


# def _format_axes_geographic(ax, target_crs, plotted_data_bounds=None):
#     """Helper function to format axes for geographic coordinate plots."""
#     ax.set_title(f"DAS Layout (CRS: {target_crs.name if target_crs else 'Unknown'})")
#     if target_crs and target_crs.is_geographic:
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#     elif target_crs:
#         try:
#             x_unit = target_crs.axis_info[0].unit_name
#             y_unit = target_crs.axis_info[1].unit_name
#             ax.set_xlabel(f"Easting ({x_unit})")
#             ax.set_ylabel(f"Northing ({y_unit})")
#         except: # Fallback if axis info fails
#             ax.set_xlabel("X coordinate")
#             ax.set_ylabel("Y coordinate")
#     else:
#         ax.set_xlabel("X coordinate")
#         ax.set_ylabel("Y coordinate")

#     # Set aspect ratio based on CRS type and data range
#     try:
#         if plotted_data_bounds is not None:
#              minx, miny, maxx, maxy = plotted_data_bounds
#              # Simple aspect ratio correction for projected or geographic near equator
#              if target_crs and (target_crs.is_projected or abs(np.mean([miny, maxy])) < 60):
#                  ax.set_aspect('equal', adjustable='box')
#              # For high latitude geographic, 'equal' can be distorted, let matplotlib decide
#              # If CRS unknown, default to equal
#              elif not target_crs:
#                   ax.set_aspect('equal', adjustable='box')
#              else:
#                   ax.set_aspect('auto')
#         else: # Fallback if no bounds calculated
#              ax.set_aspect('auto')
#     except Exception: # Catch potential issues with bounds or CRS
#          ax.set_aspect('auto')

#     # Create legend with unique entries
#     handles, labels = ax.get_legend_handles_labels()
#     if labels:
#         unique_labels = {}
#         for h, l in zip(handles, labels):
#             if l not in unique_labels:
#                 unique_labels[l] = h
#         ax.legend(
#             unique_labels.values(),
#             unique_labels.keys(),
#             facecolor="white",
#             framealpha=0.9,
#             loc="best",
#         )
#     ax.grid(True, linestyle="--", alpha=0.6)


# def plot_das_layout_geographic(
#     layout_geo,
#     ax=None,
#     plot_crs=None,
#     use_basemap=False,
#     basemap_provider=None, # Use contextily default if None
#     basemap_kwargs=None,
#     plot_style="line", # 'line' or 'points' recommended for geographic
#     show_knots=True,
#     knot_color="k",
#     knot_kwargs=None,
#     **kwargs,
# ):
#     """
#     Plot the georeferenced DAS layout, optionally with a basemap.

#     Args:
#         layout_geo (DASLayoutGeographic): The DASLayoutGeographic object to plot.
#         ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. If None,
#             a new figure and axes are created. Defaults to None.
#         plot_crs (str or pyproj.CRS, optional): The CRS to use for plotting.
#             If None, uses the layout's native CRS (`layout_geo.crs`). If `use_basemap`
#             is True, data will be projected to Web Mercator (EPSG:3857)
#             regardless of this setting, unless `plot_crs` is already 3857.
#             Defaults to None.
#         use_basemap (bool, optional): Whether to add a contextual basemap
#             using `contextily`. Requires `contextily` to be installed.
#             Defaults to False.
#         basemap_provider (contextily provider object or str, optional): The
#             basemap provider to use. Defaults to `contextily`'s default.
#         basemap_kwargs (dict, optional): Additional keyword arguments passed to
#             `contextily.add_basemap`. Defaults to None.
#         plot_style (str, optional): How to represent the layout: 'line', 'points',
#             or 'channels' (local directions). Defaults to "line".
#         show_knots (bool, optional): Whether to plot the knot points.
#             Defaults to True.
#         knot_color (str, optional): Color for the knot points. Defaults to "k".
#         knot_kwargs (dict, optional): Additional keyword arguments passed to
#             `ax.scatter` for plotting anchors. Defaults to None.
#         **kwargs: Additional keyword arguments passed to the plotting function
#             (`gdf.plot` for line/points, `ax.quiver` for channels).

#     Returns:
#         tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)

#     Raises:
#         ImportError: If `use_basemap` is True and `contextily` is not installed.
#         ValueError: If an invalid `plot_style` is provided.
#     """
#     # Check if there's anything to plot
#     has_channels = layout_geo.n_channels > 0
#     has_knots_attr = hasattr(layout_geo, 'knot_locations') and layout_geo.knot_locations is not None and layout_geo.knot_locations.shape[0] > 0
#     has_knots_to_plot = show_knots and has_knots_attr

#     if not has_channels and not has_knots_to_plot:
#         logger.warning("No channels or anchors to plot.", UserWarning)
#         if ax is None:
#             fig, ax = plt.subplots()
#         else:
#             fig = ax.figure
#         ax.set_title("Empty DAS Layout")
#         return fig, ax

#     _basemap_kwargs = basemap_kwargs or {}
#     _knot_kwargs = knot_kwargs or {}
#     _knot_kwargs.setdefault("color", knot_color)
#     _knot_kwargs.setdefault("label", "Knots")
#     _knot_kwargs.setdefault("zorder", 5) # Above basemap and layout
#     _knot_kwargs.setdefault("marker", "P")
#     _knot_kwargs.setdefault("s", 70)

#     # Determine target CRS for plotting
#     target_crs = layout_geo.crs # Default to native CRS
#     if plot_crs:
#         target_crs = pyproj.CRS(plot_crs)

#     # If using basemap, must plot in Web Mercator (EPSG:3857)
#     basemap_crs = None
#     if use_basemap:
#         if not _CONTEXTILY_AVAILABLE:
#             raise ImportError("Please install contextily to use basemaps: pip install contextily")
#         basemap_crs = pyproj.CRS("EPSG:3857")
#         target_crs = basemap_crs # Override target_crs for basemap compatibility

#     # Get GeoDataFrame in the target plotting CRS only if needed for plotting channels/line
#     gdf_plot = None
#     if has_channels:
#         try:
#             gdf_plot = layout_geo.get_gdf(output_crs=target_crs)
#             if gdf_plot.empty:
#                 logger.warning("Layout channel GDF is empty after transformation.", UserWarning)
#                 has_channels = False # Treat as if no channels if GDF is empty
#         except Exception as e:
#              logger.warning(f"Could not transform layout channels to target CRS ({target_crs}): {e}", RuntimeWarning)
#              # Fallback: try plotting in original CRS without basemap
#              target_crs = layout_geo.crs
#              try:
#                  gdf_plot = layout_geo.get_gdf(output_crs=target_crs)
#                  if gdf_plot.empty:
#                      has_channels = False
#              except Exception as e2:
#                  logger.warning(f"Could not transform layout channels to original CRS ({target_crs}) either: {e2}", RuntimeWarning)
#                  has_channels = False # Cannot plot channels

#              use_basemap = False # Disable basemap on fallback
#              logger.warning(f"Falling back to plotting in original CRS ({target_crs}) without basemap.", RuntimeWarning)

#     # Transform knot points if needed
#     x_knots_plot, y_knots_plot = None, None
#     if has_knots_to_plot:
#         try:
#             x_local_knots = layout_geo.knot_locations[:, 0]
#             y_local_knots = layout_geo.knot_locations[:, 1]
#             if layout_geo.crs.is_geographic:
#                 ref_lon, ref_lat = layout_geo.reference_point
#                 local_crs_knots = pyproj.CRS(proj='aeqd', lat_0=ref_lat, lon_0=ref_lon, datum='WGS84', units='m')
#                 transformer_knots = pyproj.Transformer.from_crs(local_crs_knots, target_crs, always_xy=True)
#                 x_knots_plot, y_knots_plot = transformer_knots.transform(x_local_knots, y_local_knots)
#             else: # Projected CRS
#                 ref_x, ref_y = layout_geo.reference_point
#                 x_translated_knots = x_local_knots + ref_x
#                 y_translated_knots = y_local_knots + ref_y
#                 if not layout_geo.crs.equals(target_crs):
#                      transformer_knots = pyproj.Transformer.from_crs(layout_geo.crs, target_crs, always_xy=True)
#                      x_knots_plot, y_knots_plot = transformer_knots.transform(x_translated_knots, y_translated_knots)
#                 else:
#                      x_knots_plot, y_knots_plot = x_translated_knots, y_translated_knots
#         except Exception as e:
#             logger.warning(f"Could not transform anchor coordinates: {e}", RuntimeWarning)
#             has_knots_to_plot = False # Cannot plot knots if transform fails

#     # Check again if anything is plottable
#     if not has_channels and not has_knots_to_plot:
#          logger.warning("No plottable data remains after transformation attempts.", UserWarning)
#          if ax is None:
#              fig, ax = plt.subplots()
#          else:
#              fig = ax.figure
#          ax.set_title(f"Empty DAS Layout (CRS: {target_crs.name if target_crs else 'Unknown'})")
#          return fig, ax

#     # Setup axes
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 10))
#         auto_format = True
#     else:
#         fig = ax.figure
#         auto_format = False

#     plotted_bounds = None # Keep track of bounds for formatting

#     # Plot layout (channels/line)
#     if has_channels and gdf_plot is not None and not gdf_plot.empty:
#         if plot_style == "line":
#             kwargs.setdefault("label", "DAS Cable")
#             kwargs.setdefault("linewidth", 2)
#             try:
#                 # Ensure points are ordered correctly if needed (already should be)
#                 line_geom = LineString(gdf_plot.geometry.tolist())
#                 line_gdf = gpd.GeoDataFrame([1], geometry=[line_geom], crs=gdf_plot.crs)
#                 line_gdf.plot(ax=ax, **kwargs)
#                 plotted_bounds = line_gdf.total_bounds
#             except Exception as e:
#                 logger.warning(f"Could not create or plot LineString: {e}. Plotting points instead.", UserWarning)
#                 point_kwargs = kwargs.copy()
#                 point_kwargs.setdefault("label", "channels")
#                 point_kwargs.setdefault("marker", ".")
#                 point_kwargs.setdefault("s", 10)
#                 point_kwargs.pop("linewidth", None)
#                 gdf_plot.plot(ax=ax, **point_kwargs)
#                 plotted_bounds = gdf_plot.total_bounds

#         elif plot_style == "points":
#             kwargs.setdefault("label", "channels")
#             kwargs.setdefault("marker", ".")
#             kwargs.setdefault("s", 10)
#             gdf_plot.plot(ax=ax, **kwargs)
#             plotted_bounds = gdf_plot.total_bounds

#         elif plot_style == "channels":
#              logger.warning("Plotting channels with direction arrows ('channels' style) in geographic coordinates."
#                            " Arrows represent LOCAL direction and are NOT transformed by the CRS.", UserWarning)
#              point_kwargs = kwargs.copy()
#              point_kwargs.setdefault("label", "channels (Local Direction)")
#              point_kwargs.setdefault("marker", ".")
#              point_kwargs.setdefault("s", 5)
#              point_kwargs.pop("width", None); point_kwargs.pop("scale", None); point_kwargs.pop("scale_units", None)
#              point_kwargs.pop("angles", None); point_kwargs.pop("pivot", None)
#              gdf_plot.plot(ax=ax, **point_kwargs)
#              plotted_bounds = gdf_plot.total_bounds # Get bounds from points

#              x_plot = gdf_plot.geometry.x.values
#              y_plot = gdf_plot.geometry.y.values
#              dx_local = gdf_plot["u_x"].values
#              dy_local = gdf_plot["u_y"].values

#              if layout_geo.channel_spacing is not None and layout_geo.channel_spacing > 0:
#                  avg_dist = layout_geo.channel_spacing
#              elif len(x_plot) > 1:
#                  avg_dist = np.mean(np.sqrt(np.diff(x_plot)**2 + np.diff(y_plot)**2))
#              else:
#                  avg_dist = 1.0
#              scale_factor = avg_dist * 0.5

#              q_kwargs = kwargs.copy()
#              q_kwargs.setdefault("color", point_kwargs.get("color", "blue"))
#              q_kwargs.setdefault("width", 0.003); q_kwargs.setdefault("scale", 1)
#              q_kwargs.setdefault("scale_units", "xy"); q_kwargs.setdefault("angles", "xy")
#              q_kwargs.setdefault("pivot", "middle")
#              q_kwargs.pop("label", None); q_kwargs.pop("marker", None); q_kwargs.pop("s", None); q_kwargs.pop("linewidth", None)
#              ax.quiver(x_plot, y_plot, dx_local * scale_factor, dy_local * scale_factor, **q_kwargs)
#         else:
#             raise ValueError(f"Invalid plot_style: {plot_style}. Choose 'line', 'points', or 'channels'.")

#     # Plot knots if requested and possible
#     if has_knots_to_plot and x_knots_plot is not None:
#         _plot_knots_helper(ax, np.column_stack((x_knots_plot, y_knots_plot)), **_knot_kwargs)
#         # Update bounds if anchors are plotted
#         min_ax, max_ax = np.min(x_knots_plot), np.max(x_knots_plot)
#         min_ay, max_ay = np.min(y_knots_plot), np.max(y_knots_plot)
#         if plotted_bounds is None:
#             plotted_bounds = (min_ax, min_ay, max_ax, max_ay)
#         else:
#             plotted_bounds = (min(plotted_bounds[0], min_ax), min(plotted_bounds[1], min_ay),
#                               max(plotted_bounds[2], max_ax), max(plotted_bounds[3], max_ay))

#     # Add basemap if requested and possible
#     if use_basemap:
#         plot_crs_str = target_crs.to_string() if target_crs else None
#         if plot_crs_str:
#             try:
#                 ctx.add_basemap(ax, crs=plot_crs_str, source=basemap_provider, **_basemap_kwargs)
#             except Exception as e:
#                 logger.warning(f"Failed to add basemap: {e}", RuntimeWarning)
#         else:
#              logger.warning("Cannot add basemap without a valid target CRS.", RuntimeWarning)

#     # Format axes
#     if auto_format:
#         _format_axes_geographic(ax, target_crs, plotted_bounds)

#     return fig, ax

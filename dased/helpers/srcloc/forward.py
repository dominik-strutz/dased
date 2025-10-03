import torch
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable
import geopandas as gpd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

__all__ = ["ForwardBase", "ForwardHomogeneous", "ForwardLayeredLookup"]


class ForwardBase(ABC):
    """
    Base class for seismic forward operators.

    This private base class provides core functionality for calculating travel times
    and signal-to-noise ratios for seismic waves between sources and sensors.
    It handles directional sensitivity calculations and coordinate transformations.

    Arguments:
        wave_type: Type of wave ('P', 'S', 'SV', 'SH', 'R', 'Rayleigh', 'Love').
        distance_relation: Function to calculate distance-based attenuation.
        sensor_type: Type of sensor ('strain', 'displacement', 'strain1C', etc.).
        direction_smoothing: Standard deviation(s) (in radians) for directional
            sensitivity smoothing. If float, the same value is used for theta
            and phi smoothing. If tuple (sigma_theta, sigma_phi), specifies
            smoothing independently. Defaults to 0.0 (no smoothing).
        incidence_max: Maximum allowed phi angle (in radians) for directionality
            calculation. If None, no clipping is applied.
    
    Note:
        This is a base class. Users should use the concrete implementations
        :class:`~dased.helpers.srcloc.forward.ForwardHomogeneous` or
        :class:`~dased.helpers.srcloc.forward.ForwardLayeredLookup` which
        inherit from this class.
    """

    def __init__(
        self,
        wave_type: str,
        distance_relation: Callable,
        sensor_type: str = "strain",
        direction_smoothing: Union[float, Tuple[float, float]] = 0.0,
        incidence_max: Union[float, None] = None,
    ):      
        self.wave_type = wave_type
        self.distance_relation = distance_relation

        # Process direction_smoothing argument
        if isinstance(direction_smoothing, (float, int)):
            self.direction_smoothing_theta = torch.tensor(direction_smoothing)
            self.direction_smoothing_phi = torch.tensor(direction_smoothing)
        elif (
            isinstance(direction_smoothing, (tuple, list))
            and len(direction_smoothing) == 2
        ):
            try:
                self.direction_smoothing_theta = torch.tensor(direction_smoothing[0])
                self.direction_smoothing_phi = torch.tensor(direction_smoothing[1])
            except (ValueError, TypeError):
                raise TypeError(
                    "Elements of direction_smoothing tuple must be numbers."
                )
        else:
            raise TypeError(
                "direction_smoothing must be a float or a tuple/list of two floats."
            )

        if self.direction_smoothing_theta < 0 or self.direction_smoothing_phi < 0:
            raise ValueError(
                "Direction smoothing standard deviations cannot be negative."
            )

        # Map aliases to their full names
        sensor_type_mapping = {"strain": "strain1C", "displacement": "displacement1C"}
        self.sensor_type = sensor_type_mapping.get(sensor_type, sensor_type)

        valid_sensor_types = [
            "strain1C",
            "displacement1C", 
            "strain3C",
            "displacement3C",
        ]
        if self.sensor_type not in valid_sensor_types:
            raise ValueError(
                f"Unknown sensor type: {sensor_type}. Must be one of {valid_sensor_types}"
            )

        self.incidence_max = incidence_max

    def __call__(self, m: torch.Tensor, design: gpd.GeoDataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate travel times and signal-to-noise ratios.

        Arguments:
            m: :class:`torch.Tensor`
                Source coordinate tensor with shape ``(..., 3)`` containing XYZ
                coordinates in meters. Last dimension must be ``[x, y, z]``.
            design: :class:`geopandas.GeoDataFrame`
                GeoDataFrame with sensor locations containing a ``geometry``
                column with Point geometries and an ``elevation`` column with
                elevations in meters. Optionally includes orientation columns
                ``u_x``, ``u_y``, ``u_z``.

        Returns:
            Tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
                Tuple of ``(travel_times, snr)`` with shapes matching the input
                source dimensions plus the number of sensors.
        """
        m_in = self._ensure_batch_dim(m)
        design_coords, sensor_directions = self._extract_design_data(design, m)

        distance, _ = self.calculate_distances(m_in, design_coords)

        # Calculate travel times and wave directions using child class implementations
        t, wave_directions = self.calculate_travel_times_and_directions(
            m_in, design_coords, distance
        )

        # Calculate SNR
        snr = self.snr(distance, wave_directions, sensor_directions)

        # Reshape outputs
        t = t.reshape(m.shape[:-1] + (-1,))
        snr = snr.reshape(m.shape[:-1] + (-1,))

        return t, snr

    def snr(self, distance: torch.Tensor, wave_direction: torch.Tensor, sensor_direction: torch.Tensor) -> torch.Tensor:
        """
        Calculate SNR based on distance and directional sensitivity.

        Arguments:
            distance: :class:`torch.Tensor`
                Distance between source and sensor.
            wave_direction: :class:`torch.Tensor`
                Direction vector of the wave.
            sensor_direction: :class:`torch.Tensor`
                Direction vector of the sensor.

        Returns:
            :class:`torch.Tensor`
                SNR values clamped to maximum of 100.0.
        """
        snr = self.distance_relation(distance)
        
        # Apply directional sensitivity only for 1C strain sensors
        if self.sensor_type.endswith("1C"):
            if self.sensor_type.startswith("strain"):
                snr *= self._directional_sensitivity(wave_direction, sensor_direction)
            # Displacement sensors assumed omnidirectional for now

        return snr.clamp_max(100.0)

    def directional_sensitivity(self, m: torch.Tensor, design: gpd.GeoDataFrame) -> torch.Tensor:
        """
        Calculate directional sensitivity between sources and sensors.

        This is a convenience function that extracts the necessary data from
        the design GeoDataFrame, calculates wave directions, and returns the
        directional sensitivity factor for each source-sensor pair.

        Arguments:
            m: :class:`torch.Tensor`
                Source coordinate tensor with shape ``(..., 3)`` containing XYZ
                coordinates in meters. Last dimension must be ``[x, y, z]``.
            design: :class:`geopandas.GeoDataFrame`
                GeoDataFrame with sensor locations containing a ``geometry``
                column with Point geometries and an ``elevation`` column with
                elevations in meters. Should include orientation columns
                ``u_x``, ``u_y``, ``u_z`` for directional sensors.

        Returns:
            :class:`torch.Tensor`
                Directional sensitivity values with shape matching the input
                source dimensions plus the number of sensors. Values range
                from 0 (no sensitivity) to 1 (maximum sensitivity).
        """
        m_in = self._ensure_batch_dim(m)
        design_coords, sensor_directions = self._extract_design_data(design, m)

        distance, _ = self.calculate_distances(m_in, design_coords)

        # Calculate wave directions using child class implementations
        _, wave_directions = self.calculate_travel_times_and_directions(
            m_in, design_coords, distance
        )

        # Calculate directional sensitivity
        dir_sens = self._directional_sensitivity(wave_directions, sensor_directions)

        # Reshape outputs to match input shape
        dir_sens = dir_sens.reshape(m.shape[:-1] + (-1,))

        return dir_sens

    def calculate_distances(self, m_in: torch.Tensor, design_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate distances between sources and receivers.

        Arguments:
            m_in: :class:`torch.Tensor`
                Source coordinates tensor with shape ``(B, 3)``.
            design_coords: :class:`torch.Tensor`
                Receiver coordinates tensor with shape ``(N, 3)``.

        Returns:
            Tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
                Tuple of ``(total_distance, horizontal_distance)`` with shapes
                ``(B, N)``.
        """
        # Calculate horizontal distance
        horizontal_distance = torch.cdist(m_in[..., :2], design_coords[..., :2])

        # Calculate vertical distance
        vertical_distance = torch.abs(
            m_in[..., 2:3] - design_coords[..., 2:3].transpose(-2, -1)
        )

        # Calculate total distance
        distance = torch.sqrt(horizontal_distance**2 + vertical_distance**2)

        return distance, horizontal_distance

    @abstractmethod
    def calculate_travel_times_and_directions(
        self, 
        m_in: torch.Tensor, 
        design_coords: torch.Tensor, 
        distance: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate travel times and wave directions. Must be implemented by subclasses.

        Arguments:
            m_in: :class:`torch.Tensor`
                Source coordinates tensor.
            design_coords: :class:`torch.Tensor`
                Receiver coordinates tensor.
            distance: Optional[:class:`torch.Tensor`]
                Precalculated distances (optional).

        Returns:
            Tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
                Tuple of ``(travel_times, wave_directions)``.
        """
        pass

    def _ensure_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has batch dimension.

        Parameters
        ----------
        tensor : :class:`torch.Tensor`
            Input tensor, will be returned with a leading batch dimension if needed.
        """
        return tensor if tensor.ndim > 1 else tensor[None, :]

    def _extract_design_data(self, design: gpd.GeoDataFrame, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract coordinates and orientation from GeoDataFrame.

        Arguments:
            design: :class:`geopandas.GeoDataFrame`
                GeoDataFrame with sensor locations.
            m: :class:`torch.Tensor`
                Source coordinate tensor.

        Returns:
            Tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
                Tuple of (design_coords, design_orientation).
        """
        if not isinstance(design, gpd.GeoDataFrame):
            raise TypeError("Design must be a GeoDataFrame")

        # Check if CRS is set - currently not supported
        if design.crs is not None:
            warnings.warn(
                "CRS found in design GeoDataFrame, but transformations are not currently supported. "
                "Assuming coordinates are in the same system as the model."
            )

        # Extract coordinates from GeoDataFrame
        design_coords = torch.tensor(
            [
                [geom.x, geom.y, row.elevation]
                for geom, row in zip(design.geometry, design.itertuples())
            ],
            device=m.device,
            dtype=m.dtype,
        )

        # Get orientation vectors if they exist
        if all(col in design.columns for col in ["u_x", "u_y", "u_z"]):
            # Extract orientation vectors from GeoDataFrame
            design_orientation = torch.tensor(
                [[row.u_x, row.u_y, row.u_z] for row in design.itertuples()],
                device=m.device,
                dtype=m.dtype,
            )
            # Normalize orientation vectors
            norm = torch.norm(design_orientation, dim=1, keepdim=True)
            # Avoid division by zero if norm is zero
            zero_norm_mask = norm < 1e-8
            if torch.any(zero_norm_mask):
                warnings.warn(
                    "Some sensor orientations have zero norm. Setting them to default vertical [0, 0, 1]."
                )
                design_orientation[zero_norm_mask.squeeze()] = torch.tensor(
                    [0.0, 0.0, 1.0], device=m.device, dtype=m.dtype
                )
                norm[zero_norm_mask] = 1.0
            design_orientation = design_orientation / norm

        else:
            # Default vertical orientation if not specified
            warnings.warn(
                "Sensor orientation columns ('u_x', 'u_y', 'u_z') not found in design GeoDataFrame. "
                "Assuming default vertical orientation [0, 0, 1] for all sensors."
            )
            design_orientation = torch.zeros_like(design_coords)
            design_orientation[:, 2] = 1.0  # Z-direction

        return design_coords, design_orientation

    def _directional_sensitivity(self, wave_direction: torch.Tensor, sensor_direction: torch.Tensor) -> torch.Tensor:
        """
        Calculate radiation pattern based on wave type and directions.

        Arguments:
            wave_direction: Direction vector of the wave (..., 3).
            sensor_direction: Direction vector of the sensor (N, 3) or (..., N, 3).

        Returns:
            Directional sensitivity factor tensor.
        """
        # Ensure vectors have correct dimensions for broadcasting
        wave_dir = (
            wave_direction.unsqueeze(-2)
            if wave_direction.ndim == sensor_direction.ndim
            else wave_direction
        )
        sensor_dir = (
            sensor_direction.unsqueeze(0)
            if wave_direction.ndim > sensor_direction.ndim
            else sensor_direction
        )

        # Normalize vectors (handle potential zero norms)
        wave_norm = torch.norm(wave_dir, dim=-1, keepdim=True)
        sensor_norm = torch.norm(sensor_dir, dim=-1, keepdim=True)

        # Avoid division by zero
        wave_dir = torch.where(
            wave_norm > 1e-8, wave_dir / wave_norm, torch.zeros_like(wave_dir)
        )
        sensor_dir = torch.where(
            sensor_norm > 1e-8, sensor_dir / sensor_norm, torch.zeros_like(sensor_dir)
        )
        # Set zero norm sensor direction to vertical to avoid issues in angle calculation
        sensor_dir = torch.where(
            sensor_norm <= 1e-8,
            torch.tensor(
                [0.0, 0.0, 1.0], device=sensor_dir.device, dtype=sensor_dir.dtype
            ),
            sensor_dir,
        )

        # Calculate angles in spherical coordinates
        theta_wave = torch.atan2(wave_dir[..., 1], wave_dir[..., 0])
        theta_fiber = torch.atan2(sensor_dir[..., 1], sensor_dir[..., 0])

        phi_wave = torch.atan2(
            wave_dir[..., 2],
            torch.sqrt(wave_dir[..., 0] ** 2 + wave_dir[..., 1] ** 2 + 1e-15),
        )
        phi_fiber = torch.atan2(
            sensor_dir[..., 2],
            torch.sqrt(sensor_dir[..., 0] ** 2 + sensor_dir[..., 1] ** 2 + 1e-15),
        )

        # Apply incidence_max clipping if set
        if self.incidence_max is not None:
            phi_wave = torch.clamp(phi_wave, max=self.incidence_max)
            phi_fiber = torch.clamp(phi_fiber, max=self.incidence_max)

        return self._directional_sensitivity_factor(
            theta_wave, theta_fiber, phi_wave, phi_fiber
        )

    def _smoothed_cos_sq(self, angle_diff: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Calculate smoothed cos^2(angle_diff) using expectation over noise."""
        if sigma == 0.0:
            return torch.cos(angle_diff) ** 2
        else:
            sigma_sq = sigma**2
            return 0.5 * (1 + torch.exp(-2.0 * sigma_sq) * torch.cos(2 * angle_diff))

    def _smoothed_sin_2x(self, angle_diff: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Calculate smoothed sin(2*angle_diff) using expectation over noise."""
        if sigma == 0.0:
            return torch.sin(2 * angle_diff)
        else:
            sigma_sq = sigma**2
            return torch.exp(-2.0 * sigma_sq) * torch.sin(2 * angle_diff)

    def _directional_sensitivity_factor(
        self, 
        theta_wave: torch.Tensor, 
        theta_fiber: torch.Tensor, 
        phi_wave: torch.Tensor, 
        phi_fiber: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the directional sensitivity factor based on angles and wave type.

        Arguments:
            theta_wave: Horizontal angle of wave direction (radians).
            theta_fiber: Horizontal angle of fiber direction (radians).
            phi_wave: Vertical angle (elevation) of wave direction (radians).
            phi_fiber: Vertical angle (elevation) of fiber direction (radians).

        Returns:
            Directional sensitivity factor.
        """
        # Ensure inputs are tensors
        device = getattr(self.distance_relation, "device", "cpu")
        theta_wave = torch.as_tensor(theta_wave, device=device)
        theta_fiber = torch.as_tensor(theta_fiber, device=theta_wave.device)
        phi_wave = torch.as_tensor(phi_wave, device=theta_wave.device)
        phi_fiber = torch.as_tensor(phi_fiber, device=theta_wave.device)

        # Apply incidence_max clipping if set
        if self.incidence_max is not None:
            phi_wave = torch.clamp(phi_wave, max=self.incidence_max)
            phi_fiber = torch.clamp(phi_fiber, max=self.incidence_max)

        # Angle differences
        delta_theta = theta_wave - theta_fiber
        delta_phi = phi_wave - phi_fiber

        # Apply smoothing based on wave type
        sigma_theta = self.direction_smoothing_theta
        sigma_phi = self.direction_smoothing_phi

        if self.wave_type in ["P"]:
            # factor = cos^2(delta_theta) * cos^2(delta_phi)
            cos_sq_theta_smoothed = self._smoothed_cos_sq(delta_theta, sigma_theta)
            cos_sq_phi_smoothed = self._smoothed_cos_sq(delta_phi, sigma_phi)
            factor = cos_sq_theta_smoothed * cos_sq_phi_smoothed
        elif self.wave_type in ["SV"]:
            # factor = cos^2(delta_theta) * sin(2 * delta_phi)
            cos_sq_theta_smoothed = self._smoothed_cos_sq(delta_theta, sigma_theta)
            sin_2phi_smoothed = self._smoothed_sin_2x(delta_phi, sigma_phi)
            factor = cos_sq_theta_smoothed * sin_2phi_smoothed
        elif self.wave_type in ["SH"]:
            # factor = sin(2 * delta_theta) * cos(delta_phi)
            sin_2theta_smoothed = self._smoothed_sin_2x(delta_theta, sigma_theta)
            if sigma_phi == 0.0:
                cos_phi_smoothed = torch.cos(delta_phi)
            else:
                cos_phi_smoothed = torch.exp(-0.5 * sigma_phi**2) * torch.cos(delta_phi)
            factor = sin_2theta_smoothed * cos_phi_smoothed
        elif self.wave_type in ["S"]:
            # Calculate smoothed SV and SH components
            cos_sq_theta_smoothed = self._smoothed_cos_sq(delta_theta, sigma_theta)
            sin_2phi_smoothed = self._smoothed_sin_2x(delta_phi, sigma_phi)
            SV_factor_smoothed = cos_sq_theta_smoothed * sin_2phi_smoothed

            sin_2theta_smoothed = self._smoothed_sin_2x(delta_theta, sigma_theta)
            if sigma_phi == 0.0:
                cos_phi_smoothed = torch.cos(delta_phi)
            else:
                cos_phi_smoothed = torch.exp(-0.5 * sigma_phi**2) * torch.cos(delta_phi)
            SH_factor_smoothed = sin_2theta_smoothed * cos_phi_smoothed

            # Combine SV and SH factors
            factor = torch.max(
                torch.max(torch.abs(SV_factor_smoothed), torch.abs(SH_factor_smoothed)),
                torch.sqrt(SV_factor_smoothed**2 + SH_factor_smoothed**2),
            )

        else:
            raise ValueError(
                f"Unknown wave type for directional sensitivity: {self.wave_type}"
            )

        return torch.abs(factor)


class ForwardHomogeneous(ForwardBase):
    """
    Calculates travel times and SNRs assuming a constant velocity medium.
    This is the simplest forward model, suitable for preliminary analysis
    or when detailed earth structure is unknown.

    Arguments:
        velocity: Wave velocity in the medium (m/s). Should be positive.
            
        wave_type: Type of wave ('P', 'S', 'SV', 'SH', etc.). Affects
            directional sensitivity calculations for strain sensors.
            
        distance_relation: Function to calculate distance-based attenuation.
            Should accept distance in meters and return amplitude/SNR values.
            
        sensor_type: Type of sensor ('strain', 'displacement', 'strain1C', etc.).
            Defaults to 'strain' which maps to 'strain1C'.
            
        direction_smoothing: Standard deviation(s) for directional sensitivity
            smoothing in radians. Can be float or tuple (sigma_theta, sigma_phi).
            Defaults to 0.0 (no smoothing).
            
        incidence_max: Maximum allowed phi angle (in radians) for directionality
            calculation. If None, no clipping is applied.

    Examples:
        Basic P-wave forward model::

            >>> from dased.helpers.srcloc import ForwardHomogeneous, MagnitudeRelation
            >>> # Setup magnitude relation for SNR calculation
            >>> mag_rel = MagnitudeRelation(magnitude_factor=1.0, reference_distance=1000)  # :class:`~dased.helpers.srcloc.magnitude_relation.MagnitudeRelation`
            >>> # Create forward model with 5 km/s P-wave velocity
            >>> forward = ForwardHomogeneous(
            ...     velocity=5000.0,
            ...     wave_type='P',
            ...     distance_relation=mag_rel
            ... )
            >>> # Calculate travel times and SNR
            >>> source = torch.tensor([[1000.0, 2000.0, 500.0]])  # x, y, z in meters, :class:`torch.Tensor`
            >>> times, snr = forward(source, design_gdf)  # returns (:class:`torch.Tensor`, :class:`torch.Tensor`)
    """

    def __init__(
        self,
        velocity: float,
        wave_type: str,
        distance_relation: Callable,
        sensor_type: str = "strain",
        direction_smoothing: Union[float, Tuple[float, float]] = 0.0,
        incidence_max: Union[float, None] = None,
    ):
        super().__init__(wave_type, distance_relation, sensor_type, direction_smoothing, incidence_max)
        
        if velocity <= 0:
            raise ValueError("Velocity must be positive")
            
        self.velocity = velocity
        # Store velocity on the same device as distance_relation if possible
        if hasattr(distance_relation, "device"):
            self.velocity = torch.tensor(velocity, device=distance_relation.device)

    def calculate_travel_times_and_directions(
        self, 
        m_in: torch.Tensor, 
        design_coords: torch.Tensor, 
        distance: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate travel times and wave directions in a homogeneous medium.

        Arguments:
            m_in: Source coordinates tensor (B, 3) or (1, 3).
            design_coords: Receiver coordinates tensor (N, 3).
            distance: Precalculated distances (optional) (B, N) or (1, N).

        Returns:
            Tuple of (travel_times, wave_directions) with shapes (B, N) and (B, N, 3).
        """
        if distance is None:
            m_expanded = m_in.unsqueeze(1) if m_in.ndim == 2 else m_in
            design_expanded = design_coords.unsqueeze(0)
            distance = torch.norm(m_expanded - design_expanded, dim=-1)

        traveltimes = distance / self.velocity

        wave_directions = design_coords.unsqueeze(0) - m_in.unsqueeze(1)

        norm = torch.norm(wave_directions, dim=-1, keepdim=True)
        wave_directions = torch.where(
            norm > 1e-8, wave_directions / norm, torch.zeros_like(wave_directions)
        )

        return traveltimes, wave_directions


class ForwardLayeredLookup(ForwardBase):
    """
    Uses precomputed travel times and incidence angles from lookup tables
    generated by ray tracing through layered earth models. This provides
    more accurate modeling for realistic earth structure.

    Arguments:
        lookup_path: Path to the lookup table file in NetCDF format.
            The file must contain coordinates 'receiver_depth', 'distance',
            'source_depth' and data variables 'arrival_time', 'incidence_angle'.
            
        wave_type: Type of wave ('P', 'S', 'SV', 'SH', etc.).
        
        distance_relation: Function to calculate distance-based attenuation.
        
        sensor_type: Type of sensor. Defaults to 'strain'.
        
        direction_smoothing: Standard deviation(s) for directional sensitivity
            smoothing. Defaults to 0.0.
            
        incidence_max: Maximum allowed phi angle (in radians) for directionality
            calculation. If None, no clipping is applied.

    Raises:
        FileNotFoundError: If lookup table file is not found.
        ValueError: If lookup table is missing required coordinates or variables.
        IOError: If lookup table cannot be loaded or initialized.

    Examples:
        Using a precomputed lookup table::
        
            >>> forward = ForwardLayeredLookup(
            ...     lookup_path='travel_times_P.nc',
            ...     wave_type='P',
            ...     distance_relation=mag_rel  # :class:`~dased.helpers.srcloc.magnitude_relation.MagnitudeRelation`
            ... )
            >>> times, snr = forward(source_locations, design_gdf)  # returns (:class:`torch.Tensor`, :class:`torch.Tensor`)

    Note:
        The lookup table assumes depths are positive downward from the surface.
        Coordinates in the lookup table should use the same units as the
        source and receiver coordinates (typically meters).
    """

    def __init__(
        self,
        lookup_path: str,
        wave_type: str,
        distance_relation: Callable,
        sensor_type: str = "strain",
        direction_smoothing: Union[float, Tuple[float, float]] = 0.0,
        incidence_max: Union[float, None] = None,
    ):
        super().__init__(wave_type, distance_relation, sensor_type, direction_smoothing, incidence_max)

        try:
            ds = xr.open_dataset(lookup_path)
            self._init_interpolators(ds)
            print(f"Loaded lookup table from {lookup_path}")
            self.device = getattr(self.distance_relation, "device", "cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"Lookup table not found at {lookup_path}")
        except Exception as e:
            raise IOError(f"Failed to load or initialize from lookup table at {lookup_path}: {e}")

    def _init_interpolators(self, ds: xr.Dataset) -> None:
        """
        Initialize interpolation functions from dataset.

        Arguments:
            ds: xarray Dataset containing lookup tables.
        """
        required_coords = ["receiver_depth", "distance", "source_depth"]
        required_vars = ["arrival_time", "incidence_angle"]
        
        if not all(coord in ds.coords for coord in required_coords):
            raise ValueError(
                f"Lookup table missing required coordinates. Expected: {required_coords}, "
                f"Found: {list(ds.coords)}"
            )
        if not all(var in ds.data_vars for var in required_vars):
            raise ValueError(
                f"Lookup table missing required data variables. Expected: {required_vars}, "
                f"Found: {list(ds.data_vars)}"
            )

        self.receiver_depths = ds.receiver_depth.values
        self.distances = ds.distance.values
        self.source_depths = ds.source_depth.values

        # Check coordinate ordering
        if not np.all(np.diff(self.receiver_depths) > 0):
            warnings.warn("Lookup table receiver_depths are not strictly ascending.")
        if not np.all(np.diff(self.distances) >= 0):
            warnings.warn("Lookup table distances are not non-decreasing.")
        if not np.all(np.diff(self.source_depths) > 0):
            warnings.warn("Lookup table source_depths are not strictly ascending.")

        coords = (self.receiver_depths, self.distances, self.source_depths)

        # Check for NaNs
        if np.isnan(ds.arrival_time.values).any():
            warnings.warn(
                "NaNs found in lookup table 'arrival_time'. Interpolation might produce NaNs."
            )
        if np.isnan(ds.incidence_angle.values).any():
            warnings.warn(
                "NaNs found in lookup table 'incidence_angle'. Interpolation might produce NaNs."
            )

        self.arrival_time_interp = RegularGridInterpolator(
            coords,
            ds.arrival_time.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        self.incidence_angle_interp = RegularGridInterpolator(
            coords,
            ds.incidence_angle.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def calculate_travel_times_and_directions(
        self, 
        m_in: torch.Tensor, 
        design_coords: torch.Tensor, 
        distance: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate travel times and wave directions using lookup tables.

        Arguments:
            m_in: Source coordinates tensor (B, 3) or (1, 3).
            design_coords: Receiver coordinates tensor (N, 3).
            distance: Precalculated total distances (optional) (B, N) or (1, N).

        Returns:
            Tuple of (travel_times, wave_directions) with shapes (B, N) and (B, N, 3).
        """
        n_sources = m_in.shape[0]
        n_receivers = design_coords.shape[0]

        horizontal_distance = torch.cdist(m_in[..., :2], design_coords[..., :2])

        # Convert to positive depths for lookup
        source_depth_lookup = -m_in[:, 2:3]
        receiver_depth_lookup = -design_coords[:, 2:3].T

        source_depth_grid = source_depth_lookup.expand(-1, n_receivers)
        receiver_depth_grid = receiver_depth_lookup.expand(n_sources, -1)
        horizontal_distance_grid = horizontal_distance

        points = (
            torch.stack(
                [
                    receiver_depth_grid.flatten(),
                    horizontal_distance_grid.flatten(),
                    source_depth_grid.flatten(),
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        try:
            interpolated_times = self.arrival_time_interp(points)
            interpolated_angles_deg = self.incidence_angle_interp(points)
        except ValueError as e:
            raise ValueError(
                f"Interpolation failed. Check if query points are outside lookup table bounds "
                f"or if NaNs exist in lookup data. Original error: {e}"
            )

        if np.isnan(interpolated_times).any() or np.isnan(interpolated_angles_deg).any():
            warnings.warn(
                "NaNs encountered during interpolation. This might be due to points outside "
                "the lookup table bounds or NaNs in the table data. Replacing NaNs with "
                "default values (time=inf, angle=90)."
            )
            interpolated_times = np.nan_to_num(interpolated_times, nan=np.inf)
            interpolated_angles_deg = np.nan_to_num(interpolated_angles_deg, nan=90.0)

        traveltimes = (
            torch.from_numpy(interpolated_times)
            .reshape(n_sources, n_receivers)
            .to(self.device, dtype=m_in.dtype)
        )
        incidence_angles_rad = (
            torch.deg2rad(torch.from_numpy(interpolated_angles_deg))
            .reshape(n_sources, n_receivers)
            .to(self.device, dtype=m_in.dtype)
        )

        # Apply incidence_max clipping if set
        if self.incidence_max is not None:
            incidence_angles_rad = torch.clamp(incidence_angles_rad, max=self.incidence_max)

        # Calculate azimuths for wave direction
        dx = design_coords[:, 0].unsqueeze(0) - m_in[:, 0:1]
        dy = design_coords[:, 1].unsqueeze(0) - m_in[:, 1:2]
        azimuths = torch.atan2(dy, dx)

        # Build wave direction vectors
        wave_directions = torch.zeros(
            (n_sources, n_receivers, 3), device=self.device, dtype=m_in.dtype
        )
        sin_inc = torch.sin(incidence_angles_rad)
        cos_inc = torch.cos(incidence_angles_rad)

        wave_directions[..., 0] = torch.cos(azimuths) * sin_inc
        wave_directions[..., 1] = torch.sin(azimuths) * sin_inc
        wave_directions[..., 2] = cos_inc

        # Normalize direction vectors
        norm = torch.norm(wave_directions, dim=-1, keepdim=True)
        wave_directions = torch.where(
            norm > 1e-8, wave_directions / norm, torch.zeros_like(wave_directions)
        )

        return traveltimes, wave_directions

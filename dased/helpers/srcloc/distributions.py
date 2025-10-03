import torch
import torch.distributions as dist
import xarray as xr
import numpy as np
from torch import Tensor, Size
from typing import Union
from shapely.geometry import Polygon, Point

__all__ = ["SurfaceField_Distribution", "PolygonUniform", "IndependentNormal"]

class WeightedMultivariateNormal:
    """
    Thin wrapper around MultivariateNormal that removes weighted mean before evaluating log_prob.
    The weights are derived from the precision matrix (inverse of covariance).
    """
    
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        # Store the original loc for weighted mean calculation
        self._original_loc = loc
        
        # Calculate weights from precision matrix
        if precision_matrix is not None:
            self._precision_matrix = precision_matrix
        elif scale_tril is not None:
            # Calculate precision matrix from scale_tril
            batch_size = scale_tril.shape[0]
            dim = scale_tril.shape[-1]
            identity = torch.eye(dim, device=scale_tril.device).expand(batch_size, dim, dim)
            self._precision_matrix = torch.linalg.solve_triangular(
                scale_tril, 
                torch.linalg.solve_triangular(
                    scale_tril.transpose(-1, -2), 
                    identity,
                    upper=True
                ),
                upper=False
            )
        elif covariance_matrix is not None:
            self._precision_matrix = torch.linalg.inv(covariance_matrix)
        else:
            raise ValueError("Must provide either covariance_matrix, precision_matrix, or scale_tril")
        
        # Calculate weights (sum of precision matrix rows, normalized)
        weights = torch.sum(self._precision_matrix, dim=-1)
        self._weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        
        # Calculate weighted mean and subtract from loc
        weighted_mean = torch.sum(loc * self._weights, dim=-1, keepdim=True)
        adjusted_loc = loc - weighted_mean
        
        # Create the underlying distribution with adjusted loc
        self._dist = dist.MultivariateNormal(adjusted_loc, covariance_matrix=covariance_matrix, 
                                           precision_matrix=precision_matrix, scale_tril=scale_tril, 
                                           validate_args=validate_args)
    
    def log_prob(self, value):
        # Calculate weighted mean of the input value and subtract it
        weighted_mean = torch.sum(value * self._weights, dim=-1, keepdim=True)
        adjusted_value = value - weighted_mean
        return self._dist.log_prob(adjusted_value)
    
    def sample(self, sample_shape=torch.Size()):
        # Sample from the adjusted distribution and add back the weighted mean of original loc
        samples = self._dist.sample(sample_shape)
        original_weighted_mean = torch.sum(self._original_loc * self._weights, dim=-1, keepdim=True)
        return samples + original_weighted_mean
    
    def expand(self, batch_shape, _instance=None):
        # Expand the underlying distribution and create a new wrapper with expanded weights
        expanded_dist = self._dist.expand(batch_shape, _instance)
        
        # Create new instance with expanded parameters
        new_instance = WeightedMultivariateNormal.__new__(WeightedMultivariateNormal)
        new_instance._dist = expanded_dist
        
        # Expand weights and original_loc to match the new batch shape
        expanded_batch_shape = torch.Size(batch_shape)
        original_batch_shape = self._weights.shape[:-1]
        
        # Calculate the expansion dimensions
        expand_dims = expanded_batch_shape + self._weights.shape[len(original_batch_shape):]
        expand_dims_loc = expanded_batch_shape + self._original_loc.shape[len(original_batch_shape):]
        
        new_instance._weights = self._weights.expand(expand_dims)
        new_instance._original_loc = self._original_loc.expand(expand_dims_loc)
        new_instance._precision_matrix = self._precision_matrix.expand(
            expanded_batch_shape + self._precision_matrix.shape[len(original_batch_shape):])
        
        return new_instance
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to the underlying distribution
        return getattr(self._dist, name)


class WeightedIndependentNormal:
    """
    Thin wrapper around Independent Normal that removes weighted mean before evaluating log_prob.
    The weights are derived from the inverse variance (precision).
    """
    
    def __init__(self, loc, scale, validate_args=None):
        # Store the original loc for weighted mean calculation
        self._original_loc = loc
        
        # Calculate weights from precision (inverse variance squared)
        precision = scale.reciprocal() ** 2
        self._weights = precision / torch.sum(precision, dim=-1, keepdim=True)
        
        # Calculate weighted mean and subtract from loc
        weighted_mean = torch.sum(loc * self._weights, dim=-1, keepdim=True)
        adjusted_loc = loc - weighted_mean
        
        # Create the underlying distribution with adjusted loc
        base_dist = dist.Normal(adjusted_loc, scale, validate_args=validate_args)
        self._dist = dist.Independent(base_dist, 1, validate_args=validate_args)
    
    def log_prob(self, value):
        # Calculate weighted mean of the input value and subtract it
        weighted_mean = torch.sum(value * self._weights, dim=-1, keepdim=True)
        adjusted_value = value - weighted_mean
        return self._dist.log_prob(adjusted_value)
    
    def sample(self, sample_shape=torch.Size()):
        # Sample from the adjusted distribution and add back the weighted mean of original loc
        samples = self._dist.sample(sample_shape)
        original_weighted_mean = torch.sum(self._original_loc * self._weights, dim=-1, keepdim=True)
        return samples + original_weighted_mean
    
    def expand(self, batch_shape, _instance=None):
        # Expand the underlying distribution and create a new wrapper with expanded weights
        expanded_dist = self._dist.expand(batch_shape, _instance)
        
        # Create new instance with expanded parameters
        new_instance = WeightedIndependentNormal.__new__(WeightedIndependentNormal)
        new_instance._dist = expanded_dist
        
        # Expand weights and original_loc to match the new batch shape
        expanded_batch_shape = torch.Size(batch_shape)
        original_batch_shape = self._weights.shape[:-1]
        
        # Calculate the expansion dimensions
        expand_dims = expanded_batch_shape + self._weights.shape[len(original_batch_shape):]
        expand_dims_loc = expanded_batch_shape + self._original_loc.shape[len(original_batch_shape):]
        
        new_instance._weights = self._weights.expand(expand_dims)
        new_instance._original_loc = self._original_loc.expand(expand_dims_loc)
        
        return new_instance
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to the underlying distribution
        return getattr(self._dist, name)

class IndependentNormal(dist.Distribution):
    """
    Independent Normal distribution with element-wise independence.
    
    This distribution represents a multivariate normal distribution where
    each component is independent (diagonal covariance matrix). It's more
    efficient than a full multivariate normal when correlations are not needed.
    
    Arguments:
        loc: Mean of the normal distribution with shape (..., n).
        scale: Standard deviation of the normal distribution with shape (..., n).
        
    Examples:
        Create independent normal distribution::
        
            >>> loc = torch.tensor([0.0, 1.0, -0.5])  # :class:`torch.Tensor`
            >>> scale = torch.tensor([1.0, 2.0, 0.5])  # :class:`torch.Tensor`
            >>> dist = IndependentNormal(loc, scale)
            >>> samples = dist.sample((100,))  # Sample 100 points
    """

    arg_constraints = {'loc': dist.constraints.real, 'scale': dist.constraints.positive}

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        super(IndependentNormal, self).__init__(
            batch_shape=loc.shape[:-1], 
            event_shape=loc.shape[-1:]
        )
        self.base_dist = dist.Normal(loc, scale)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability density."""
        return self.base_dist.log_prob(value).sum(dim=-1)
    
    def sample(self, sample_shape: Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution."""
        return self.base_dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sampling for gradient flow."""
        return self.base_dist.rsample(sample_shape)
    
    def expand(self, batch_shape: Size, _instance=None) -> "IndependentNormal":
        """Returns a new distribution with expanded batch dimensions."""
        batch_shape = torch.Size(batch_shape)
        loc = self.loc.expand(batch_shape + self.event_shape)
        scale = self.scale.expand(batch_shape + self.event_shape)
        return IndependentNormal(loc, scale)

    @property
    def mean(self) -> torch.Tensor:
        """Compute the mean of the distribution."""
        return self.loc
    
    @property
    def variance(self) -> torch.Tensor:
        """Compute the variance of the distribution."""
        return self.scale ** 2

    @property
    def stddev(self) -> torch.Tensor:
        """Compute the standard deviation of the distribution."""
        return self.scale


def _get_elevation(points: torch.Tensor, topo_data: xr.DataArray) -> torch.Tensor:
    """
    Extract elevation values from topography data at given points.
    
    Arguments:
        points: Tensor of (x, y) coordinates with shape (..., 2).
        topo_data: :class:`~xarray.DataArray` with topography data.

    Returns:
        Tensor of elevation values as a :class:`torch.Tensor`.
    """
    x = xr.DataArray(points[..., 0], dims='points')
    y = xr.DataArray(points[..., 1], dims='points')
    elevations = topo_data.interp(x=x, y=y, method='linear').values
    return torch.from_numpy(elevations)


class SurfaceField_Distribution(dist.Distribution):
    """
    3D spatial distribution that follows surface topography with depth variation.
    
    This distribution generates 3D points where the horizontal coordinates follow
    a specified 2D distribution and the vertical coordinate is distributed 
    between the surface topography and a maximum depth below the surface.
    
    Arguments:
        distribution: 2D probability distribution for horizontal coordinates.
            Should be a PyTorch distribution that samples (x, y) pairs.
            
        topo_data: :class:`xarray.DataArray` containing topography data with
            ``'x'`` and ``'y'`` coordinates and elevation values. Used for
            elevation at sampled horizontal coordinates.
            
        depth: Maximum depth below surface in meters. Depth is sampled uniformly
            between the surface and this maximum depth. Must be positive.
            Defaults to 200 meters.
    
    Examples:
        Create distribution over mountainous terrain::

            >>> import torch.distributions as dist
            >>> # Define horizontal distribution
            >>> horizontal_dist = dist.Uniform(
            ...     torch.tensor([0.0, 0.0]), 
            ...     torch.tensor([10000.0, 10000.0])
            ... )
            >>> # Assume topo_data is loaded as :class:`~xarray.DataArray`
            >>> spatial_dist = SurfaceField_Distribution(
            ...     distribution=horizontal_dist,
            ...     topo_data=topo_data,
            ...     depth=500.0  # Up to 500m below surface
            ... )
            >>> samples = spatial_dist.sample((1000,))  # Sample 1000 3D points
    """
    #TODO: maybe a better name
    
    def __init__(
        self, 
        distribution: dist.Distribution, 
        topo_data: xr.DataArray, 
        depth: float = 200.0
    ):
        if depth <= 0:
            raise ValueError("Depth must be positive")
            
        self.hor_distribution = distribution
        self.topo_data = topo_data
        self.depth = float(depth)
        
        # Define bounds with buffer to avoid interpolation edge effects
        buffer = 60.0  # meters
        #TODO: make this relative, also why is it needed in the first place?
        self.bounds = torch.tensor([
            [self.topo_data['x'].values.min() + buffer, self.topo_data['y'].values.min() + buffer],
            [self.topo_data['x'].values.max() - buffer, self.topo_data['y'].values.max() - buffer]
        ])
        
        # Vertical distribution: uniform from surface to max depth below
        self.vert_dist = dist.Uniform(-1, self.depth)
        
        # Distribution properties
        super(SurfaceField_Distribution, self).__init__(
            batch_shape=distribution.batch_shape,
            event_shape=torch.Size((3,))
        )
        
    def sample(self, sample_shape: Union[int, tuple, Size] = 1) -> Tensor:
        """
        Sample 3D points following surface topography.
        
        Arguments:
            sample_shape: Shape of samples to generate. Can be int, tuple, or Size.
            
        Returns:
            Tensor of 3D points with shape sample_shape + (3,).
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_shape, (tuple, list)):
            sample_shape = torch.Size(sample_shape)
            
        n = torch.prod(torch.tensor(sample_shape))
        
        # Sample horizontal coordinates
        rand_points = self.hor_distribution.sample((n,))

        # Clamp to bounds to stay within topography data
        rand_points = torch.clamp(rand_points, self.bounds[0], self.bounds[1])

        # Get surface elevations at sampled points
        elevations = _get_elevation(rand_points, self.topo_data)
        
        # Sample depths uniformly from 0 to max depth
        depth_samples = torch.rand(n) * self.depth
        
        # Calculate final z coordinates (elevation - depth)
        z_coords = elevations - depth_samples
        
        # Combine horizontal and vertical coordinates
        rand_points = torch.cat((rand_points, z_coords[:, None]), dim=1)        
        rand_points = rand_points.reshape(sample_shape + (3,))
        
        # Additional clipping to bounds with small offset
        offset_from_bounds = 0.01 * (self.bounds[1] - self.bounds[0])
        rand_points[..., 0] = torch.clamp(
            rand_points[..., 0], 
            self.topo_data['x'].values.min() + offset_from_bounds[0],
            self.topo_data['x'].values.max() - offset_from_bounds[0]
        )
        rand_points[..., 1] = torch.clamp(
            rand_points[..., 1], 
            self.topo_data['y'].values.min() + offset_from_bounds[1],
            self.topo_data['y'].values.max() - offset_from_bounds[1]
        )

        return rand_points.float()
    
    def log_prob(self, value: Tensor, fast_eval: bool = True) -> Tensor:
        """
        Calculate log probability density for given points.
        
        Arguments:
            value: Points to evaluate with shape (..., 3).
            fast_eval: If True, uses approximate elevation for speed. If False,
                interpolates actual elevation for each point (slower but more accurate).
                
        Returns:
            Log probability densities with shape (...,).
        """
        if value.ndim == 1:
            value = value[None, ...]
            
        # Log probability of horizontal coordinates
        log_prob_hori = self.hor_distribution.log_prob(value[..., :2])        

        if not fast_eval:
            # Exact evaluation: interpolate elevation for each point
            elevation = _get_elevation(value[..., :2].detach(), self.topo_data)
            log_prob_vert = self.vert_dist.log_prob(elevation - value[..., 2])
        else:
            # Fast evaluation: assume points are at mid-depth
            log_prob_vert = self.vert_dist.log_prob(0.5 * self.depth)
        
        return log_prob_hori + log_prob_vert


class PolygonUniform(dist.Distribution):
    """
    Uniform distribution within a Shapely polygon with custom vertical distribution.
    
    This distribution generates 3D points where the horizontal coordinates are
    uniformly distributed within a specified polygon and the vertical coordinate
    follows a given distribution.
    
    Arguments:
        polygon: Shapely Polygon object defining the horizontal bounds.
        vertical_dist: PyTorch distribution for the vertical (z) component.
            Should be a 1D distribution.
    
    Examples:
        Create uniform distribution within rectangular area::

            >>> from shapely.geometry import Polygon
            >>> import torch.distributions as dist
            >>> # Define rectangular polygon
            >>> coords = [(0, 0), (1000, 0), (1000, 500), (0, 500), (0, 0)]
            >>> polygon = Polygon(coords)
            >>> # Define vertical distribution (uniform from 0 to 1000m depth)
            >>> vertical_dist = dist.Uniform(0, 1000)
            >>> # Create 3D distribution
            >>> spatial_dist = PolygonUniform(polygon, vertical_dist)
            >>> samples = spatial_dist.sample((500,))  # :class:`torch.Tensor` of shape (500, 3)
            
    Note:
        This distribution uses rejection sampling for horizontal coordinates,
        which may be slow for polygons with low area-to-bounding-box ratios.
        The implementation includes adaptive oversampling to improve efficiency.
    """

    arg_constraints = {}
    has_rsample = False
    has_enumerate_support = False

    def __init__(self, polygon: Polygon, vertical_dist: dist.Distribution):
        if not isinstance(polygon, Polygon):
            raise TypeError("polygon must be a Shapely Polygon object")
            
        self.polygon = polygon
        self.vertical_dist = vertical_dist
        
        # Get the bounds of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        self.bounds = torch.tensor([[minx, miny], [maxx, maxy]])
        
        # Calculate the area of the polygon for log_prob
        self.polygon_area = torch.tensor(float(polygon.area))
        
        if self.polygon_area <= 0:
            raise ValueError("Polygon must have positive area")
        
        # Determine batch_shape and event_shape
        _batch_shape = self.vertical_dist.batch_shape
        _event_shape = torch.Size((3,))

        super(PolygonUniform, self).__init__(
            batch_shape=_batch_shape,
            event_shape=_event_shape
        )
    
    def _points_in_polygon(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the polygon using Shapely.
        
        Arguments:
            points: Tensor of (x, y) coordinates with shape (..., 2).
            
        Returns:
            Boolean tensor indicating which points are inside.
        """
        points_np = points.detach().cpu().numpy()
        original_shape = points.shape[:-1]
        points_np = points_np.reshape(-1, 2)
        
        inside = np.zeros(points_np.shape[0], dtype=bool)
        for i in range(points_np.shape[0]):
            # Ensure points are valid before checking containment
            if np.all(np.isfinite(points_np[i])):
                try:
                    inside[i] = self.polygon.contains(Point(points_np[i, 0], points_np[i, 1]))
                except Exception:
                    inside[i] = False
            else:
                inside[i] = False

        return torch.tensor(inside, device=points.device).reshape(original_shape)

    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        """
        Sample points uniformly from within the polygon.
        
        Arguments:
            sample_shape: Shape of samples to generate.
            
        Returns:
            Tensor of 3D points with shape sample_shape + batch_shape + (3,).
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_shape, (tuple, list)):
            sample_shape = torch.Size(sample_shape)
        
        # Combine sample_shape with batch_shape
        combined_shape = sample_shape + self.batch_shape
        n_total = torch.prod(torch.tensor(combined_shape)).item()
        
        # Calculate oversampling based on area ratio
        bounding_box_area = (self.bounds[1, 0] - self.bounds[0, 0]) * (self.bounds[1, 1] - self.bounds[0, 1])
        if bounding_box_area <= 0:
            raise ValueError("Polygon bounding box has zero area.")
        
        area_ratio = self.polygon_area.item() / bounding_box_area
        if area_ratio <= 1e-9:
            oversampling_factor = 1000  # Default large oversampling
        else:
            oversampling_factor = 2.0 / area_ratio
        
        batch_size = max(n_total, int(n_total * oversampling_factor))
        
        # Initialize tensor to collect valid points
        try:
            dtype = next(self.vertical_dist.parameters()).dtype
            device = next(self.vertical_dist.parameters()).device
        except (StopIteration, AttributeError):
            dtype = torch.float32
            device = self.bounds.device

        valid_points_xy = torch.empty((0, 2), device=device, dtype=dtype)

        # Keep generating points until we have enough
        attempts = 0
        max_attempts = 100

        while valid_points_xy.shape[0] < n_total and attempts < max_attempts:
            # Generate random points within bounding box
            new_xy = torch.rand(batch_size, 2, device=device, dtype=dtype) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            
            # Filter points inside the polygon
            inside_mask = self._points_in_polygon(new_xy)
            new_valid = new_xy[inside_mask]
            
            # Add to collection
            valid_points_xy = torch.cat([valid_points_xy, new_valid], dim=0)
            attempts += 1
            
            # Adjust batch_size dynamically if needed
            if valid_points_xy.shape[0] < n_total:
                needed = n_total - valid_points_xy.shape[0]
                estimated_yield = area_ratio if area_ratio > 1e-9 else 1e-3
                batch_size = max(needed, int(needed * oversampling_factor / estimated_yield))

        if valid_points_xy.shape[0] < n_total:
            raise RuntimeError(
                f"Failed to generate enough points ({valid_points_xy.shape[0]}/{n_total}) "
                f"within the polygon after {max_attempts} attempts. "
                f"Check polygon area or increase max_attempts/oversampling."
            )

        # If we have more than n_total points, randomly select n_total of them
        if valid_points_xy.shape[0] > n_total:
            indices = torch.randperm(valid_points_xy.shape[0], device=device)[:n_total]
            valid_points_xy = valid_points_xy[indices]
        
        # Sample z coordinates
        expanded_vertical_dist = self.vertical_dist.expand(torch.Size((n_total,)) + self.vertical_dist.batch_shape)
        z_coords = expanded_vertical_dist.sample()
        z_coords = z_coords.reshape(n_total, -1)

        # Combine xy and z, reshape to final desired shape
        final_points = torch.cat([valid_points_xy, z_coords], dim=1)
        
        return final_points.reshape(combined_shape + self.event_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Calculate log probability density for points.
        
        Arguments:
            value: Points to evaluate with shape (..., 3).
            
        Returns:
            Log probability densities.
        """
        # Retain original shape for final output
        original_shape = value.shape[:-1]
        
        # Reshape value to (N, 3) for processing
        value_reshaped = value.reshape(-1, 3)
        n_points = value_reshaped.shape[0]

        # Ensure value tensor is on the correct device
        device = self.bounds.device
        value_reshaped = value_reshaped.to(device)

        # Check which points are inside the polygon horizontally
        inside_mask = self._points_in_polygon(value_reshaped[:, :2])

        # Initialize log_probs with -inf
        log_probs = torch.full((n_points,), float('-inf'), device=device, dtype=value.dtype)

        # Calculate log_prob for points inside the polygon
        if inside_mask.any():
            # Horizontal component: uniform density = 1 / area
            log_prob_horizontal = -torch.log(self.polygon_area.to(device))

            # Vertical component: use log_prob of the vertical distribution
            points_inside = value_reshaped[inside_mask]
            z_values_inside = points_inside[:, 2:]

            # Handle batch shapes if vertical_dist has them
            if self.vertical_dist.batch_shape:
                try:
                    expanded_batch_shape = original_shape
                    expanded_vertical_dist = self.vertical_dist.expand(expanded_batch_shape)
                    flat_expanded_vertical_dist = expanded_vertical_dist.reshape(-1, *expanded_vertical_dist.event_shape)
                    log_prob_vertical = flat_expanded_vertical_dist.log_prob(z_values_inside)[inside_mask.flatten()]
                except (RuntimeError, ValueError) as e:
                    # Fallback to base vertical_dist
                    print(f"Warning: Could not align vertical_dist batch shape. Error: {e}")
                    log_prob_vertical = self.vertical_dist.log_prob(z_values_inside)
            else:
                log_prob_vertical = self.vertical_dist.log_prob(z_values_inside)

            # Combine horizontal and vertical log_probs
            log_probs[inside_mask] = log_prob_horizontal + log_prob_vertical.squeeze(-1)

        # Reshape log_probs back to the original shape
        return log_probs.reshape(original_shape)

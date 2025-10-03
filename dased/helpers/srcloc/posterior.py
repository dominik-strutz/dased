import torch
from typing import Union, Tuple, Callable, Optional

__all__ = ["calculate_posterior"]


def calculate_posterior(
    true_source: torch.Tensor,
    design,
    data_likelihood,
    x_range: Tuple[float, float] = (-5000.0, 5000.0),
    y_range: Tuple[float, float] = (-5000.0, 5000.0),
    z_range: Optional[Tuple[float, float]] = None,
    grid_size: Union[int, Tuple[int, ...]] = 100,
    prior_dist: Optional[Callable] = None,
    clean_data: bool = True,
    downsample: Optional[int] = None,
    seed: int = 0,
    posterior_std_obs: float = 0.02,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict],
]:
    """
    Calculate Bayesian posterior distribution over a 2D or 3D grid.

    This function computes the posterior probability distribution for source
    locations by evaluating the likelihood function over a regular grid and
    combining it with a prior distribution (uniform if not specified).

    Types in this module use Sphinx roles, e.g. :class:`torch.Tensor` for the
    grid and :class:`~dased.helpers.srcloc.DataLikelihood` for likelihood
    objects. The ``design`` argument typically expects a :class:`geopandas.GeoDataFrame`.

    Arguments:
        true_source: :class:`torch.Tensor`
            True source location tensor with shape ``(3,)`` containing
            ``[x, y, z]`` coordinates in meters. Used to generate synthetic data
            and as reference for evaluation.

        design: :class:`geopandas.GeoDataFrame`
            Design GeoDataFrame containing sensor information. Must include a
            ``geometry`` column with Point geometries and an ``elevation``
            column. Used by the data likelihood function.

        data_likelihood: :class:`~dased.helpers.srcloc.DataLikelihood`
            DataLikelihood instance for computing likelihood values. Should be
            properly configured with forward models and uncertainty
            specifications.

        x_range: Tuple of (min, max) values for x-axis in meters.
            Defines the spatial extent of the grid in the x-direction.
            Defaults to (-5000.0, 5000.0).

        y_range: Tuple of (min, max) values for y-axis in meters.
            Defines the spatial extent of the grid in the y-direction.
            Defaults to (-5000.0, 5000.0).

        z_range: Optional tuple of (min, max) values for z-axis in meters.
            If None, computes 2D posterior at the true source depth.
            If provided, computes 3D posterior over the specified depth range.

        grid_size: Number of grid points in each dimension. Can be:

        - Int: Same size for all dimensions (nx=ny=nz=grid_size)
        - Tuple: Specify size for each dimension (nx, ny) or (nx, ny, nz)

            Larger values give higher resolution but increase computation time.
            Defaults to 100.

        prior_dist: Optional prior distribution. Should be a callable that
            accepts grid points and returns log prior probabilities.
            If None, uses uniform prior. Defaults to None.

        clean_data: Whether to use clean synthetic data (True) or add noise (False).
            Clean data is useful for testing and validation.
            Defaults to True.

        downsample: Optional downsampling factor for sensors. If provided,
            uses every ``downsample``-th sensor to reduce computation.
            Useful for large sensor arrays. Defaults to None (use all sensors).

        seed: Random seed for reproducibility of noise generation and sampling.
            Defaults to 0.

        posterior_std_obs: Fill-in value for observation uncertainty in posterior computation. Defaults to 0.02.

        Returns:
                For 2D case: Tuple of (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, dict)
                For 3D case: Tuple of (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, dict)

                Where:
                - x_grid, y_grid, z_grid: Grid coordinate tensors as :class:`torch.Tensor`
                - log_posterior: Log posterior probabilities as :class:`torch.Tensor` with same shape as coordinate grids
                - post_info: Dictionary with additional information including design (:class:`geopandas.GeoDataFrame`),
                    observed data (:class:`torch.Tensor`), data likelihood distribution (:class:`~dased.helpers.srcloc.DataLikelihood`), etc.

    Examples:
        Basic 2D posterior calculation::

            >>> import torch
            >>> # Setup true source location
            >>> true_source = torch.tensor([1000.0, 2000.0, 500.0])
            >>> # Calculate 2D posterior
            >>> x_grid, y_grid, log_post, info = calculate_posterior(
            ...     true_source=true_source,
            ...     design=design_gdf,
            ...     data_likelihood=likelihood,
            ...     x_range=(-2000, 4000),
            ...     y_range=(-1000, 5000),
            ...     grid_size=50
            ... )
            >>> # Find maximum likelihood location
            >>> max_idx = torch.argmax(log_post)
            >>> max_x = x_grid[max_idx // log_post.shape[1]]
            >>> max_y = y_grid[max_idx % log_post.shape[1]]

        3D posterior with custom prior::

            >>> import torch
            >>> # Define prior favoring shallow sources
            >>> class ShallowPrior:
            ...     def log_prob(self, points):
            ...         z_values = points[..., 2]
            ...         return -0.001 * z_values
            >>> 
            >>> shallow_prior = ShallowPrior()
            >>> x_grid, y_grid, z_grid, log_post, info = calculate_posterior(
            ...     true_source=true_source,
            ...     design=design_gdf,
            ...     data_likelihood=likelihood,
            ...     z_range=(0, 2000),  # 0 to 2 km depth
            ...     grid_size=(40, 40, 20),  # Higher resolution in x,y
            ...     prior_dist=shallow_prior
            ... )
    """
    
    # Extract range bounds
    x_min, x_max = x_range
    y_min, y_max = y_range
    is_3d = z_range is not None

    design = design.copy()

    # Apply downsampling if requested
    if downsample is not None and downsample > 1:
        num_channels = len(design)
        indices = []
        for start in range(0, num_channels, downsample):
            end = min(start + downsample, num_channels)
            block_size = end - start
            if block_size > 0:
                mid = start + block_size // 2
                indices.append(mid)

        if not indices:
            raise ValueError(
                "Downsampling factor is too large for the number of channels."
            )
        design = design.iloc[indices].reset_index(drop=True)

    # Determine grid sizes for each dimension
    if isinstance(grid_size, int):
        nx = ny = grid_size
        if is_3d:
            nz = grid_size
    elif isinstance(grid_size, tuple):
        if is_3d:
            if len(grid_size) != 3:
                raise ValueError(
                    "grid_size tuple must have length 3 for 3D grid (nx, ny, nz)"
                )
            nx, ny, nz = grid_size
        else:
            if len(grid_size) != 2:
                raise ValueError(
                    "grid_size tuple must have length 2 for 2D grid (nx, ny)"
                )
            nx, ny = grid_size
    else:
        raise TypeError("grid_size must be an integer or a tuple")

    # Generate synthetic data from true source
    torch.manual_seed(seed)
    d_like_true = data_likelihood(true_source, design)
    data_observed = d_like_true.sample().squeeze()
    
    # Extract clean data and standard deviation using robust attribute access
    try:
        data_clean = d_like_true.loc.squeeze()
        data_std = d_like_true.stddev.squeeze()
    except AttributeError:
        try:
            data_clean = d_like_true.base_dist.loc.squeeze()
            data_std = d_like_true.base_dist.scale.squeeze()
        except AttributeError:
            try:
                data_clean = d_like_true.mean.squeeze()
                data_std = d_like_true.stddev.squeeze()
            except AttributeError:
                raise AttributeError(
                    "Cannot extract location and scale parameters from data likelihood distribution. "
                    "Expected attributes: 'loc'/'stddev' or 'base_dist.loc'/'base_dist.scale' or 'mean'/'stddev'"
                )

    # Create coordinate grids
    x_posterior = torch.linspace(x_min, x_max, nx)
    y_posterior = torch.linspace(y_min, y_max, ny)

    if is_3d:
        z_min, z_max = z_range
        z_posterior = torch.linspace(z_min, z_max, nz)
        X_posterior, Y_posterior, Z_posterior = torch.meshgrid(
            x_posterior, y_posterior, z_posterior, indexing="ij"
        )

        # Stack coordinates to form grid of points
        posterior_grid_full = torch.stack(
            [X_posterior.ravel(), Y_posterior.ravel(), Z_posterior.ravel()], dim=-1
        )
    else:
        X_posterior, Y_posterior = torch.meshgrid(
            x_posterior, y_posterior, indexing="ij"
        )

        # Stack coordinates with fixed z from true_source
        posterior_grid_full = torch.stack(
            [
                X_posterior.ravel(),
                Y_posterior.ravel(),
                true_source[2] * torch.ones_like(X_posterior.ravel()),
            ],
            dim=-1,
        )
        
    # Apply masking based on standard deviation cutoff
    std_diag = d_like_true.stddev.squeeze()
    mask = (std_diag < data_likelihood.std_cutoff).squeeze()

    # Convert mask to boolean numpy array for indexing
    if isinstance(mask, torch.Tensor):
        mask_bool = mask.cpu().numpy()
    else:
        mask_bool = mask
    
    # Handle different mask shapes
    if mask_bool.ndim > 1:
        mask_bool = mask_bool.flatten()
    
    # Apply mask to design and data
    design_masked = design[mask_bool].reset_index(drop=True)
    
    if clean_data:
        data_observed = data_clean[mask]
    else:
        data_observed = data_observed[mask]

    # Calculate log likelihood for all grid points        
    log_likelihood = data_likelihood(
        posterior_grid_full, design_masked, posterior=posterior_std_obs
    ).log_prob(data_observed)

    # Calculate log prior (default to uniform)    
    if prior_dist is None:
        log_prior = torch.zeros_like(log_likelihood)
    else:
        log_prior = prior_dist.log_prob(posterior_grid_full).detach()

    # Calculate unnormalized posterior
    unnormalized_posterior = log_likelihood + log_prior

    # Normalize posterior (subtract max for numerical stability)
    log_posterior = unnormalized_posterior - torch.logsumexp(
        unnormalized_posterior, dim=0
    )

    # Reshape to grid dimensions
    log_posterior = log_posterior.reshape(X_posterior.shape)

    # Prepare additional information
    post_info = dict(
        design=design,
        design_masked=design_masked,
        data=data_observed,
        data_std=data_std,
        data_clean=data_clean,
        mask=mask,
    )

    # Return results based on dimensionality
    if is_3d:
        return x_posterior, y_posterior, z_posterior, log_posterior, post_info
    else:
        return x_posterior, y_posterior, log_posterior, post_info

import torch
import torch.distributions as dist
from typing import Union, List, Dict, Callable, Optional
from .distributions import WeightedIndependentNormal, WeightedMultivariateNormal

__all__ = ["DataLikelihood"]


def _decompose_covariance_matrix(
    diagonal: torch.Tensor, correlation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Decompose a covariance matrix into D @ L for Cholesky decomposition.
    
    Arguments:
        diagonal: 2D tensor of standard deviations (batch, N).
        correlation_matrix: 3D tensor of correlation matrices (batch, N, N).

    Returns:
        The decomposed covariance matrix (batch, N, N).
        
    Raises:
        ValueError: If input dimensions don't match.
    """
    if diagonal.dim() != 2:
        raise ValueError("Diagonal must be a 2D tensor.")
    if correlation_matrix.dim() != 3:
        raise ValueError("Correlation matrix must be a 3D tensor.")
    if diagonal.size(1) != correlation_matrix.size(1) or diagonal.size(1) != correlation_matrix.size(2):
        raise ValueError("Dimensions of diagonal and correlation matrix must match.")

    corr = correlation_matrix.clone().double()
    corr.diagonal(dim1=1, dim2=2).fill_(1.0)

    L = torch.tril(corr)
    torch.diagonal(L, dim1=1, dim2=2).fill_(1.0)

    D = torch.diag_embed(diagonal.double())
    return D @ L


def _shannon_hartley_theorem(
    snr: torch.Tensor, f_max: Union[float, torch.Tensor], K: float = 10
) -> torch.Tensor:
    """
    Compute standard deviation from SNR using the Shannon-Hartley theorem.
    
    Arguments:
        snr: Signal-to-noise ratio tensor.
        f_max: Maximum frequency.
        K: Constant factor.

    Returns:
        Standard deviation tensor.
    """
    snr = snr.double()
    snr_db = 20.0 * torch.log10(snr)

    if isinstance(f_max, (int, float)):
        f_max = torch.tensor(f_max, device=snr.device, dtype=torch.float64)
    else:
        f_max = f_max.double()

    eps_sq = torch.full_like(snr, torch.nan, dtype=torch.float64)
    valid_mask = snr > 1.1
    if valid_mask.any():
        eps_sq[valid_mask] = torch.pow(
            torch.log2(1.0 + snr_db[valid_mask] / float(K)) * 2.0 * f_max, -2
        )

    return eps_sq


def _get_std_function(
    std: Union[Dict[str, Callable], Callable, float], name: str
) -> Union[Dict[str, Callable], Callable]:
    """
    Utility to wrap a float as a function or pass through callables.
    
    Arguments:
        std: Standard deviation specification.
        name: Name for error messages.

    Returns:
        Callable or dict of callables.
    """
    if isinstance(std, (int, float)):
        value = float(std)
        if name == "std_corr":
            return lambda data: torch.sqrt(data.double()) * value
        else:
            return lambda data: torch.full_like(data, value, dtype=torch.float64)
    return std


def _get_design_indices(
    data_types: List[str],
    data_type_design_map: Optional[Dict[str, List[int]]],
    design_length: int,
) -> Dict[str, List[int]]:
    """
    Get the mapping from data type to design row indices.
    
    Arguments:
        data_types: List of data type names.
        data_type_design_map: Optional mapping.
        design_length: Number of design rows.

    Returns:
        Mapping from data type to list of indices.
    """
    if data_type_design_map is not None:
        return {dt: data_type_design_map.get(dt, []) for dt in data_types}
    else:
        return {dt: list(range(design_length)) for dt in data_types}


def _build_reduced_design(design, data_types: List[str], design_indices_per_type: Dict[str, List[int]]):
    """
    Build a reduced design DataFrame for correlated case.
    
    Arguments:
        design: The full design DataFrame.
        data_types: List of data type names.
        design_indices_per_type: Mapping from data type to indices.

    Returns:
        Reduced design DataFrame.
    """
    if design_indices_per_type is not None:
        reduced_design_rows = []
        for dt in data_types:
            reduced_design_rows.extend(design_indices_per_type[dt])
        return design.iloc[reduced_design_rows].reset_index(drop=True)
    else:
        return design


class DataLikelihood:
    """
    Computes likelihood of observed data given source model parameters and
    experimental design.

    This class handles computation of data likelihood for Bayesian source
    localization. It supports correlated and uncorrelated noise models and
    multiple data types. Docstrings reference types using Sphinx roles such
    as :class:`torch.Tensor`, :class:`pandas.DataFrame` and
    :class:`geopandas.GeoDataFrame`.

    Arguments:
        forward_function: Forward model(s) for each data type. Can be:
        
            - Single callable: Used for all data (single data type)
            - Dict of callables: Keys are data type names, values are forward functions
            
            Each forward function should accept (model_samples, design) and return
            either predictions or (predictions, snr_values) tuple.
            
        std_corr: Correlated standard deviation specification. Can be:
        
            - Float: Constant correlated uncertainty for all data
            - Callable: Function of predicted data returning std values
            - Dict: Mapping data types to float/callable specifications
            
        std_uncorr: Uncorrelated standard deviation specification.
            Same format options as ``std_corr``.
            
        cor_length: Correlation length for spatial correlation in meters.
            If 0.0, assumes uncorrelated measurements. Defaults to 0.0.
            
        std_cutoff: Maximum allowed standard deviation for SNR conversion.
            Used to prevent numerical issues. Defaults to 10.0.
            
        f_max: Maximum frequency for SNR to standard deviation conversion
            using Shannon-Hartley theorem. Defaults to 10.0 Hz.
            
        K_sh: Constant for Shannon-Hartley SNR conversion. Defaults to 10.0.
        
        data_type_design_map: Optional mapping from data type names to lists
            of design row indices. If None, all design rows are used for all
            data types. Useful when different data types use different subsets
            of sensors.

    Examples:
        Basic setup with single forward model::

            >>> from dased.helpers.srcloc import DataLikelihood, ForwardHomogeneous
            >>> from dased.helpers.srcloc import MagnitudeRelation
            >>> 
            >>> # Setup forward model
            >>> mag_rel = MagnitudeRelation(magnitude_factor=1.0, reference_distance=1000)
            >>> forward = ForwardHomogeneous(velocity=5000, wave_type='P', distance_relation=mag_rel)  # :class:`~dased.helpers.srcloc.forward.ForwardHomogeneous`
            >>> 
            >>> # Create likelihood with uncorrelated noise
            >>> likelihood = DataLikelihood(  # :class:`~dased.helpers.srcloc.data_likelihood.DataLikelihood`
            ...     forward_function=forward,
            ...     std_corr=0.1,     # 0.1 second correlated uncertainty
            ...     std_uncorr=0.05,  # 0.05 second uncorrelated uncertainty
            ...     cor_length=0.0    # No spatial correlation
            ... )
            
        Multiple data types with different forward models::
        
            >>> forward_models = {
            ...     'P_wave': forward_p,
            ...     'S_wave': forward_s
            ... }
            >>> std_specs = {
            ...     'P_wave': 0.1,
            ...     'S_wave': 0.2
            ... }
            >>> likelihood = DataLikelihood(
            ...     forward_function=forward_models,
            ...     std_corr=std_specs,
            ...     std_uncorr=0.05
            ... )
    """
    #TODO: make sure code examples actually work

    def __init__(
        self,
        forward_function: Union[Dict[str, Callable], Callable],
        std_corr: Union[Dict[str, Callable], Callable, float],
        std_uncorr: Union[Dict[str, Callable], Callable, float],
        cor_length: float = 0.0,
        std_cutoff: float = 10.0,
        f_max: float = 10.0,
        K_sh: float = 10.0,
        data_type_design_map: Optional[Dict[str, List[int]]] = None,
        remove_mean=True,
    ):
        self.forward_function = forward_function
        self.std_corr = _get_std_function(std_corr, "std_corr")
        self.std_uncorr = _get_std_function(std_uncorr, "std_uncorr")
        self.cor_length = float(cor_length)
        self.std_cutoff = float(std_cutoff)
        self.f_max = float(f_max)
        self.K_sh = float(K_sh)
        self.data_type_design_map = data_type_design_map
        self._validate_inputs()
        self.remove_mean = remove_mean

    def __call__(
        self, 
        model_samples: torch.Tensor, 
        design,
        posterior=None,
    ) -> Union[WeightedIndependentNormal, WeightedMultivariateNormal]:
        """
        Compute the data likelihood distribution for given model samples and design.

        Arguments:
            model_samples: Model parameter samples with shape (..., n_params).
                Last dimension typically contains [x, y, z] coordinates.
                
            design: Design DataFrame containing sensor information. Must include
                'geometry' column with Point geometries and 'elevation' column.
                For correlated case, geometries are used for distance calculations.

        Returns:
            Likelihood distribution. Returns IndependentNormal for uncorrelated
            case or MultivariateNormal for correlated case.
        """        
        self._validate_design(design)
        design = design.reset_index(drop=True)

        if model_samples.ndim == 1:
            model_samples = model_samples.unsqueeze(0)
        model_samples = model_samples.double()

        # Determine data types
        if isinstance(self.forward_function, dict):
            data_types = list(self.forward_function.keys())
        else:
            data_types = ["generic"]

        design_indices_per_type = _get_design_indices(
            data_types, self.data_type_design_map, len(design)
        )
        
        N_designs_per_type = [len(design_indices_per_type[dt]) for dt in data_types]
        total_measurements = sum(N_designs_per_type)
        data_shape = model_samples.shape[:-1] + (total_measurements,)

        # Initialize output tensors
        arrival_times = torch.zeros(
            data_shape, dtype=torch.float64, device=model_samples.device
        )
        std_obs = torch.zeros_like(arrival_times, dtype=torch.float64)
        std_corr = torch.zeros_like(arrival_times, dtype=torch.float64)
        std_uncorr = torch.zeros_like(arrival_times, dtype=torch.float64)

        # Process each data type
        offset = 0
        for i, data_type in enumerate(data_types):
            indices = range(offset, offset + N_designs_per_type[i])
            design_rows = design_indices_per_type[data_type]
            if not design_rows:
                continue

            # Get forward function for this data type
            forward_func = (
                self.forward_function[data_type]
                if isinstance(self.forward_function, dict)
                else self.forward_function
            )
            
            design_subset = design.iloc[design_rows].reset_index(drop=True)
            output = forward_func(model_samples, design_subset)
            
            # Handle output format
            if isinstance(output, tuple):
                arrival_times[..., indices], snr_values = output
            else:
                arrival_times[..., indices] = output
                snr_values = None

            if posterior is None:
                # Process SNR values if provided
                if snr_values is not None:
                    snr_values = snr_values.double()
                    
                    # Apply K factors for SNR scaling
                    k_col_specific = f"K_{data_type}"
                    k_col_generic = "K"
                    if k_col_specific in design_subset.columns:
                        k_factors = torch.tensor(
                            design_subset[k_col_specific].fillna(1.0).values,
                            device=snr_values.device,
                            dtype=torch.float64,
                        )
                        k_factors[k_factors < 0] = 0.0
                        snr_values *= k_factors
                    elif k_col_generic in design_subset.columns:
                        k_factors = torch.tensor(
                            design_subset[k_col_generic].fillna(1.0).values,
                            device=snr_values.device,
                            dtype=torch.float64,
                        )
                        k_factors[k_factors < 0] = 0.0
                        snr_values *= k_factors
                        
                    # Apply signal strength factors
                    if "signal_strength" in design_subset.columns:
                        sig_strength = torch.tensor(
                            design_subset["signal_strength"].fillna(1.0).values,
                            device=snr_values.device,
                            dtype=torch.float64,
                        )
                        sig_strength[sig_strength < 0] = 0.0
                        snr_values *= sig_strength

                    std_obs[..., indices] = self.snr2std(snr_values)
                    
            else:
                std_obs[..., indices] = posterior

            # Calculate uncertainty components
            std_corr_func = (
                self.std_corr[data_type] if isinstance(self.std_corr, dict) else self.std_corr
            )
            std_uncorr_func = (
                self.std_uncorr[data_type] if isinstance(self.std_uncorr, dict) else self.std_uncorr
            )
            std_corr[..., indices] = std_corr_func(arrival_times[..., indices].double())
            std_uncorr[..., indices] = std_uncorr_func(arrival_times[..., indices].double())

            offset += N_designs_per_type[i]

        # Build reduced design for correlated case
        reduced_design = _build_reduced_design(
            design, data_types, design_indices_per_type if self.data_type_design_map is not None else None
        )

        # Return appropriate likelihood distribution
        if self.cor_length == 0.0:
            return self._handle_uncorrelated_case(
                arrival_times, std_obs, std_corr, std_uncorr
            )
        else:
            return self._handle_correlated_case(
                arrival_times, std_obs, std_corr, std_uncorr, reduced_design
            )

    def _handle_uncorrelated_case(
        self, 
        mean: torch.Tensor, 
        std_obs: torch.Tensor, 
        std_corr: torch.Tensor, 
        std_uncorr: torch.Tensor, 
    ) -> WeightedIndependentNormal:
        """
        Handle the uncorrelated likelihood case.

        Arguments:
            mean: Mean tensor.
            std_obs: Observational std tensor.
            std_corr: Correlated std tensor.
            std_uncorr: Uncorrelated std tensor.

        Returns:
            IndependentNormal distribution.
        """
        mean = mean.double()
        std_obs = std_obs.double()
        std_corr = std_corr.double()
        std_uncorr = std_uncorr.double()
        
        var = std_obs.pow(2) + std_corr.pow(2) + std_uncorr.pow(2)
        # var = std_corr.pow(2) + std_uncorr.pow(2)
            
            
        # # Weighted mean calculation for relative likelihood
        # weights = var.reciprocal()
        # weights = weights / torch.nansum(weights, dim=-1, keepdim=True)
        # weighted_mean = torch.nansum(mean * weights, dim=-1, keepdim=True)
        # mean = mean - weighted_mean
    
        if self.remove_mean:            
            return WeightedIndependentNormal(mean.float(), var.sqrt().float())
        else:
            return dist.Independent(dist.Normal(mean.float(), var.sqrt().float()), 1)
            
    def _handle_correlated_case(
        self, 
        mean: torch.Tensor, 
        std_obs: torch.Tensor, 
        std_corr: torch.Tensor, 
        std_uncorr: torch.Tensor, 
        design, 
    ) -> WeightedMultivariateNormal:
        """
        Handle the correlated likelihood case.

        Arguments:
            mean: Mean tensor.
            std_obs: Observational std tensor.
            std_corr: Correlated std tensor.
            std_uncorr: Uncorrelated std tensor.
            design: Reduced design DataFrame.

        Returns:
            MultivariateNormal distribution.
        """
        mean = mean.double()
        std_obs = std_obs.double()
        std_corr = std_corr.double()
        std_uncorr = std_uncorr.double()
        
        # Calculate spatial correlation matrix
        coordinates = torch.tensor(
            [[float(geom.x), float(geom.y)] for geom in design.geometry],
            dtype=torch.float64,
            device=mean.device,
        )
        
        # Expand coordinates if needed for multiple data types
        coordinates = (
            coordinates.repeat(len(mean[0]) // len(design), 1)
            if mean.shape[-1] > len(design)
            else coordinates
        )
        
        # Calculate distance-based correlation
        distance_tensor = torch.cdist(coordinates, coordinates, p=2).double()
        exp_factor = -0.5 / (self.cor_length**2)
        correlation_matrix = torch.exp(distance_tensor.pow(2) * exp_factor)
        correlation_matrix = torch.clamp(correlation_matrix, max=0.99)
        correlation_matrix.diagonal().fill_(1.0)
        correlation_matrix = correlation_matrix.expand(mean.shape[0], -1, -1).double()
        
        # Decompose covariance matrix
        tt_tril = _decompose_covariance_matrix(std_corr, correlation_matrix)
        tt_tril = torch.tril(tt_tril)
        
        # Add uncorrelated and observational noise to diagonal
        diag_indices = torch.arange(tt_tril.shape[-1], device=tt_tril.device)
        diag_values = tt_tril[:, diag_indices, diag_indices]
        tt_tril[:, diag_indices, diag_indices] = torch.sqrt(
            diag_values.pow(2) + std_uncorr.pow(2) + std_obs.pow(2)
        )
            
        # # Calculate weighted mean for relative likelihood
        # identity = torch.eye(
        #     tt_tril.size(-1), device=tt_tril.device, dtype=torch.float64
        # ).expand(tt_tril.shape[:-2] + (-1, -1))
        # precision_matrices = torch.cholesky_solve(identity, tt_tril)
        # weights = torch.sum(precision_matrices, dim=-1, keepdim=True).squeeze(-1)
        # weight_sum = torch.sum(weights, dim=-1, keepdim=True)
        # weighted_mean = torch.sum(mean * weights, dim=-1, keepdim=True) / weight_sum
        # centered_mean = mean - weighted_mean
    
        if self.remove_mean:            
            return WeightedMultivariateNormal(mean.float(), scale_tril=tt_tril.float())
        else:
            return dist.MultivariateNormal(mean.float(), scale_tril=tt_tril.float())

    def snr2std(self, snr: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Convert SNR to standard deviation using the Shannon-Hartley theorem.

        Arguments:
            snr: SNR tensor or None.

        Returns:
            Standard deviation tensor.
        """
        if snr is None:
            return torch.tensor(self.std_cutoff, dtype=torch.float64)
        result = _shannon_hartley_theorem(snr, self.f_max, self.K_sh)
        return result.nan_to_num_(nan=self.std_cutoff).clamp_max(self.std_cutoff)

    def _validate_design(self, design) -> None:
        """
        Validate the design DataFrame.

        Arguments:
            design: Design DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = ["geometry"]
        for col in required_columns:
            if col not in design.columns:
                raise ValueError(f"Design data must contain a '{col}' column")
        if "elevation" not in design.columns:
            design["elevation"] = 0.0

    def _validate_inputs(self) -> None:
        """
        Validate the initialization arguments.

        Raises:
            TypeError: If arguments have wrong types.
            ValueError: If arguments have invalid values.
        """
        for name, param in [
            ("forward_function", self.forward_function),
            ("std_corr", self.std_corr),
            ("std_uncorr", self.std_uncorr),
        ]:
            if not (
                callable(param)
                or (
                    isinstance(param, dict) and all(callable(v) for v in param.values())
                )
            ):
                raise TypeError(
                    f"{name} must be either a callable or a dictionary of callables"
                )
                
        for name, param in [
            ("cor_length", self.cor_length),
            ("std_cutoff", self.std_cutoff),
            ("K_sh", self.K_sh),
        ]:
            if not isinstance(param, (int, float)):
                raise TypeError(f"{name} must be a numeric value")
            if param < 0:
                raise ValueError(f"{name} must be non-negative")

import logging
from typing import Any, Dict, Union

import numpy as np
import torch
from geopandas import GeoDataFrame

from ..layout import DASLayout

try:
    from geobed import BED_base_explicit

    _GEOBED_AVAILABLE = True
except ImportError:
    _GEOBED_AVAILABLE = False
    BED_base_explicit = None

__all__ = ["EIGCriterion"]

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
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)


# Ensure logger has a handler at module import
_ensure_logger_handler(logger)


class EIGCriterion:
    r"""
    Expected Information Gain (EIG) criterion for Bayesian experimental design
    of DAS layouts.

    This class evaluates DAS layout designs based on the Expected Information
    Gain (EIG), which quantifies how much information a design is expected to
    provide about model parameters given prior knowledge and a data likelihood
    function. The implementation relies on the ``geobed`` package for EIG
    computation.

    Arguments:
        samples: Prior samples of model parameters as a 2D array/tensor with
            shape ``(N_samples, N_parameters)``. Each row represents one
            sample from the prior distribution. Can be a :class:`~numpy.ndarray`
                or :class:`torch.Tensor`. The implementation converts inputs to
                :class:`torch.Tensor` internally for compatibility with ``geobed``.

        data_likelihood: Callable function that computes the likelihood of observed
            data given model parameters and experimental design. Should have signature:

                        ``data_likelihood(model_params, design_gdf, data) -> distribution``

                        Where:
                        - ``model_params``: Current parameter values from prior samples
                        - ``design_gdf``: :class:`geopandas.GeoDataFrame` representing the
                            DAS layout (see :meth:`~dased.layout.DASLayout.get_gdf`)
                        - ``data``: Observed or simulated data
                        - ``distribution``: A :class:`torch.distributions.Distribution`
                            instance representing the likelihood of the data given the
                            parameters and design.

        eig_method: Method for EIG calculation. Supported methods depend on `geobed`
            implementation. Common options include:

            - ``"NMC"``: Nested Monte Carlo (default)
            - ``"DN"``: :math:`\mathrm{D}_N`-Method

            Defaults to "NMC".

        random_seed: Random seed for reproducible EIG calculations. Ensures
            consistent results across multiple runs with the same inputs.
            Defaults to 0.

        downsample: Optional downsampling factor for reducing computational cost.
            If specified, reduces the number of channels in the layout by selecting
            representative channels from blocks of size `downsample`. Useful when full resolution is not required.

            If None, uses all channels. If > 1, applies downsampling.
            Defaults to None.

        **kwargs: Additional keyword arguments passed to the EIG calculation method.
            Common options include:

            - ``N``: Number of samples to use (default: min(1000, n_samples))
            - ``reuse_M``: Whether to reuse samples in the inner loop of the NMC method for efficiency (NMC only)

            See `geobed` documentation for method-specific parameters.
    Raises:
        ImportError: If `geobed` package is not installed.
        TypeError: If `samples` is not a numpy array or torch tensor, or if
            `data_likelihood` is not callable.
        ValueError: If `samples` is not 2D, has fewer than required samples for
            the chosen method, or if EIG method parameters are invalid.

    Examples:
        Basic EIG criterion setup::

            >>> import numpy as np
            >>> from geopandas import GeoDataFrame
            >>>
            >>> # Generate prior samples (e.g., velocity model parameters) :class:`~numpy.ndarray`
            >>> prior_samples = np.random.normal(3000, 300, (1000, 2))  # :class:`~numpy.ndarray`
            >>>
            >>> # Define likelihood function
            >>> def likelihood_func(params, design_gdf, data):
            ...     # Example likelihood: Gaussian with mean based on params and design
            ...     mean = f(params, design_gdf)  # User-defined function
            ...     return torch.distributions.Normal(loc=mean, scale=1.0)  # :class:`torch.distributions.Normal`
            >>>
            >>> # Create EIG criterion
            >>> eig_criterion = EIGCriterion(
            ...     samples=prior_samples,
            ...     data_likelihood=likelihood_func,
            ...     eig_method="NMC"
            ... )

        Evaluating a DAS layout::

            >>> from dased.layout import DASLayout
            >>>
            >>> # Create layout (knots as a NumPy array :class:`~numpy.ndarray`)
            >>> knots = np.array([[0, 0], [1000, 0]])  # :class:`~numpy.ndarray`
            >>> layout = DASLayout(knots, spacing=10.0)
            >>>
            >>> # Calculate EIG
            >>> eig_value = eig_criterion(layout)
            >>> print(f"Expected Information Gain: {eig_value:.3f}")
    """

    # TODO: Check if code example runs

    def __init__(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        data_likelihood: callable,
        eig_method: str = "NMC",
        random_seed: int = 0,
        downsample: int = None,
        **kwargs: Any,
    ):
        if not _GEOBED_AVAILABLE:
            raise ImportError(
                "The geobed package is required for EIG calculations. "
                "Please install it using: pip install geobed"
            )

        self._validate_samples(samples)
        self.samples = (
            samples if isinstance(samples, torch.Tensor) else torch.tensor(samples)
        )
        self.N_samples = self.samples.shape[0]
        self.eig_method = eig_method
        self.kwargs = self._configure_eig_kwargs(kwargs, eig_method, self.N_samples)
        self.random_seed = random_seed
        self.downsample = downsample

        if not callable(data_likelihood):
            raise TypeError("data_likelihood must be a callable function.")

        self.BEDclass = BED_base_explicit(
            m_prior_dist=self.samples,
            data_likelihood_func=data_likelihood,
        )

    def _validate_samples(self, samples: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Validate the input prior samples.

        Arguments:
            samples: Prior samples to validate

        Raises:
            TypeError: If samples is not a numpy array or torch tensor
            ValueError: If samples is not 2D or has insufficient samples
        """
        if not isinstance(samples, (np.ndarray, torch.Tensor)):
            raise TypeError("Samples must be a numpy array or torch tensor.")
        if samples.ndim != 2:
            raise ValueError(
                f"Samples must be a 2D array/tensor. Got shape: {samples.shape}"
            )
        if samples.shape[0] < 10:
            logger.warning(
                f"Only {samples.shape[0]} samples provided. "
                "More samples are generally recommended for stable EIG estimates."
            )

    def _configure_eig_kwargs(
        self, kwargs: Dict[str, Any], eig_method: str, n_samples: int
    ) -> Dict[str, Any]:
        """
        Configure and validate keyword arguments for EIG calculation.

        Arguments:
            kwargs: User-provided keyword arguments
            eig_method: Selected EIG calculation method
            n_samples: Number of available prior samples

        Returns:
            Dictionary of validated and configured EIG parameters

        Raises:
            ValueError: If parameters are invalid for the chosen method
        """
        eig_kwargs = kwargs.copy()
        eig_kwargs.setdefault("N", min(1000, n_samples))

        if eig_method == "NMC":
            eig_kwargs.setdefault("reuse_M", True)

        if eig_kwargs["N"] > n_samples:
            raise ValueError(
                f"Number of samples for EIG method ({eig_kwargs['N']}) "
                f"cannot exceed available prior samples ({n_samples})."
            )
        if eig_kwargs["N"] <= 0:
            raise ValueError("Number of samples 'N' must be positive.")

        return eig_kwargs

    def __call__(self, design: DesignInput) -> float:
        """
        Calculate the Expected Information Gain for a given DAS layout design.

        This method evaluates how much information the specified layout is expected
        to provide about the model parameters, considering the prior uncertainty
        and the data likelihood function.

        Arguments:
            design: DAS layout design to evaluate. Can be either:

            - :class:`~dased.layout.DASLayout`: Layout object containing channel positions and properties
            - :class:`geopandas.GeoDataFrame`: Direct GeoDataFrame representation of the layout

                The design must contain valid geometry (Point objects for channels)
                and any required properties for the likelihood function.

        Returns:
            Expected Information Gain value as a float. Higher values indicate
            designs that are expected to provide more information about the
            model parameters.

        Raises:
            TypeError: If design is not a DASLayout or GeoDataFrame
            ValueError: If design lacks required geometry or has invalid structure,
                or if downsampling parameters result in no remaining channels

        Examples:
            Evaluate a simple linear layout::

                >>> import numpy as np
                >>> from dased.layout import DASLayout
                >>>
                >>> # Create layout (knots as a NumPy array :class:`~numpy.ndarray`)
                >>> knots = np.array([[0, 0], [1000, 0]])  # :class:`~numpy.ndarray`
                >>> layout = DASLayout(knots, spacing=25.0)
                >>>
                >>> # Calculate EIG (assuming criterion is already configured)
                >>> eig_value = eig_criterion(layout)
                >>> print(f"EIG: {eig_value:.3f}")

            Evaluate with downsampling::

                >>> # Create criterion with downsampling
                >>> eig_criterion_ds = EIGCriterion(
                ...     samples=prior_samples,
                ...     data_likelihood=likelihood_func,
                ...     downsample=4  # Use every 4th channel
                ... )
                >>> eig_value_ds = eig_criterion_ds(layout)
        """
        if isinstance(design, DASLayout):
            design_gdf = design.get_gdf()
        elif isinstance(design, GeoDataFrame):
            design_gdf = design
        else:
            raise TypeError("Input 'design' must be a DASLayout or GeoDataFrame.")

        # Apply downsampling if specified
        if self.downsample is not None and self.downsample > 1:
            design_gdf = self._apply_downsampling(design_gdf)

        if (
            not isinstance(design_gdf, GeoDataFrame)
            or "geometry" not in design_gdf.columns
        ):
            raise ValueError(
                "Input design must result in a GeoDataFrame with a 'geometry' column."
            )

        eig, _ = self.BEDclass.calculate_EIG(
            design=[design_gdf],
            eig_method=self.eig_method,
            eig_method_kwargs=self.kwargs,
            random_seed=self.random_seed,
        )

        return float(eig[0]) if isinstance(eig, (list, np.ndarray)) else float(eig)

    def _apply_downsampling(self, design_gdf: GeoDataFrame) -> GeoDataFrame:
        """
        Apply downsampling to the design by selecting representative channels.

        Arguments:
            design_gdf: Original GeoDataFrame with all channels

        Returns:
            Downsampled GeoDataFrame with reduced channel count

        Raises:
            ValueError: If downsampling factor is too large for the channel count
        """
        num_channels = len(design_gdf)
        indices = []

        for start in range(0, num_channels, self.downsample):
            end = min(start + self.downsample, num_channels)
            block_size = end - start
            if block_size > 0:
                mid = start + block_size // 2
                indices.append(mid)

        if not indices:
            raise ValueError(
                f"Downsampling factor ({self.downsample}) is too large for the "
                f"number of channels ({num_channels})."
            )

        return design_gdf.iloc[indices].reset_index(drop=True)

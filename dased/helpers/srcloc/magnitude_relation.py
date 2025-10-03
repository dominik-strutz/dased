import torch
from typing import Union

__all__ = ["MagnitudeRelation", "MagnitudeMultiplier"]

# Small epsilon to avoid numerical issues
_EPSILON = torch.finfo(torch.float32).eps

#TODO: is this redundant?
class MagnitudeMultiplier:
    """
    Wrapper to apply a multiplier to any decay function.
    
    This class allows scaling the output of any distance-based decay function
    by a constant multiplier. Docstrings use Sphinx roles such as
    :class:`torch.Tensor` for return types where appropriate.
    
    Parameters
    ----------
    decay_instance : callable
        The decay function instance to wrap. Should be callable and accept distance as input.
    multiplier : float, optional
        Factor to multiply the decay value by. Defaults to 1.0.
        
    Examples
    --------
    Apply a 2x multiplier to a magnitude relation::

        >>> base_relation = MagnitudeRelation(magnitude_factor=1.0, reference_distance=1000)
        >>> amplified_relation = MagnitudeMultiplier(base_relation, multiplier=2.0)
        >>> signal = amplified_relation(distance=500)  # 2x the base signal
    """
    
    def __init__(self, decay_instance, multiplier: float = 1.0):
        self.decay_instance = decay_instance
        self.multiplier = float(multiplier)
        
    def __call__(self, distance: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate decay with applied multiplier.
        
        Parameters
        ----------
        distance : torch.Tensor or float
            Tensor or scalar of distances in meters.

        Returns
        -------
        torch.Tensor
            Tensor of decay values with multiplier applied.
        """
        return self.decay_instance(distance) * self.multiplier


class MagnitudeRelation:
    """
    Magnitude-distance relationship for seismic signal amplitude modeling.

    Implements the relationship::
    
        log10(Amplitude) = magnitude_factor * M_l + log_coeff * log10(R)

    Where R is the distance in kilometers and M_l is the local magnitude.
    This relationship is commonly used in seismology to relate observed
    signal amplitudes to earthquake magnitude and distance.

    Arguments:
        log_coeff (float): Coefficient for the log10(distance) term. Controls how
            amplitude decreases with distance. Typical values range from
            -1.0 to -3.0. Defaults to 0 (no distance dependence).

        magnitude_factor (float): Coefficient for the magnitude term. Controls how
            amplitude scales with magnitude. Defaults to 1.0.

        reference_distance (float | None): Reference distance in meters for relative amplitude
            calculations. If provided, :pyattr:`reference_magnitude` is calculated
            assuming amplitude=1 at this distance. Cannot be used with
            ``reference_magnitude`` or ``reference_relation``.

        reference_magnitude (float | None): Explicit reference magnitude value. Overrides
            calculation from ``reference_distance``. Cannot be used with
            ``reference_distance`` or ``reference_relation``.

        reference_relation (:class:`~dased.helpers.srcloc.magnitude_relation.MagnitudeRelation` | None):
            Another :class:`~dased.helpers.srcloc.magnitude_relation.MagnitudeRelation` instance to inherit
            the reference magnitude from. Overrides both ``reference_magnitude``
            and ``reference_distance``.

    Raises:
        ValueError: If ``magnitude_factor`` is zero, or if none of the reference
            parameters are provided, or if multiple reference parameters are given.

    Examples:
        Standard magnitude-distance relationship::

            >>> # Richter magnitude relation with -2.0 distance decay
            >>> relation = MagnitudeRelation(
            ...     log_coeff=-2.0, 
            ...     magnitude_factor=1.0,
            ...     reference_distance=1000.0  # 1 km reference
            ... )
            >>> amplitude = relation(distance=5000.0)  # 5 km distance, returns :class:`torch.Tensor`
            
        Using magnitude and distance to calculate amplitude::
        
            >>> magnitude = 4.5
            >>> distance = 2000.0  # 2 km
            >>> amplitude = relation.get_amplitude(magnitude, distance)  # :class:`torch.Tensor`
            
        Converting observed amplitude to magnitude::
        
            >>> observed_amplitude = 0.01
            >>> distance = 3000.0  # 3 km
            >>> estimated_magnitude = relation.get_magnitude(observed_amplitude, distance)  # :class:`torch.Tensor`
    """
    
    def __init__(
        self,
        log_coeff: float = 0,
        magnitude_factor: float = 1.0,
        reference_distance: Union[float, None] = None,
        reference_magnitude: Union[float, None] = None,
        reference_relation: Union["MagnitudeRelation", None] = None,
    ):
        self.log_coeff = float(log_coeff)
        self.magnitude_factor = float(magnitude_factor)
        self.reference_distance = reference_distance

        if self.magnitude_factor == 0:
            raise ValueError("magnitude_factor cannot be zero.")

        # Count how many reference parameters are provided
        ref_params = [reference_relation, reference_magnitude, reference_distance]
        provided_params = sum(1 for param in ref_params if param is not None)
        
        if provided_params == 0:
            raise ValueError(
                "Either reference_magnitude, reference_distance, or reference_relation must be provided."
            )
        elif provided_params > 1:
            raise ValueError(
                "Only one of reference_magnitude, reference_distance, or reference_relation can be provided."
            )

        # Determine the reference magnitude
        if reference_relation is not None:
            self.reference_magnitude = reference_relation.reference_magnitude
            self.reference_distance = reference_relation.reference_distance
        elif reference_magnitude is not None:
            self.reference_magnitude = reference_magnitude
        elif reference_distance is not None:
            # Calculate reference magnitude assuming Amplitude=1 at reference_distance
            ref_amp = torch.tensor(1.0)
            ref_dist = torch.tensor(reference_distance) if not isinstance(reference_distance, torch.Tensor) else reference_distance
            self.reference_magnitude = self.get_magnitude(ref_amp, ref_dist)

        # Ensure reference_magnitude is a tensor
        if not isinstance(self.reference_magnitude, torch.Tensor):
            self.reference_magnitude = torch.tensor(self.reference_magnitude, dtype=torch.float32)

    def _ensure_tensor(self, value: Union[torch.Tensor, float]) -> torch.Tensor:
        """Convert input to a tensor if it isn't already."""
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, dtype=torch.float32)
        return value

    def get_magnitude(self, amplitude: Union[torch.Tensor, float], distance: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the magnitude (M_l) based on observed amplitude and distance.

        Uses the inverse relationship::
        
            M_l = (log10(Amplitude) - log_coeff * log10(R)) / magnitude_factor

        Arguments:
            amplitude: Observed amplitude(s) of the signal. Must be positive.
            distance: Distance(s) from the source in meters.

        Returns:
            Calculated magnitude(s) as a torch.Tensor.
            
        Note:
            Distance is converted to kilometers internally for the calculation.
        """
        #TODO: display math properly in docstring
        amplitude = self._ensure_tensor(amplitude)
        distance = self._ensure_tensor(distance) / 1e3  # Convert to km

        # Clamp values to avoid log(0) or log(<0)
        safe_distance = torch.clamp(distance, min=_EPSILON)
        safe_amplitude = torch.clamp(amplitude, min=_EPSILON)

        log_dist_term = self.log_coeff * torch.log10(safe_distance)
        log_amp_term = torch.log10(safe_amplitude)

        magnitude = (log_amp_term - log_dist_term) / self.magnitude_factor
        return magnitude

    def get_amplitude(self, magnitude: Union[torch.Tensor, float], distance: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the amplitude based on magnitude (M_l) and distance.

        Uses the relationship::
        
            Amplitude = 10^(magnitude_factor * M_l + log_coeff * log10(R))

        Arguments:
            magnitude: Local magnitude (M_l) value(s).
            distance: Distance(s) from the source in meters.

        Returns:
            Calculated amplitude(s) as a torch.Tensor.
            
        Note:
            Distance is converted to kilometers internally for the calculation.
        """
        #TODO: display math properly in docstring
        magnitude = self._ensure_tensor(magnitude)
        distance = self._ensure_tensor(distance) / 1e3  # Convert to km

        # Clamp distance to avoid log(0) or log(<0)
        safe_distance = torch.clamp(distance, min=_EPSILON)

        log_dist_term = self.log_coeff * torch.log10(safe_distance)
        exponent = self.magnitude_factor * magnitude + log_dist_term
        amplitude = torch.pow(10.0, exponent)

        return amplitude

    def __call__(self, distance: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate amplitude for given distance using the stored reference magnitude.
        
        This represents the amplitude relative to a baseline defined by the
        reference magnitude, useful for signal-to-noise ratio calculations.

        Arguments:
            distance: Distance(s) from the source in meters.

        Returns:
            Calculated amplitude(s) for the reference magnitude as a torch.Tensor.
        """
        distance = self._ensure_tensor(distance)
        # Ensure reference magnitude is on the same device as distance
        ref_mag_device = self.reference_magnitude.to(distance.device)
        return self.get_amplitude(ref_mag_device, distance)

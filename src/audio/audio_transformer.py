"""
Audio Transformation Engine.

Allowed operations:
- Selective muting
- Volume attenuation
- Frequency filtering
- Directional dampening

All operations must be:
- Identity-grounded
- Spatially aware
- Temporally smooth
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from loguru import logger

from src.core.contracts import (
    AudioOperation,
    TransformParameters,
    SafetyConstraints,
)


@dataclass
class FilterState:
    """State for IIR filters to maintain continuity."""
    zi: Optional[NDArray[np.float64]] = None


class AudioTransformer:
    """
    Audio transformation engine.
    
    Guarantees:
    - All changes are temporally eased
    - No sudden silence
    - Ambient continuity preserved
    - Operations are reversible
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        min_volume: float = 0.05,
        max_transition_rate: float = 0.2,  # per second
    ):
        """
        Initialize audio transformer.
        
        Args:
            sample_rate: Audio sample rate
            min_volume: Minimum allowed volume (never fully silent)
            max_transition_rate: Maximum volume change per second
        """
        self.sample_rate = sample_rate
        self.min_volume = min_volume
        self.max_transition_rate = max_transition_rate
        
        # Current state for smooth transitions
        self._current_volume: float = 1.0
        self._target_volume: float = 1.0
        
        # Filter states for continuity
        self._filter_states: Dict[str, FilterState] = {}
        
        # Precomputed filters
        self._filters: Dict[str, tuple] = {}
        self._initialize_filters()
    
    def _initialize_filters(self):
        """Pre-compute common filter coefficients."""
        # Low-pass filter at 4kHz (reduce harsh sounds)
        self._filters['lowpass_4k'] = signal.butter(
            4, 4000, btype='low', fs=self.sample_rate
        )
        
        # High-pass filter at 200Hz (reduce bass)
        self._filters['highpass_200'] = signal.butter(
            4, 200, btype='high', fs=self.sample_rate
        )
        
        # Band-stop filter for mid frequencies (1-3kHz)
        self._filters['bandstop_mids'] = signal.butter(
            4, [1000, 3000], btype='bandstop', fs=self.sample_rate
        )
        
        # Notch filter for specific frequencies (e.g., 60Hz hum)
        self._filters['notch_60'] = signal.iirnotch(
            60, 30, fs=self.sample_rate
        )
    
    def apply_volume_attenuation(
        self,
        audio: NDArray[np.float32],
        target_factor: float,
        transition_samples: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """
        Apply volume attenuation with smooth transition.
        
        Args:
            audio: Input audio (samples x channels)
            target_factor: Target volume (0 = silent, 1 = unchanged)
            transition_samples: Samples over which to transition
            
        Returns:
            Volume-adjusted audio
        """
        # Enforce minimum volume
        target_factor = max(target_factor, self.min_volume)
        
        if transition_samples is None:
            transition_samples = len(audio)
        
        # Calculate safe transition
        max_delta = self.max_transition_rate * len(audio) / self.sample_rate
        actual_delta = target_factor - self._current_volume
        clamped_delta = np.clip(actual_delta, -max_delta, max_delta)
        
        safe_target = self._current_volume + clamped_delta
        
        # Create smooth gain ramp
        ramp = np.linspace(
            self._current_volume,
            safe_target,
            min(transition_samples, len(audio))
        )
        
        # Extend ramp if needed
        if len(ramp) < len(audio):
            ramp = np.concatenate([ramp, np.full(len(audio) - len(ramp), safe_target)])
        
        # Apply to each channel
        if audio.ndim == 1:
            result = audio * ramp
        else:
            result = audio * ramp.reshape(-1, 1)
        
        # Update current volume
        self._current_volume = safe_target
        self._target_volume = target_factor
        
        return result.astype(np.float32)
    
    def apply_selective_muting(
        self,
        audio: NDArray[np.float32],
        mute_factor: float = 0.0,
        preserve_ambient: bool = True,
    ) -> NDArray[np.float32]:
        """
        Apply selective muting with ambient preservation.
        
        Args:
            audio: Input audio
            mute_factor: 0 = unchanged, 1 = fully muted (to min_volume)
            preserve_ambient: Whether to preserve ambient sounds
            
        Returns:
            Muted audio
        """
        # Calculate target volume
        if preserve_ambient:
            target = 1.0 - mute_factor * (1.0 - self.min_volume)
        else:
            target = max(1.0 - mute_factor, self.min_volume)
        
        return self.apply_volume_attenuation(audio, target)
    
    def apply_frequency_filter(
        self,
        audio: NDArray[np.float32],
        filter_type: str,
        strength: float = 1.0,
    ) -> NDArray[np.float32]:
        """
        Apply frequency filtering.
        
        Args:
            audio: Input audio
            filter_type: 'lowpass_4k', 'highpass_200', 'bandstop_mids', 'notch_60'
            strength: Filter strength (0 = bypass, 1 = full filter)
            
        Returns:
            Filtered audio
        """
        if filter_type not in self._filters:
            logger.warning(f"Unknown filter type: {filter_type}")
            return audio
        
        if strength <= 0:
            return audio
        
        b, a = self._filters[filter_type]
        
        # Get or initialize filter state
        if filter_type not in self._filter_states:
            self._filter_states[filter_type] = FilterState()
        
        state = self._filter_states[filter_type]
        
        # Apply filter to each channel
        if audio.ndim == 1:
            if state.zi is None:
                state.zi = signal.lfilter_zi(b, a)
            
            filtered, state.zi = signal.lfilter(b, a, audio, zi=state.zi)
        else:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                key = f"{filter_type}_ch{ch}"
                if key not in self._filter_states:
                    self._filter_states[key] = FilterState()
                
                ch_state = self._filter_states[key]
                if ch_state.zi is None:
                    ch_state.zi = signal.lfilter_zi(b, a)
                
                filtered[:, ch], ch_state.zi = signal.lfilter(
                    b, a, audio[:, ch], zi=ch_state.zi
                )
        
        # Blend based on strength
        if strength < 1.0:
            result = audio * (1 - strength) + filtered * strength
        else:
            result = filtered
        
        return result.astype(np.float32)
    
    def apply_directional_dampening(
        self,
        audio: NDArray[np.float32],
        target_azimuth: float,
        dampening_strength: float,
        source_azimuth: float = 0.0,
        falloff_degrees: float = 30.0,
    ) -> NDArray[np.float32]:
        """
        Apply directional dampening based on spatial position.
        
        Args:
            audio: Stereo input audio
            target_azimuth: Direction to dampen (degrees, 0 = center)
            dampening_strength: 0 = no dampening, 1 = full dampening
            source_azimuth: Estimated source direction
            falloff_degrees: Angular width of dampening
            
        Returns:
            Spatially dampened audio
        """
        if audio.ndim < 2 or audio.shape[1] < 2:
            # Can't do spatial processing on mono
            return audio
        
        # Calculate angular distance from target
        angular_diff = abs(source_azimuth - target_azimuth)
        
        # Calculate attenuation based on angular proximity
        if angular_diff <= falloff_degrees:
            # Source is in dampening region
            proximity = 1.0 - (angular_diff / falloff_degrees)
            attenuation = 1.0 - (proximity * dampening_strength * (1.0 - self.min_volume))
        else:
            # Source is outside dampening region
            attenuation = 1.0
        
        return self.apply_volume_attenuation(audio, attenuation)
    
    def apply_transform(
        self,
        audio: NDArray[np.float32],
        operation: AudioOperation,
        parameters: TransformParameters,
        safety: SafetyConstraints,
    ) -> NDArray[np.float32]:
        """
        Apply a transformation based on operation type.
        
        Args:
            audio: Input audio
            operation: Type of operation
            parameters: Transform parameters
            safety: Safety constraints
            
        Returns:
            Transformed audio
        """
        if operation == AudioOperation.VOLUME_ATTENUATION:
            return self.apply_volume_attenuation(
                audio,
                parameters.volume_factor,
                int(parameters.transition_duration_seconds * self.sample_rate)
            )
        
        elif operation == AudioOperation.SELECTIVE_MUTING:
            mute_factor = 1.0 - parameters.volume_factor
            return self.apply_selective_muting(
                audio,
                mute_factor,
                preserve_ambient=True
            )
        
        elif operation == AudioOperation.FREQUENCY_FILTERING:
            filter_config = parameters.frequency_filter or {}
            filter_type = filter_config.get('type', 'lowpass_4k')
            strength = filter_config.get('strength', 0.5)
            return self.apply_frequency_filter(audio, filter_type, strength)
        
        elif operation == AudioOperation.DIRECTIONAL_DAMPENING:
            return self.apply_directional_dampening(
                audio,
                target_azimuth=0.0,  # Would come from intent resolution
                dampening_strength=parameters.directional_dampen,
            )
        
        else:
            logger.warning(f"Unknown audio operation: {operation}")
            return audio
    
    def reset(self):
        """Reset transformer state."""
        self._current_volume = 1.0
        self._target_volume = 1.0
        self._filter_states.clear()
        logger.debug("Audio transformer reset")



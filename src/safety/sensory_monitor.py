"""
Sensory Load Monitor.

Tracks cumulative sensory load from active transformations.
Triggers automatic dampening when threshold exceeded.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import numpy as np
from loguru import logger

from src.core.contracts import (
    TransformOperation,
    VisualOperation,
    AudioOperation,
    Modality,
)


@dataclass
class SensoryMetrics:
    """Current sensory load metrics."""
    visual_load: float = 0.0
    audio_load: float = 0.0
    total_load: float = 0.0
    
    # Rate of change metrics
    visual_change_rate: float = 0.0
    audio_change_rate: float = 0.0
    
    # Time since last major change
    time_since_visual_change: float = float('inf')
    time_since_audio_change: float = float('inf')


class SensoryLoadMonitor:
    """
    Monitors cumulative sensory load from transformations.
    
    The sensory load model considers:
    - Number of active transformations
    - Intensity of each transformation
    - Rate of change
    - Modality (visual vs audio)
    
    When load exceeds threshold:
    - Automatically dampen secondary stimuli
    - Bias toward predictability and smoothness
    """
    
    def __init__(
        self,
        load_threshold: float = 0.7,
        visual_weight: float = 1.0,
        audio_weight: float = 1.2,  # Audio changes are slightly more impactful
        decay_rate: float = 0.1,    # Load decay per second
        auto_dampen_enabled: bool = True,
    ):
        """
        Initialize sensory load monitor.
        
        Args:
            load_threshold: Threshold above which dampening is triggered
            visual_weight: Weight for visual load contribution
            audio_weight: Weight for audio load contribution
            decay_rate: Rate at which load naturally decays
            auto_dampen_enabled: Whether to automatically dampen on overload
        """
        self.load_threshold = load_threshold
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.decay_rate = decay_rate
        self.auto_dampen_enabled = auto_dampen_enabled
        
        # Current state
        self._current_metrics = SensoryMetrics()
        self._operation_loads: Dict[str, float] = {}  # operation_id -> load
        
        # History for rate calculation
        self._visual_history: List[tuple[float, float]] = []  # (timestamp, load)
        self._audio_history: List[tuple[float, float]] = []
        self._max_history = 100
        
        # Timestamps
        self._last_visual_change: float = 0.0
        self._last_audio_change: float = 0.0
        self._last_update: float = time.time()
    
    def update(
        self,
        active_operations: List[TransformOperation],
    ) -> SensoryMetrics:
        """
        Update sensory load based on active operations.
        
        Args:
            active_operations: Currently active transform operations
            
        Returns:
            Updated sensory metrics
        """
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time
        
        # Calculate load from active operations
        visual_load = 0.0
        audio_load = 0.0
        
        for op in active_operations:
            op_load = self._calculate_operation_load(op)
            
            if op.modality in [Modality.VISUAL, Modality.BOTH]:
                visual_load += op_load * self.visual_weight
            if op.modality in [Modality.AUDIO, Modality.BOTH]:
                audio_load += op_load * self.audio_weight
        
        # Apply decay to existing load
        previous_visual = self._current_metrics.visual_load
        previous_audio = self._current_metrics.audio_load
        
        decayed_visual = max(0, previous_visual - self.decay_rate * dt)
        decayed_audio = max(0, previous_audio - self.decay_rate * dt)
        
        # Combine new and decayed load
        self._current_metrics.visual_load = max(visual_load, decayed_visual)
        self._current_metrics.audio_load = max(audio_load, decayed_audio)
        self._current_metrics.total_load = (
            self._current_metrics.visual_load + 
            self._current_metrics.audio_load
        ) / 2.0
        
        # Calculate rate of change
        self._update_history(current_time)
        self._current_metrics.visual_change_rate = self._calculate_rate(self._visual_history)
        self._current_metrics.audio_change_rate = self._calculate_rate(self._audio_history)
        
        # Update time since last change
        self._current_metrics.time_since_visual_change = current_time - self._last_visual_change
        self._current_metrics.time_since_audio_change = current_time - self._last_audio_change
        
        # Check for significant changes
        if abs(visual_load - previous_visual) > 0.1:
            self._last_visual_change = current_time
        if abs(audio_load - previous_audio) > 0.1:
            self._last_audio_change = current_time
        
        return self._current_metrics
    
    def _calculate_operation_load(
        self,
        operation: TransformOperation,
    ) -> float:
        """
        Calculate sensory load contribution from a single operation.
        """
        # Base load from operation type
        base_load = self._get_operation_base_load(operation.operation)
        
        # Modulate by parameter intensity
        intensity = self._calculate_parameter_intensity(operation.parameters)
        
        # Modulate by transition progress (mid-transition = higher load)
        progress = operation.progress
        transition_factor = 1.0 - abs(progress - 0.5) * 2  # Peak at 0.5
        
        return base_load * intensity * (0.5 + 0.5 * transition_factor)
    
    def _get_operation_base_load(
        self,
        operation: VisualOperation | AudioOperation,
    ) -> float:
        """Get base load for an operation type."""
        # Visual operations
        if operation == VisualOperation.BRIGHTNESS_ATTENUATION:
            return 0.4  # Medium impact
        elif operation == VisualOperation.SATURATION_REDUCTION:
            return 0.3  # Lower impact
        elif operation == VisualOperation.COLOR_TEMPERATURE_SHIFT:
            return 0.25  # Lower impact
        elif operation == VisualOperation.EDGE_PRESERVING_BLUR:
            return 0.35  # Medium impact
        elif operation == VisualOperation.TEXTURE_SIMPLIFICATION:
            return 0.3
        
        # Audio operations
        elif operation == AudioOperation.SELECTIVE_MUTING:
            return 0.5  # Higher impact - sudden silence is jarring
        elif operation == AudioOperation.VOLUME_ATTENUATION:
            return 0.35
        elif operation == AudioOperation.FREQUENCY_FILTERING:
            return 0.3
        elif operation == AudioOperation.DIRECTIONAL_DAMPENING:
            return 0.4
        
        return 0.3  # Default
    
    def _calculate_parameter_intensity(
        self,
        params,  # TransformParameters
    ) -> float:
        """Calculate intensity from parameter values."""
        intensities = []
        
        # Brightness deviation from 1.0
        if params.brightness_factor != 1.0:
            intensities.append(abs(1.0 - params.brightness_factor))
        
        # Saturation deviation from 1.0
        if params.saturation_factor != 1.0:
            intensities.append(abs(1.0 - params.saturation_factor))
        
        # Volume deviation from 1.0
        if params.volume_factor != 1.0:
            intensities.append(abs(1.0 - params.volume_factor))
        
        # Blur intensity (normalized to 0-1 assuming max 30px)
        if params.blur_radius > 0:
            intensities.append(min(params.blur_radius / 30.0, 1.0))
        
        # Temperature shift magnitude
        if params.color_temperature_shift != 0:
            intensities.append(abs(params.color_temperature_shift))
        
        if not intensities:
            return 0.5
        
        return float(np.mean(intensities))
    
    def _update_history(self, current_time: float):
        """Update load history for rate calculation."""
        self._visual_history.append((current_time, self._current_metrics.visual_load))
        self._audio_history.append((current_time, self._current_metrics.audio_load))
        
        # Trim history
        cutoff = current_time - 5.0  # Keep 5 seconds
        self._visual_history = [(t, v) for t, v in self._visual_history if t > cutoff]
        self._audio_history = [(t, v) for t, v in self._audio_history if t > cutoff]
    
    def _calculate_rate(
        self,
        history: List[tuple[float, float]],
    ) -> float:
        """Calculate rate of change from history."""
        if len(history) < 2:
            return 0.0
        
        times = [t for t, _ in history]
        values = [v for _, v in history]
        
        if times[-1] - times[0] <= 0:
            return 0.0
        
        # Simple linear regression slope
        n = len(times)
        sum_t = sum(times)
        sum_v = sum(values)
        sum_tv = sum(t * v for t, v in history)
        sum_t2 = sum(t * t for t in times)
        
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0
        
        slope = (n * sum_tv - sum_t * sum_v) / denom
        return abs(slope)
    
    def should_dampen(self) -> bool:
        """Check if automatic dampening should be applied."""
        if not self.auto_dampen_enabled:
            return False
        return self._current_metrics.total_load > self.load_threshold
    
    def get_dampening_factor(self) -> float:
        """
        Get factor to dampen secondary stimuli.
        
        Returns:
            Factor from 0.5 (heavy dampening) to 1.0 (no dampening)
        """
        if not self.should_dampen():
            return 1.0
        
        # Calculate how much over threshold
        excess = self._current_metrics.total_load - self.load_threshold
        max_excess = 1.0 - self.load_threshold
        
        if max_excess <= 0:
            return 0.5
        
        # Scale dampening with excess
        dampen = 1.0 - (excess / max_excess) * 0.5
        return max(0.5, dampen)
    
    @property
    def current_load(self) -> float:
        """Get current total sensory load."""
        return self._current_metrics.total_load
    
    @property
    def metrics(self) -> SensoryMetrics:
        """Get current metrics."""
        return self._current_metrics
    
    def reset(self):
        """Reset monitor state."""
        self._current_metrics = SensoryMetrics()
        self._operation_loads.clear()
        self._visual_history.clear()
        self._audio_history.clear()
        self._last_update = time.time()
        logger.debug("Sensory load monitor reset")



"""
Safety Layer - Core Guardrails.

Enforces:
- Maximum parameter delta per second
- Minimum transition durations
- No sudden silence
- No sudden darkness

When constraints cannot be satisfied:
- Fail safely
- Log the failure
- Revert to unmodified passthrough
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import time
from loguru import logger

from src.core.contracts import (
    SafetyConstraints,
    TransformOperation,
    TransformParameters,
    VisualOperation,
    AudioOperation,
)


@dataclass
class ParameterHistory:
    """Tracks parameter changes over time."""
    parameter_name: str
    values: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    max_history: int = 100
    
    def add(self, value: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.values.append((timestamp, value))
        if len(self.values) > self.max_history:
            self.values.pop(0)
    
    def get_rate_of_change(self, window_seconds: float = 1.0) -> float:
        """Calculate rate of change over the last window_seconds."""
        if len(self.values) < 2:
            return 0.0
        
        current_time = time.time()
        cutoff = current_time - window_seconds
        
        recent = [(t, v) for t, v in self.values if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        
        time_diff = recent[-1][0] - recent[0][0]
        if time_diff <= 0:
            return 0.0
        
        value_diff = abs(recent[-1][1] - recent[0][1])
        return value_diff / time_diff


class SafetyLayer:
    """
    Core safety layer for sensory-safe operation.
    
    HARD CONSTRAINTS that CANNOT be violated:
    - All visual edits MUST originate from segmentation masks
    - All audio edits MUST be spatially or identity grounded
    - No global brightness, contrast, or volume changes unless explicitly requested
    - All changes MUST be temporally eased (no step functions)
    
    When uncertain:
    - Do less, not more
    - Ask for clarification
    - Preserve the user's baseline perception
    """
    
    def __init__(
        self,
        constraints: Optional[SafetyConstraints] = None,
    ):
        """
        Initialize safety layer.
        
        Args:
            constraints: Safety constraints (uses defaults if None)
        """
        self.constraints = constraints or SafetyConstraints()
        
        # Parameter history for rate limiting
        self._param_history: Dict[str, ParameterHistory] = {}
        
        # Emergency state
        self._emergency_revert_active: bool = False
        self._last_safe_state: Optional[Dict] = None
        
        # Violation tracking
        self._violation_count: int = 0
        self._last_violation_time: float = 0.0
        self._violation_log: List[str] = []
    
    def validate_operation(
        self,
        operation: TransformOperation,
    ) -> Tuple[bool, Optional[str], Optional[TransformOperation]]:
        """
        Validate a transform operation against safety constraints.
        
        Args:
            operation: Proposed operation
            
        Returns:
            Tuple of (is_valid, rejection_reason, constrained_operation)
            - is_valid: Whether operation can proceed
            - rejection_reason: If invalid, why
            - constrained_operation: If valid but constrained, the safe version
        """
        # Check if operation type is allowed
        if isinstance(operation.operation, VisualOperation):
            if operation.operation in [
                VisualOperation.OBJECT_REMOVAL,
                VisualOperation.GEOMETRY_DISTORTION,
            ]:
                return (
                    False,
                    f"Operation '{operation.operation.value}' is not allowed for safety reasons.",
                    None
                )
        
        # Check transition time
        if operation.transition_time_seconds < self.constraints.min_transition_seconds:
            # Constrain to minimum
            constrained = self._copy_operation(operation)
            constrained.transition_time_seconds = self.constraints.min_transition_seconds
            logger.info(
                f"Transition time constrained from {operation.transition_time_seconds}s "
                f"to {self.constraints.min_transition_seconds}s"
            )
            return (True, None, constrained)
        
        # Check parameter safety
        params_valid, param_reason, constrained_params = self._validate_parameters(
            operation.parameters
        )
        
        if not params_valid:
            return (False, param_reason, None)
        
        if constrained_params != operation.parameters:
            constrained = self._copy_operation(operation)
            constrained.parameters = constrained_params
            return (True, None, constrained)
        
        return (True, None, None)
    
    def _validate_parameters(
        self,
        params: TransformParameters,
    ) -> Tuple[bool, Optional[str], TransformParameters]:
        """
        Validate and potentially constrain transform parameters.
        """
        constrained = TransformParameters(
            brightness_factor=params.brightness_factor,
            saturation_factor=params.saturation_factor,
            color_temperature_shift=params.color_temperature_shift,
            blur_radius=params.blur_radius,
            texture_simplification=params.texture_simplification,
            volume_factor=params.volume_factor,
            frequency_filter=params.frequency_filter,
            directional_dampen=params.directional_dampen,
            transition_duration_seconds=params.transition_duration_seconds,
        )
        
        was_constrained = False
        
        # Brightness - never go below minimum
        if params.brightness_factor < self.constraints.min_brightness:
            constrained.brightness_factor = self.constraints.min_brightness
            was_constrained = True
            logger.info(
                f"Brightness constrained to minimum {self.constraints.min_brightness}"
            )
        
        # Volume - never go below minimum
        if params.volume_factor < self.constraints.min_volume:
            constrained.volume_factor = self.constraints.min_volume
            was_constrained = True
            logger.info(
                f"Volume constrained to minimum {self.constraints.min_volume}"
            )
        
        # Blur - cap at maximum
        if params.blur_radius > self.constraints.max_blur_radius:
            constrained.blur_radius = self.constraints.max_blur_radius
            was_constrained = True
            logger.info(
                f"Blur radius constrained to maximum {self.constraints.max_blur_radius}px"
            )
        
        # Transition duration - enforce minimum
        if params.transition_duration_seconds < self.constraints.min_transition_seconds:
            constrained.transition_duration_seconds = self.constraints.min_transition_seconds
            was_constrained = True
        
        return (True, None, constrained if was_constrained else params)
    
    def check_rate_limits(
        self,
        param_name: str,
        current_value: float,
        target_value: float,
        delta_time_seconds: float,
    ) -> Tuple[float, bool]:
        """
        Check if a parameter change respects rate limits.
        
        Args:
            param_name: Name of parameter
            current_value: Current value
            target_value: Desired target value
            delta_time_seconds: Time available for transition
            
        Returns:
            Tuple of (safe_target_value, was_constrained)
        """
        # Get max delta for this parameter
        max_delta_per_second = self._get_max_delta_for_param(param_name)
        
        if max_delta_per_second is None:
            # No rate limit for this parameter
            return (target_value, False)
        
        # Calculate maximum allowed change
        max_change = max_delta_per_second * delta_time_seconds
        actual_change = target_value - current_value
        
        if abs(actual_change) <= max_change:
            return (target_value, False)
        
        # Constrain to maximum allowed change
        if actual_change > 0:
            safe_target = current_value + max_change
        else:
            safe_target = current_value - max_change
        
        self._log_violation(
            f"Rate limit exceeded for {param_name}: "
            f"requested {actual_change:.3f}, allowed {max_change:.3f}"
        )
        
        return (safe_target, True)
    
    def _get_max_delta_for_param(self, param_name: str) -> Optional[float]:
        """Get maximum change rate for a parameter."""
        rates = {
            'brightness': self.constraints.max_brightness_delta,
            'brightness_factor': self.constraints.max_brightness_delta,
            'volume': self.constraints.max_volume_delta,
            'volume_factor': self.constraints.max_volume_delta,
            'saturation': self.constraints.max_saturation_delta,
            'saturation_factor': self.constraints.max_saturation_delta,
            'blur': self.constraints.max_blur_delta,
            'blur_radius': self.constraints.max_blur_delta,
        }
        return rates.get(param_name)
    
    def trigger_emergency_revert(self, reason: str):
        """
        Trigger emergency revert to unmodified passthrough.
        
        This is called when:
        - User presses ESC
        - Multiple constraint violations in short time
        - System detects unsafe state
        """
        self._emergency_revert_active = True
        self._log_violation(f"EMERGENCY REVERT: {reason}")
        logger.warning(f"Emergency revert triggered: {reason}")
    
    def clear_emergency_revert(self):
        """Clear emergency revert state."""
        self._emergency_revert_active = False
        logger.info("Emergency revert cleared")
    
    @property
    def is_emergency_revert_active(self) -> bool:
        return self._emergency_revert_active
    
    def update_sensory_load(self, load: float):
        """
        Update current sensory load metric.
        
        If load exceeds threshold, automatic dampening may be applied.
        """
        self.constraints.current_sensory_load = load
        
        if load > self.constraints.sensory_load_threshold:
            logger.warning(
                f"Sensory load ({load:.2f}) exceeds threshold "
                f"({self.constraints.sensory_load_threshold})"
            )
    
    def _log_violation(self, message: str):
        """Log a constraint violation."""
        self._violation_count += 1
        self._last_violation_time = time.time()
        self._violation_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        
        # Keep log size manageable
        if len(self._violation_log) > 100:
            self._violation_log.pop(0)
        
        logger.warning(f"Safety violation: {message}")
        
        # Check for violation storm
        if self._violation_count > 10:
            recent_violations = sum(
                1 for t in [self._last_violation_time]
                if time.time() - t < 5.0
            )
            if recent_violations > 5:
                self.trigger_emergency_revert("Too many violations in short time")
    
    def _copy_operation(
        self,
        operation: TransformOperation,
    ) -> TransformOperation:
        """Create a copy of an operation."""
        return TransformOperation(
            operation_id=operation.operation_id,
            target_ids=operation.target_ids.copy(),
            modality=operation.modality,
            operation=operation.operation,
            parameters=operation.parameters,
            transition_time_seconds=operation.transition_time_seconds,
            start_timestamp=operation.start_timestamp,
            is_active=operation.is_active,
            progress=operation.progress,
            original_state=operation.original_state,
        )
    
    def get_violation_log(self) -> List[str]:
        """Get recent violation log."""
        return self._violation_log.copy()
    
    def reset(self):
        """Reset safety layer state."""
        self._param_history.clear()
        self._emergency_revert_active = False
        self._violation_count = 0
        self._violation_log.clear()
        logger.info("Safety layer reset")



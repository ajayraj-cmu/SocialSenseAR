"""
Natural Language Intent Parser.

Converts user commands into structured transformation operations.

Command format understanding:
- target: object, person, region
- modality: audio or visual
- attribute: brightness, color, volume, texture
- magnitude: slightly, much, a lot
- directionality: left/right/near/far
"""

from __future__ import annotations

import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger

from src.core.contracts import (
    UserIntent,
    TransformOperation,
    TransformParameters,
    Modality,
    VisualOperation,
    AudioOperation,
    SpatialRelation,
    ObjectClass,
    Magnitude,
)


# ============================================================
# INTENT PATTERNS
# ============================================================

# Magnitude patterns
MAGNITUDE_PATTERNS = {
    Magnitude.SLIGHT: r'\b(slightly|a little|a bit|somewhat|mild)\b',
    Magnitude.MODERATE: r'\b(moderately|some|medium)\b',
    Magnitude.STRONG: r'\b(much|a lot|significantly|strongly|greatly)\b',
    Magnitude.MAXIMUM: r'\b(completely|fully|entirely|maximum|totally|all the way)\b',
}

# Spatial relation patterns
SPATIAL_PATTERNS = {
    SpatialRelation.LEFT: r'\b(left|to my left|on the left)\b',
    SpatialRelation.RIGHT: r'\b(right|to my right|on the right)\b',
    SpatialRelation.FRONT: r'\b(front|in front|ahead|forward)\b',
    SpatialRelation.BEHIND: r'\b(behind|back|backward)\b',
    SpatialRelation.NEAR: r'\b(near|close|nearby|closest)\b',
    SpatialRelation.FAR: r'\b(far|distant|farthest|furthest)\b',
    SpatialRelation.CENTER: r'\b(center|middle|central)\b',
    SpatialRelation.ABOVE: r'\b(above|up|upper|top)\b',
    SpatialRelation.BELOW: r'\b(below|down|lower|bottom)\b',
}

# Object class patterns
CLASS_PATTERNS = {
    ObjectClass.PERSON: r'\b(person|people|someone|human|man|woman|guy|girl)\b',
    ObjectClass.FACE: r'\b(face|faces)\b',
    ObjectClass.SCREEN: r'\b(screen|monitor|display|tv|television)\b',
    ObjectClass.LIGHT_SOURCE: r'\b(light|lamp|bulb|sun|window)\b',
    ObjectClass.VEHICLE: r'\b(car|vehicle|truck|bike|bicycle)\b',
    ObjectClass.ANIMAL: r'\b(animal|dog|cat|pet|bird)\b',
    ObjectClass.TEXT: r'\b(text|sign|label|words)\b',
    ObjectClass.HAND: r'\b(hand|hands|finger|fingers)\b',
}

# Operation patterns - MORE FLEXIBLE
VISUAL_OP_PATTERNS = {
    VisualOperation.BRIGHTNESS_ATTENUATION: r'\b(dim|darken|brighten|lighter|darker|brightness|dark|black|shadow)\b',
    VisualOperation.SATURATION_REDUCTION: r'\b(desaturate|grayscale|gray|grey|saturation|muted|black and white|monochrome)\b',
    VisualOperation.COLOR_TEMPERATURE_SHIFT: r'\b(color|warmer|cooler|warm|cool|temperature|yellow|orange|blue|red|green|pink|purple|tint|shade|change.{0,10}color|make.{0,10}(yellow|orange|blue|red|green|pink|purple))\b',
    VisualOperation.EDGE_PRESERVING_BLUR: r'\b(blur|blurry|soft|soften|smooth|fuzzy|out of focus|unfocus|mask|hide|obscure|cover)\b',
    VisualOperation.TEXTURE_SIMPLIFICATION: r'\b(simplify|simple|reduce detail|less detail|clean|cartoon|pixelate|pixel)\b',
}

# Color name to RGB mapping
COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'pink': (255, 105, 180),
    'purple': (128, 0, 128),
    'cyan': (0, 255, 255),
    'white': (255, 255, 255),
    'neon': (57, 255, 20),
}

AUDIO_OP_PATTERNS = {
    AudioOperation.SELECTIVE_MUTING: r'\b(mute|silence|quiet|hush|shut up|stop sound)\b',
    AudioOperation.VOLUME_ATTENUATION: r'\b(volume|louder|quieter|soften|lower|reduce sound)\b',
    AudioOperation.FREQUENCY_FILTERING: r'\b(filter|bass|treble|harsh|frequency)\b',
    AudioOperation.DIRECTIONAL_DAMPENING: r'\b(directional|from that direction|that side)\b',
}

# Unsafe operation patterns
UNSAFE_PATTERNS = [
    r'\b(remove|delete|hide|invisible|gone|disappear)\b',  # Object removal
    r'\b(flash|flashing|strobe|blink|rapid)\b',  # Photosensitivity risk
    r'\b(distort|warp|twist|morph)\b',  # Geometry distortion
    r'\b(black|blackout|dark|pitch)\s+(everything|all|screen)\b',  # Full darkness
]


class IntentParser:
    """
    Parser for natural language user commands.
    
    Converts commands like:
        "mute the person to my right"
        "dim the screen on the left"
        "make everything slightly less saturated"
    
    Into structured TransformOperation objects.
    """
    
    def __init__(
        self,
        require_confirmation_for_unsafe: bool = True,
        max_targets_per_command: int = 5,
    ):
        """
        Initialize intent parser.
        
        Args:
            require_confirmation_for_unsafe: Require confirmation for unsafe ops
            max_targets_per_command: Maximum targets per single command
        """
        self.require_confirmation_for_unsafe = require_confirmation_for_unsafe
        self.max_targets_per_command = max_targets_per_command
        
        # Compile regex patterns
        self._magnitude_re = {k: re.compile(v, re.I) for k, v in MAGNITUDE_PATTERNS.items()}
        self._spatial_re = {k: re.compile(v, re.I) for k, v in SPATIAL_PATTERNS.items()}
        self._class_re = {k: re.compile(v, re.I) for k, v in CLASS_PATTERNS.items()}
        self._visual_op_re = {k: re.compile(v, re.I) for k, v in VISUAL_OP_PATTERNS.items()}
        self._audio_op_re = {k: re.compile(v, re.I) for k, v in AUDIO_OP_PATTERNS.items()}
        self._unsafe_re = [re.compile(p, re.I) for p in UNSAFE_PATTERNS]
    
    def parse(self, command: str) -> UserIntent:
        """
        Parse a natural language command into structured intent.
        
        Args:
            command: User's natural language command
            
        Returns:
            UserIntent with parsed information
        """
        command = command.strip()
        
        if not command:
            return UserIntent(
                raw_command=command,
                target_description="",
                is_valid=False,
                rejection_reason="Empty command",
            )
        
        # Check for unsafe patterns
        safety_result = self._check_safety(command)
        if not safety_result[0]:
            return UserIntent(
                raw_command=command,
                target_description="",
                is_valid=False,
                is_safe=False,
                rejection_reason=safety_result[1],
            )
        
        # Parse components
        magnitude = self._extract_magnitude(command)
        spatial_relation = self._extract_spatial(command)
        object_class = self._extract_object_class(command)
        modality, operation = self._extract_operation(command)
        
        # Build target description
        target_parts = []
        if object_class:
            target_parts.append(object_class.value)
        if spatial_relation:
            target_parts.append(f"to my {spatial_relation.value}")
        target_description = " ".join(target_parts) if target_parts else "unspecified target"
        
        # Check if we need clarification - BE MORE LENIENT
        needs_clarification = False
        clarification_prompt = None
        
        if not operation:
            # Try to guess operation from context
            if 'everything' in command or 'all' in command or 'whole' in command:
                operation = VisualOperation.EDGE_PRESERVING_BLUR
                modality = Modality.VISUAL
            else:
                needs_clarification = True
                clarification_prompt = "What would you like me to do? (e.g., dim, mute, blur)"
        
        # DON'T require a target - apply to whole screen if not specified
        # This is the key fix - allow global effects
        if not object_class and not spatial_relation:
            target_description = "whole screen"
            # Mark as global effect
        
        return UserIntent(
            raw_command=command,
            target_description=target_description,
            target_spatial_relation=spatial_relation,
            target_class=object_class,
            modality=modality or Modality.VISUAL,
            operation_type=operation.value if operation else None,
            magnitude=magnitude,
            is_valid=not needs_clarification,
            is_safe=True,
            requires_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
        )
    
    def _check_safety(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if command contains unsafe patterns.
        
        Returns:
            Tuple of (is_safe, rejection_reason)
        """
        for pattern in self._unsafe_re:
            if pattern.search(command):
                return (
                    False,
                    "This operation is not allowed for safety reasons. "
                    "Object removal, flashing effects, and full darkness are disabled."
                )
        return (True, None)
    
    def _extract_magnitude(self, command: str) -> Magnitude:
        """Extract magnitude from command."""
        for magnitude, pattern in self._magnitude_re.items():
            if pattern.search(command):
                return magnitude
        return Magnitude.MODERATE  # Default
    
    def _extract_spatial(self, command: str) -> Optional[SpatialRelation]:
        """Extract spatial relation from command."""
        for relation, pattern in self._spatial_re.items():
            if pattern.search(command):
                return relation
        return None
    
    def _extract_object_class(self, command: str) -> Optional[ObjectClass]:
        """Extract object class from command."""
        for obj_class, pattern in self._class_re.items():
            if pattern.search(command):
                return obj_class
        return None
    
    def _extract_operation(
        self,
        command: str,
    ) -> Tuple[Optional[Modality], Optional[VisualOperation | AudioOperation]]:
        """
        Extract operation type from command.
        
        Returns:
            Tuple of (modality, operation)
        """
        # Check audio operations first (they're more specific)
        for op, pattern in self._audio_op_re.items():
            if pattern.search(command):
                return (Modality.AUDIO, op)
        
        # Check visual operations
        for op, pattern in self._visual_op_re.items():
            if pattern.search(command):
                return (Modality.VISUAL, op)
        
        return (None, None)
    
    def intent_to_operation(
        self,
        intent: UserIntent,
        resolved_object_ids: List[str],
    ) -> Optional[TransformOperation]:
        """
        Convert a resolved intent into a TransformOperation.
        
        Args:
            intent: Parsed user intent
            resolved_object_ids: Object IDs resolved from target description
            
        Returns:
            TransformOperation or None if invalid
        """
        if not intent.is_valid or not resolved_object_ids:
            return None
        
        # Build parameters based on operation type
        params = self._build_parameters(intent)
        
        # Determine operation enum
        operation = self._get_operation_enum(intent)
        if not operation:
            return None
        
        import uuid
        
        return TransformOperation(
            operation_id=str(uuid.uuid4())[:8],
            target_ids=resolved_object_ids[:self.max_targets_per_command],
            modality=intent.modality,
            operation=operation,
            parameters=params,
            transition_time_seconds=0.5,  # Default transition
            is_active=True,
            progress=0.0,
        )
    
    def _build_parameters(self, intent: UserIntent) -> TransformParameters:
        """Build transform parameters from intent."""
        magnitude_value = intent.magnitude.value
        
        params = TransformParameters()
        
        if intent.modality == Modality.AUDIO:
            if intent.operation_type == AudioOperation.SELECTIVE_MUTING.value:
                params.volume_factor = 1.0 - magnitude_value * 0.9  # Keep min 10%
            elif intent.operation_type == AudioOperation.VOLUME_ATTENUATION.value:
                params.volume_factor = 1.0 - magnitude_value * 0.5
        
        elif intent.modality == Modality.VISUAL:
            if intent.operation_type == VisualOperation.BRIGHTNESS_ATTENUATION.value:
                params.brightness_factor = 1.0 - magnitude_value * 0.7
            elif intent.operation_type == VisualOperation.SATURATION_REDUCTION.value:
                params.saturation_factor = 1.0 - magnitude_value * 0.8
            elif intent.operation_type == VisualOperation.EDGE_PRESERVING_BLUR.value:
                params.blur_radius = magnitude_value * 25  # Max 25px blur
            elif intent.operation_type == VisualOperation.TEXTURE_SIMPLIFICATION.value:
                params.texture_simplification = magnitude_value
            elif intent.operation_type == VisualOperation.COLOR_TEMPERATURE_SHIFT.value:
                # Extract color from command
                color_shift = self._extract_color_shift(intent.raw_command)
                params.color_temperature_shift = color_shift
        
        return params
    
    def _extract_color_shift(self, command: str) -> float:
        """Extract color temperature shift from command."""
        command_lower = command.lower()
        
        # Warm colors = positive shift
        if any(c in command_lower for c in ['yellow', 'orange', 'warm', 'red']):
            return 0.8
        # Cool colors = negative shift
        elif any(c in command_lower for c in ['blue', 'cool', 'cyan']):
            return -0.8
        # Other colors
        elif 'green' in command_lower:
            return 0.3
        elif 'pink' in command_lower or 'purple' in command_lower:
            return -0.4
        
        return 0.5  # Default moderate warm shift
    
    def _get_operation_enum(
        self,
        intent: UserIntent,
    ) -> Optional[VisualOperation | AudioOperation]:
        """Get operation enum from intent."""
        if not intent.operation_type:
            return None
        
        # Try visual operations
        for op in VisualOperation:
            if op.value == intent.operation_type:
                return op
        
        # Try audio operations
        for op in AudioOperation:
            if op.value == intent.operation_type:
                return op
        
        return None



"""
Core pipeline components for the Perceptual Modulation Engine.

Pipeline execution order (NEVER REORDER):
1. Acquire synchronized RGB + audio
2. Segment objects using SAM-3
3. Assign persistent object IDs via tracking
4. Estimate depth (hardware if available, MiDaS fallback)
5. Bind audio sources to visual entities
6. Parse user intent (language → structured ops)
7. Resolve intent to object IDs
8. Apply parameterized audio/visual transformations
9. Enforce safety constraints and temporal smoothing
10. Render output streams
"""

from .contracts import (
    SegmentedObject,
    TransformOperation,
    AudioVisualBinding,
    SafetyConstraints,
    PipelineState,
)



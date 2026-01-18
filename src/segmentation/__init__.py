"""
Segmentation module using SAM-3 (Segment Anything Model).

Responsibilities:
- Real-time object segmentation
- Mask temporal smoothing
- Confidence scoring
- Morphological cleaning
"""

from .sam_segmenter import SAMSegmenter
from .mask_processor import MaskProcessor



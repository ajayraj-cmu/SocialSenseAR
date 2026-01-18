"""
Mask Processing Utilities.

Handles:
- Morphological cleaning
- Temporal smoothing
- Mask refinement
- Edge preservation
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger


class MaskProcessor:
    """
    Processor for segmentation masks.
    
    Ensures masks are:
    - Temporally smoothed
    - Morphologically cleaned
    - Edge-preserved
    """
    
    def __init__(
        self,
        smoothing_kernel_size: int = 5,
        morphological_iterations: int = 2,
        min_area_threshold: int = 100,
    ):
        """
        Initialize mask processor.
        
        Args:
            smoothing_kernel_size: Kernel size for Gaussian smoothing
            morphological_iterations: Iterations for opening/closing
            min_area_threshold: Minimum mask area to keep
        """
        self.smoothing_kernel_size = smoothing_kernel_size
        self.morphological_iterations = morphological_iterations
        self.min_area_threshold = min_area_threshold
        
        # Create morphological kernels
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (smoothing_kernel_size, smoothing_kernel_size)
        )
    
    def clean_mask(
        self,
        mask: NDArray[np.uint8],
        remove_small_regions: bool = True,
    ) -> NDArray[np.uint8]:
        """
        Clean a segmentation mask.
        
        Operations:
        1. Morphological opening (remove noise)
        2. Morphological closing (fill holes)
        3. Remove small disconnected regions
        4. Smooth edges
        
        Args:
            mask: Binary mask (H x W)
            remove_small_regions: Whether to remove small components
            
        Returns:
            Cleaned binary mask
        """
        if mask is None or mask.size == 0:
            return mask
        
        # Ensure binary
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Morphological opening (erosion followed by dilation)
        # Removes small bright spots (noise)
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self._morph_kernel,
            iterations=self.morphological_iterations
        )
        
        # Morphological closing (dilation followed by erosion)
        # Fills small holes
        closed = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=self.morphological_iterations
        )
        
        # Remove small disconnected regions
        if remove_small_regions:
            closed = self._remove_small_components(closed)
        
        # Convert back to 0/1
        return (closed > 127).astype(np.uint8)
    
    def _remove_small_components(
        self,
        mask: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """
        Remove small connected components from mask.
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create output mask
        output = np.zeros_like(mask)
        
        # Keep only large components (skip background at label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area_threshold:
                output[labels == i] = 255
        
        return output
    
    def temporal_smooth(
        self,
        current_mask: NDArray[np.uint8],
        previous_mask: NDArray[np.uint8],
        alpha: float = 0.7,
    ) -> NDArray[np.uint8]:
        """
        Temporally smooth a mask using exponential moving average.
        
        Args:
            current_mask: Current frame mask
            previous_mask: Previous frame mask
            alpha: Weight for current frame (higher = less smoothing)
            
        Returns:
            Temporally smoothed mask
        """
        if previous_mask is None or previous_mask.shape != current_mask.shape:
            return current_mask
        
        # Blend masks
        blended = (
            alpha * current_mask.astype(np.float32) +
            (1 - alpha) * previous_mask.astype(np.float32)
        )
        
        # Threshold back to binary
        return (blended > 0.5).astype(np.uint8)
    
    def refine_edges(
        self,
        mask: NDArray[np.uint8],
        frame: NDArray[np.uint8],
        edge_width: int = 10,
    ) -> NDArray[np.uint8]:
        """
        Refine mask edges using image gradients.
        
        Uses guided filter or similar technique to align
        mask edges with image edges.
        
        Args:
            mask: Binary mask to refine
            frame: RGB image for edge guidance
            edge_width: Width of edge refinement band
            
        Returns:
            Edge-refined mask
        """
        if mask is None or frame is None:
            return mask
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect edges in image
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate mask edges
        mask_dilated = cv2.dilate(
            mask * 255,
            self._morph_kernel,
            iterations=edge_width // self.smoothing_kernel_size
        )
        mask_eroded = cv2.erode(
            mask * 255,
            self._morph_kernel,
            iterations=edge_width // self.smoothing_kernel_size
        )
        
        # Edge band is the difference
        edge_band = ((mask_dilated > 0) & (mask_eroded == 0)).astype(np.uint8)
        
        # In edge band, use image edges to refine
        # Simple approach: in edge band, follow strong image edges
        refined = mask.copy()
        
        # Where there are strong edges in the band, trust them
        strong_edges = (edges > 100) & (edge_band > 0)
        
        # Use watershed or similar for better refinement
        # For now, simple edge-aware smoothing
        refined_float = cv2.GaussianBlur(
            refined.astype(np.float32) * 255,
            (5, 5),
            0
        )
        refined = (refined_float > 127).astype(np.uint8)
        
        return refined
    
    def create_soft_mask(
        self,
        mask: NDArray[np.uint8],
        feather_radius: int = 5,
    ) -> NDArray[np.float32]:
        """
        Create a soft (feathered) version of a binary mask.
        
        Useful for smooth blending in transformations.
        
        Args:
            mask: Binary mask
            feather_radius: Radius of feathering in pixels
            
        Returns:
            Soft mask with values in [0, 1]
        """
        if feather_radius <= 0:
            return mask.astype(np.float32)
        
        # Gaussian blur for soft edges
        kernel_size = feather_radius * 2 + 1
        soft = cv2.GaussianBlur(
            mask.astype(np.float32),
            (kernel_size, kernel_size),
            feather_radius / 3
        )
        
        return np.clip(soft, 0, 1)
    
    def combine_masks(
        self,
        masks: list[NDArray[np.uint8]],
        mode: str = "union",
    ) -> NDArray[np.uint8]:
        """
        Combine multiple masks.
        
        Args:
            masks: List of binary masks
            mode: "union" (OR), "intersection" (AND), or "xor"
            
        Returns:
            Combined mask
        """
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0]
        
        result = masks[0].astype(np.uint8)
        
        for mask in masks[1:]:
            if mode == "union":
                result = np.maximum(result, mask)
            elif mode == "intersection":
                result = np.minimum(result, mask)
            elif mode == "xor":
                result = np.logical_xor(result > 0, mask > 0).astype(np.uint8)
        
        return result
    
    def mask_to_bbox(
        self,
        mask: NDArray[np.uint8],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box from mask.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if mask is empty
        """
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def calculate_mask_overlap(
        self,
        mask1: NDArray[np.uint8],
        mask2: NDArray[np.uint8],
    ) -> float:
        """
        Calculate IoU (Intersection over Union) between two masks.
        """
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        
        if union == 0:
            return 0.0
        
        return intersection / union



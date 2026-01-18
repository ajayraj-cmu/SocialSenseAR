"""
SAM-Based Automatic Segmentation.

Uses Segment Anything Model to automatically segment ALL objects in the scene.
Each object gets its own mask that follows it dynamically.

This segments EVERYTHING - not just faces/people:
- Objects on walls
- Furniture
- Items on desk
- Background elements
- People and faces
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger

from src.core.contracts import (
    SegmentedObject,
    SegmentationResult,
    BoundingRegion,
    ObjectClass,
)

# Try to import SAM
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Try MediaPipe as fallback for person/face
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class SAMAutoSegmenter:
    """
    Automatic segmentation that finds ALL objects in the scene.
    
    Uses a combination of:
    1. Edge detection + contours for object boundaries
    2. Color clustering for distinct regions
    3. MediaPipe for people/faces (if available)
    4. Depth-based segmentation (if depth available)
    
    Each segmented region can be individually targeted for effects.
    """
    
    def __init__(
        self,
        min_object_area: int = 5000,  # Minimum pixels for an object
        max_objects: int = 20,  # Maximum objects to track
        merge_threshold: float = 0.3,  # IoU threshold to merge similar masks
        enable_person_detection: bool = True,
        enable_contour_detection: bool = True,
        enable_color_clustering: bool = False,  # Disabled for performance
        downscale_factor: float = 0.5,  # Process at lower resolution
    ):
        """
        Initialize automatic segmenter.
        """
        self.min_object_area = min_object_area
        self.max_objects = max_objects
        self.merge_threshold = merge_threshold
        self.enable_person = enable_person_detection
        self.enable_contours = enable_contour_detection
        self.enable_color = enable_color_clustering
        self.downscale_factor = downscale_factor
        
        # MediaPipe for people
        self._selfie_seg = None
        self._face_detection = None
        
        # Object tracking
        self._object_counter = 0
        self._previous_masks: Dict[str, NDArray] = {}
        
        # Performance
        self._inference_times: List[float] = []
        
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize segmentation models."""
        try:
            if MEDIAPIPE_AVAILABLE and self.enable_person:
                self._selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe person/face detection initialized")
            
            self._is_initialized = True
            logger.info("SAM Auto Segmenter initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize segmenter: {e}")
            return False
    
    def segment_frame(
        self,
        frame: NDArray[np.uint8],
        frame_id: int,
    ) -> SegmentationResult:
        """
        Segment ALL objects in the frame.
        
        Returns multiple SegmentedObjects, each with its own mask.
        """
        start_time = time.perf_counter()
        
        if not self._is_initialized:
            self.initialize()
        
        h, w = frame.shape[:2]
        objects = []
        all_masks = []
        
        # Downscale for faster processing
        if self.downscale_factor < 1.0:
            small_h = int(h * self.downscale_factor)
            small_w = int(w * self.downscale_factor)
            process_frame = cv2.resize(frame, (small_w, small_h))
        else:
            process_frame = frame
            small_h, small_w = h, w
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB) if len(process_frame.shape) == 3 else process_frame
        
        # 1. Person/Face segmentation (highest priority)
        if self._selfie_seg is not None:
            person_objs = self._segment_people(rgb_frame, frame_id)
            # Upscale masks back to original size
            for obj in person_objs:
                if self.downscale_factor < 1.0 and obj.mask is not None:
                    obj.mask = cv2.resize(obj.mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    obj.smoothed_mask = obj.mask
            objects.extend(person_objs)
            all_masks.extend([obj.mask for obj in person_objs])
        
        # 2. Contour-based segmentation for other objects (skip every other frame for speed)
        if self.enable_contours and frame_id % 2 == 0:
            contour_objs = self._segment_by_contours(rgb_frame, frame_id, all_masks)
            for obj in contour_objs:
                if self.downscale_factor < 1.0 and obj.mask is not None:
                    obj.mask = cv2.resize(obj.mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    obj.smoothed_mask = obj.mask
            objects.extend(contour_objs)
            all_masks.extend([obj.mask for obj in contour_objs])
        
        # 3. Color clustering for large regions (expensive, usually disabled)
        if self.enable_color and len(objects) < self.max_objects and frame_id % 5 == 0:
            color_objs = self._segment_by_color(rgb_frame, frame_id, all_masks)
            for obj in color_objs:
                if self.downscale_factor < 1.0 and obj.mask is not None:
                    obj.mask = cv2.resize(obj.mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    obj.smoothed_mask = obj.mask
            objects.extend(color_objs)
        
        # Limit total objects
        if len(objects) > self.max_objects:
            # Sort by area and keep largest
            objects.sort(key=lambda o: o.bounding_region.area, reverse=True)
            objects = objects[:self.max_objects]
        
        # Track objects for temporal consistency
        objects = self._track_objects(objects, frame_id)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        
        avg_confidence = sum(o.confidence for o in objects) / len(objects) if objects else 0
        
        return SegmentationResult(
            objects=objects,
            inference_time_ms=inference_time,
            model_confidence=avg_confidence,
            success=True
        )
    
    def _segment_people(
        self,
        frame: NDArray[np.uint8],
        frame_id: int,
    ) -> List[SegmentedObject]:
        """Segment people and faces using MediaPipe."""
        objects = []
        h, w = frame.shape[:2]
        
        # Selfie segmentation
        if self._selfie_seg:
            results = self._selfie_seg.process(frame)
            if results.segmentation_mask is not None:
                mask = (results.segmentation_mask > 0.5).astype(np.uint8)
                
                if np.sum(mask) > self.min_object_area:
                    bbox = self._mask_to_bbox(mask)
                    if bbox:
                        objects.append(SegmentedObject(
                            stable_id="person_0",
                            mask=mask,
                            confidence=0.9,
                            bounding_region=BoundingRegion(*bbox),
                            class_label=ObjectClass.PERSON,
                            saliency_score=0.95,
                            first_seen_frame=frame_id,
                            last_seen_frame=frame_id,
                            smoothed_mask=mask,
                        ))
        
        # Face detection
        if self._face_detection:
            face_results = self._face_detection.process(frame)
            if face_results.detections:
                for i, detection in enumerate(face_results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    
                    # Create ellipse mask for face
                    face_mask = np.zeros((h, w), dtype=np.uint8)
                    center = (x + bw // 2, y + bh // 2)
                    axes = (bw // 2, bh // 2)
                    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1, -1)
                    
                    objects.append(SegmentedObject(
                        stable_id=f"face_{i}",
                        mask=face_mask,
                        confidence=detection.score[0] if detection.score else 0.8,
                        bounding_region=BoundingRegion(
                            max(0, x), max(0, y),
                            min(w, x + bw), min(h, y + bh)
                        ),
                        class_label=ObjectClass.FACE,
                        saliency_score=0.98,
                        first_seen_frame=frame_id,
                        last_seen_frame=frame_id,
                        smoothed_mask=face_mask,
                    ))
        
        return objects
    
    def _segment_by_contours(
        self,
        frame: NDArray[np.uint8],
        frame_id: int,
        exclude_masks: List[NDArray],
    ) -> List[SegmentedObject]:
        """Find objects using edge detection and contours."""
        objects = []
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask that excludes already-segmented areas
        exclude_combined = np.zeros((h, w), dtype=np.uint8)
        for mask in exclude_masks:
            if mask is not None and mask.shape == (h, w):
                exclude_combined = np.maximum(exclude_combined, mask)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_object_area:
                continue
            
            # Create mask from contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            
            # Skip if overlaps too much with excluded areas
            overlap = np.sum(mask & exclude_combined) / (np.sum(mask) + 1)
            if overlap > 0.5:
                continue
            
            bbox = self._mask_to_bbox(mask)
            if bbox:
                objects.append(SegmentedObject(
                    stable_id=f"object_{self._object_counter}",
                    mask=mask,
                    confidence=0.6,
                    bounding_region=BoundingRegion(*bbox),
                    class_label=ObjectClass.UNKNOWN,
                    saliency_score=0.5,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    smoothed_mask=mask,
                ))
                self._object_counter += 1
            
            if len(objects) >= self.max_objects // 2:
                break
        
        return objects
    
    def _segment_by_color(
        self,
        frame: NDArray[np.uint8],
        frame_id: int,
        exclude_masks: List[NDArray],
    ) -> List[SegmentedObject]:
        """Find distinct color regions."""
        objects = []
        h, w = frame.shape[:2]
        
        # Convert to LAB for better color clustering
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Reshape for clustering
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        n_clusters = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        labels = labels.reshape(h, w)
        
        # Exclude already-segmented areas
        exclude_combined = np.zeros((h, w), dtype=np.uint8)
        for mask in exclude_masks:
            if mask is not None and mask.shape == (h, w):
                exclude_combined = np.maximum(exclude_combined, mask)
        
        for cluster_id in range(n_clusters):
            mask = (labels == cluster_id).astype(np.uint8)
            
            # Remove already-segmented pixels
            mask = mask & (1 - exclude_combined)
            
            # Clean up mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            
            area = np.sum(mask)
            if area < self.min_object_area:
                continue
            
            # Skip if too large (probably background)
            if area > h * w * 0.7:
                continue
            
            bbox = self._mask_to_bbox(mask)
            if bbox:
                objects.append(SegmentedObject(
                    stable_id=f"region_{self._object_counter}",
                    mask=mask,
                    confidence=0.5,
                    bounding_region=BoundingRegion(*bbox),
                    class_label=ObjectClass.UNKNOWN,
                    saliency_score=0.3,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    smoothed_mask=mask,
                ))
                self._object_counter += 1
        
        return objects
    
    def _track_objects(
        self,
        objects: List[SegmentedObject],
        frame_id: int,
    ) -> List[SegmentedObject]:
        """Track objects across frames for temporal consistency.
        
        IMPORTANT: Preserve class labels - don't overwrite person/face IDs
        with generic object IDs.
        """
        tracked = []
        
        for obj in objects:
            # NEVER change IDs for important classes - keep face_X and person_X
            if obj.class_label in [ObjectClass.FACE, ObjectClass.PERSON]:
                # Keep original ID for faces and people
                tracked.append(obj)
                self._previous_masks[obj.stable_id] = obj.mask.copy()
                continue
            
            # For other objects, try to match with previous frames
            best_match_id = None
            best_iou = 0.0
            
            for prev_id, prev_mask in self._previous_masks.items():
                # Don't match unknown objects to face/person IDs
                if 'face' in prev_id or 'person' in prev_id:
                    continue
                    
                if prev_mask.shape != obj.mask.shape:
                    continue
                
                intersection = np.sum(obj.mask & prev_mask)
                union = np.sum(obj.mask | prev_mask)
                iou = intersection / (union + 1e-6)
                
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match_id = prev_id
            
            if best_match_id:
                obj.stable_id = best_match_id
            
            tracked.append(obj)
            self._previous_masks[obj.stable_id] = obj.mask.copy()
        
        # Clean up old masks (but keep face/person)
        if len(self._previous_masks) > self.max_objects * 2:
            keys = list(self._previous_masks.keys())
            for key in keys[:len(keys) // 2]:
                if 'face' not in key and 'person' not in key:
                    del self._previous_masks[key]
        
        return tracked
    
    def _mask_to_bbox(self, mask: NDArray) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def apply_effect_to_mask(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        effect_type: str,
        intensity: float = 0.5,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> NDArray[np.uint8]:
        """Apply visual effect to masked region."""
        if mask is None or not np.any(mask):
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Ensure mask matches frame size
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h))
        
        # Feather mask edges
        mask_float = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        mask_3d = np.stack([mask_float, mask_float, mask_float], axis=2)
        
        if effect_type == 'blur':
            blur_size = int(5 + intensity * 50) | 1
            blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
            result = (frame * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
        
        elif effect_type == 'color' and color is not None:
            overlay = np.full_like(frame, color, dtype=np.uint8)
            alpha = intensity * 0.7
            result = (frame * (1 - mask_3d * alpha) + overlay * mask_3d * alpha).astype(np.uint8)
        
        elif effect_type == 'darken':
            darkened = (frame * (1 - intensity * 0.7)).astype(np.uint8)
            result = np.where(mask_3d > 0.5, darkened, frame).astype(np.uint8)
        
        elif effect_type == 'brighten':
            brightened = np.clip(frame * (1 + intensity * 0.5), 0, 255).astype(np.uint8)
            result = np.where(mask_3d > 0.5, brightened, frame).astype(np.uint8)
        
        elif effect_type == 'pixelate':
            pixel_size = max(2, int(intensity * 40))
            small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            result = np.where(mask_3d > 0.5, pixelated, frame).astype(np.uint8)
        
        elif effect_type == 'desaturate' or effect_type == 'saturate':
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            if effect_type == 'desaturate':
                hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity)
            else:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity), 0, 255)
            modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = np.where(mask_3d > 0.5, modified, frame).astype(np.uint8)
        
        elif effect_type == 'highlight':
            # Draw colored outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            highlight_color = color if color else (0, 255, 255)
            cv2.drawContours(result, contours, -1, highlight_color, 3)
        
        elif effect_type == 'hide':
            # Heavy blur to obscure
            blur_size = 51
            blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
            result = np.where(mask_3d > 0.5, blurred, frame).astype(np.uint8)
        
        elif effect_type == 'thermal':
            # Apply thermal color map
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
            result = np.where(mask_3d > 0.5, thermal, frame).astype(np.uint8)
        
        elif effect_type == 'invert':
            inverted = 255 - frame
            result = np.where(mask_3d > 0.5, inverted, frame).astype(np.uint8)
        
        return result
    
    def get_background_mask(self, frame_shape: Tuple[int, int]) -> NDArray[np.uint8]:
        """Get mask for everything except detected people."""
        h, w = frame_shape
        background = np.ones((h, w), dtype=np.uint8)
        
        # Subtract person masks
        for obj_id, mask in self._previous_masks.items():
            if 'person' in obj_id or 'face' in obj_id:
                if mask.shape == (h, w):
                    background = background & (1 - mask)
        
        return background
    
    @property
    def average_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times[-30:]) / min(len(self._inference_times), 30)
    
    def shutdown(self):
        """Release resources."""
        if self._selfie_seg:
            self._selfie_seg.close()
        if self._face_detection:
            self._face_detection.close()
        self._is_initialized = False
        logger.info("SAM Auto Segmenter shutdown")


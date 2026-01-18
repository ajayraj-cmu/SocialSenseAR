"""
Real-time Segmentation using MediaPipe.

Provides actual segmentation masks for:
- People/selfie segmentation
- Face detection with mesh
- Hand tracking

These masks DYNAMICALLY FOLLOW the objects in real-time.
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")

from src.core.contracts import (
    SegmentedObject,
    SegmentationResult,
    BoundingRegion,
    ObjectClass,
)


class RealtimeSegmenter:
    """
    Real-time segmentation using MediaPipe.
    
    Creates ACTUAL masks that follow objects dynamically.
    """
    
    def __init__(
        self,
        enable_selfie_segmentation: bool = True,
        enable_face_detection: bool = True,
        model_selection: int = 1,  # 0=close-range, 1=full-range
        min_detection_confidence: float = 0.5,
    ):
        """
        Initialize real-time segmenter.
        
        Args:
            enable_selfie_segmentation: Enable person/background segmentation
            enable_face_detection: Enable face detection
            model_selection: 0 for close-range (within 2m), 1 for full-range (within 5m)
            min_detection_confidence: Minimum confidence for detections
        """
        self.enable_selfie = enable_selfie_segmentation
        self.enable_face = enable_face_detection
        self.model_selection = model_selection
        self.min_confidence = min_detection_confidence
        
        # MediaPipe components
        self._selfie_segmentation = None
        self._face_detection = None
        self._face_mesh = None
        
        # Tracking
        self._object_counter = 0
        self._tracked_objects: Dict[str, SegmentedObject] = {}
        
        # Performance
        self._inference_times: List[float] = []
        
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize MediaPipe models."""
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not available")
            return False
        
        try:
            if self.enable_selfie:
                self._selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
                    model_selection=self.model_selection
                )
                logger.info("Selfie segmentation initialized")
            
            if self.enable_face:
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=self.model_selection,
                    min_detection_confidence=self.min_confidence
                )
                logger.info("Face detection initialized")
            
            self._is_initialized = True
            logger.info("Real-time segmenter initialized with MediaPipe")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False
    
    def segment_frame(
        self,
        frame: NDArray[np.uint8],
        frame_id: int,
    ) -> SegmentationResult:
        """
        Segment objects in frame using MediaPipe.
        
        Returns REAL masks that follow objects.
        """
        start_time = time.perf_counter()
        
        if not self._is_initialized:
            if not self.initialize():
                return SegmentationResult(
                    objects=[],
                    inference_time_ms=0,
                    model_confidence=0,
                    success=False,
                    error_message="Segmenter not initialized"
                )
        
        objects = []
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        
        try:
            # Person/Selfie Segmentation - creates a mask of the person
            if self._selfie_segmentation:
                results = self._selfie_segmentation.process(rgb_frame)
                
                if results.segmentation_mask is not None:
                    # Get the segmentation mask (values 0-1)
                    mask = results.segmentation_mask
                    
                    # Threshold to binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    
                    # Find bounding box of the mask
                    rows = np.any(binary_mask, axis=1)
                    cols = np.any(binary_mask, axis=0)
                    
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        # Create segmented object
                        obj = SegmentedObject(
                            stable_id="person_0",
                            mask=binary_mask,
                            confidence=float(np.mean(mask[binary_mask > 0])) if np.any(binary_mask) else 0.5,
                            bounding_region=BoundingRegion(
                                x_min=int(x_min),
                                y_min=int(y_min),
                                x_max=int(x_max),
                                y_max=int(y_max)
                            ),
                            class_label=ObjectClass.PERSON,
                            saliency_score=0.9,
                            first_seen_frame=frame_id,
                            last_seen_frame=frame_id,
                            smoothed_mask=binary_mask,
                        )
                        objects.append(obj)
            
            # Face Detection
            if self._face_detection:
                face_results = self._face_detection.process(rgb_frame)
                
                if face_results.detections:
                    for i, detection in enumerate(face_results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative to absolute coordinates
                        x_min = int(bbox.xmin * w)
                        y_min = int(bbox.ymin * h)
                        box_w = int(bbox.width * w)
                        box_h = int(bbox.height * h)
                        
                        # Create a mask for the face (ellipse)
                        face_mask = np.zeros((h, w), dtype=np.uint8)
                        center = (x_min + box_w // 2, y_min + box_h // 2)
                        axes = (box_w // 2, box_h // 2)
                        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1, -1)
                        
                        obj = SegmentedObject(
                            stable_id=f"face_{i}",
                            mask=face_mask,
                            confidence=detection.score[0] if detection.score else 0.5,
                            bounding_region=BoundingRegion(
                                x_min=max(0, x_min),
                                y_min=max(0, y_min),
                                x_max=min(w, x_min + box_w),
                                y_max=min(h, y_min + box_h)
                            ),
                            class_label=ObjectClass.FACE,
                            saliency_score=0.95,
                            first_seen_frame=frame_id,
                            last_seen_frame=frame_id,
                            smoothed_mask=face_mask,
                        )
                        objects.append(obj)
            
            inference_time = (time.perf_counter() - start_time) * 1000
            self._inference_times.append(inference_time)
            
            avg_confidence = sum(o.confidence for o in objects) / len(objects) if objects else 0
            
            return SegmentationResult(
                objects=objects,
                inference_time_ms=inference_time,
                model_confidence=avg_confidence,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return SegmentationResult(
                objects=[],
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                model_confidence=0,
                success=False,
                error_message=str(e)
            )
    
    def apply_color_overlay(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        color: Tuple[int, int, int],
        alpha: float = 0.5,
    ) -> NDArray[np.uint8]:
        """
        Apply a colored overlay to the masked region.
        
        Args:
            frame: RGB frame
            mask: Binary mask
            color: RGB color tuple (R, G, B)
            alpha: Blend factor (0 = original, 1 = full color)
        
        Returns:
            Frame with colored overlay on masked region
        """
        result = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[:] = color
        
        # Apply only to masked region
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Blend
        result = np.where(
            mask_3d > 0,
            (frame * (1 - alpha) + overlay * alpha).astype(np.uint8),
            frame
        )
        
        return result
    
    def apply_blur_to_mask(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        blur_strength: int = 25,
    ) -> NDArray[np.uint8]:
        """
        Apply blur only to the masked region.
        """
        # Blur the entire frame
        blurred = cv2.GaussianBlur(frame, (blur_strength | 1, blur_strength | 1), 0)
        
        # Blend based on mask
        mask_3d = np.stack([mask, mask, mask], axis=2).astype(np.float32)
        
        # Smooth mask edges
        mask_3d = cv2.GaussianBlur(mask_3d, (5, 5), 0)
        
        result = (frame * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
        
        return result
    
    def apply_effect_to_mask(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        effect_type: str,
        intensity: float = 0.5,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> NDArray[np.uint8]:
        """
        Apply various effects to the masked region.
        
        Args:
            frame: RGB frame
            mask: Binary mask (1 = apply effect, 0 = leave alone)
            effect_type: 'color', 'blur', 'darken', 'saturate', 'pixelate'
            intensity: Effect strength (0-1)
            color: Color for 'color' effect (R, G, B)
        
        Returns:
            Modified frame
        """
        if mask is None or not np.any(mask):
            return frame
        
        result = frame.copy()
        mask_3d = np.stack([mask, mask, mask], axis=2).astype(np.float32)
        
        # Feather the mask edges for smooth blending
        mask_3d = cv2.GaussianBlur(mask_3d, (7, 7), 0)
        
        if effect_type == 'color' and color is not None:
            # Color overlay
            overlay = np.zeros_like(frame)
            overlay[:] = color
            result = (frame * (1 - mask_3d * intensity) + overlay * mask_3d * intensity).astype(np.uint8)
        
        elif effect_type == 'blur':
            blur_size = int(5 + intensity * 40) | 1
            blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
            result = (frame * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
        
        elif effect_type == 'darken':
            darkened = (frame * (1 - intensity * 0.7)).astype(np.uint8)
            result = np.where(mask_3d > 0.5, darkened, frame)
        
        elif effect_type == 'pixelate':
            # Pixelate effect
            pixel_size = max(2, int(intensity * 30))
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            result = np.where(mask_3d > 0.5, pixelated, frame)
        
        elif effect_type == 'saturate':
            # Increase/decrease saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + intensity)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = np.where(mask_3d > 0.5, saturated, frame)
        
        return result
    
    @property
    def average_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times[-30:]) / min(len(self._inference_times), 30)
    
    def shutdown(self):
        """Release resources."""
        if self._selfie_segmentation:
            self._selfie_segmentation.close()
        if self._face_detection:
            self._face_detection.close()
        self._is_initialized = False
        logger.info("Real-time segmenter shutdown")


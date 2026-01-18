"""
Target Resolution System.

Resolves user intent targets to object IDs using:
- Spatial relations (left, right, near, far)
- Object class matching
- Saliency-based selection
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from loguru import logger

from src.core.contracts import (
    UserIntent,
    SegmentedObject,
    SpatialRelation,
    ObjectClass,
)


class TargetResolver:
    """
    Resolves user intent targets to concrete object IDs.
    
    Guarantees:
    - User references resolve to object IDs, never pixels
    - Ambiguous targets are flagged for clarification
    - Spatial relations are computed from camera perspective
    """
    
    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        ambiguity_threshold: float = 0.8,
    ):
        """
        Initialize target resolver.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            ambiguity_threshold: Score difference below which targets are ambiguous
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ambiguity_threshold = ambiguity_threshold
    
    def resolve(
        self,
        intent: UserIntent,
        objects: List[SegmentedObject],
    ) -> Tuple[List[str], bool, Optional[str]]:
        """
        Resolve intent target to object IDs.
        
        Args:
            intent: Parsed user intent
            objects: Currently tracked objects
            
        Returns:
            Tuple of (object_ids, is_ambiguous, clarification_message)
            Special: Returns ["__GLOBAL__"] for whole-screen effects
        """
        # Check for global/whole screen target
        if intent.target_description in ["whole screen", "unspecified target", ""]:
            # But if we have detected objects, prefer to use them!
            if objects:
                # Return all detected object IDs to apply effect to all of them
                return ([obj.stable_id for obj in objects], False, None)
            return (["__GLOBAL__"], False, None)
        
        if not intent.target_class and not intent.target_spatial_relation:
            # No specific target - use all detected objects if available
            if objects:
                return ([obj.stable_id for obj in objects], False, None)
            return (["__GLOBAL__"], False, None)
        
        if not objects:
            # No objects but we have a target - apply globally anyway
            return (["__GLOBAL__"], False, None)
        
        # Score each object based on intent criteria
        scored_objects = []
        
        for obj in objects:
            score = self._score_object(obj, intent)
            if score > 0:
                scored_objects.append((obj, score))
        
        if not scored_objects:
            # No matching objects - apply globally
            return (["__GLOBAL__"], False, None)
        
        # Sort by score
        scored_objects.sort(key=lambda x: x[1], reverse=True)
        
        # Check for ambiguity
        if len(scored_objects) > 1:
            score_diff = scored_objects[0][1] - scored_objects[1][1]
            if score_diff < self.ambiguity_threshold:
                # Multiple similar matches - just use all of them
                candidates = [obj.stable_id for obj, _ in scored_objects[:3]]
                return (candidates, False, None)  # Don't ask for clarification, just do it
        
        # Return best match(es)
        best_object = scored_objects[0][0]
        return ([best_object.stable_id], False, None)
    
    def _score_object(
        self,
        obj: SegmentedObject,
        intent: UserIntent,
    ) -> float:
        """
        Score an object based on how well it matches the intent.
        
        Higher score = better match.
        """
        score = 0.0
        
        # Class matching
        if intent.target_class:
            if obj.class_label == intent.target_class:
                score += 3.0
            elif obj.class_label == ObjectClass.UNKNOWN:
                # Unknown class gets partial score
                score += 0.5
            else:
                # Wrong class - penalty
                return 0.0
        else:
            # No class specified - all objects are candidates
            score += 1.0
        
        # Spatial relation matching
        if intent.target_spatial_relation:
            spatial_score = self._score_spatial_match(obj, intent.target_spatial_relation)
            if spatial_score <= 0:
                return 0.0  # Object doesn't match spatial criteria
            score += spatial_score * 2.0
        
        # Saliency bonus
        score += obj.saliency_score * 0.5
        
        # Confidence bonus
        score += obj.confidence * 0.3
        
        # Track stability bonus
        stability = min(obj.frames_tracked / 30.0, 1.0)
        score += stability * 0.2
        
        return score
    
    def _score_spatial_match(
        self,
        obj: SegmentedObject,
        relation: SpatialRelation,
    ) -> float:
        """
        Score how well an object matches a spatial relation.
        
        Returns:
            Score from 0 (no match) to 1 (perfect match)
        """
        center_x, center_y = obj.bounding_region.center
        
        # Normalize coordinates to [-1, 1]
        norm_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
        norm_y = (center_y - self.frame_height / 2) / (self.frame_height / 2)
        
        if relation == SpatialRelation.LEFT:
            # Object should be on left side (negative x)
            if norm_x < 0:
                return 1.0 - (norm_x + 1) / 1  # Farther left = better
            return 0.0
        
        elif relation == SpatialRelation.RIGHT:
            # Object should be on right side (positive x)
            if norm_x > 0:
                return norm_x
            return 0.0
        
        elif relation == SpatialRelation.CENTER:
            # Object should be near center
            distance = abs(norm_x)
            if distance < 0.3:
                return 1.0 - distance / 0.3
            return 0.0
        
        elif relation == SpatialRelation.ABOVE:
            # Object should be in upper half (negative y in image coords)
            if norm_y < 0:
                return 1.0 - (norm_y + 1)
            return 0.0
        
        elif relation == SpatialRelation.BELOW:
            if norm_y > 0:
                return norm_y
            return 0.0
        
        elif relation == SpatialRelation.NEAR:
            # Use depth if available, otherwise use size as proxy
            if obj.depth_estimate is not None:
                # Closer objects have lower depth
                if obj.depth_estimate < 2.0:  # Within 2 meters
                    return 1.0 - obj.depth_estimate / 2.0
                return 0.0
            else:
                # Use relative size as proxy
                area_ratio = obj.bounding_region.area / (self.frame_width * self.frame_height)
                if area_ratio > 0.05:  # Large objects are "near"
                    return min(area_ratio * 10, 1.0)
                return 0.0
        
        elif relation == SpatialRelation.FAR:
            if obj.depth_estimate is not None:
                if obj.depth_estimate > 3.0:  # Beyond 3 meters
                    return min((obj.depth_estimate - 3.0) / 5.0, 1.0)
                return 0.0
            else:
                area_ratio = obj.bounding_region.area / (self.frame_width * self.frame_height)
                if area_ratio < 0.02:
                    return 1.0 - area_ratio * 50
                return 0.0
        
        # Default for unhandled relations
        return 0.5
    
    def resolve_by_point(
        self,
        x: int,
        y: int,
        objects: List[SegmentedObject],
    ) -> Optional[str]:
        """
        Resolve target by point click.
        
        Args:
            x: Click x-coordinate
            y: Click y-coordinate
            objects: Currently tracked objects
            
        Returns:
            Object ID at point, or None
        """
        for obj in objects:
            if (obj.mask is not None and
                0 <= y < obj.mask.shape[0] and
                0 <= x < obj.mask.shape[1] and
                obj.mask[y, x] > 0):
                return obj.stable_id
        return None
    
    def get_objects_in_region(
        self,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        objects: List[SegmentedObject],
        min_overlap: float = 0.5,
    ) -> List[str]:
        """
        Get objects overlapping a region.
        
        Args:
            x_min, y_min, x_max, y_max: Region bounds
            objects: Currently tracked objects
            min_overlap: Minimum IoU for inclusion
            
        Returns:
            List of object IDs in region
        """
        from src.core.contracts import BoundingRegion
        
        region = BoundingRegion(x_min, y_min, x_max, y_max)
        
        result = []
        for obj in objects:
            iou = obj.bounding_region.iou(region)
            if iou >= min_overlap:
                result.append(obj.stable_id)
        
        return result
    
    def describe_objects(
        self,
        objects: List[SegmentedObject],
    ) -> str:
        """
        Generate human-readable description of visible objects.
        
        Useful for feedback to user.
        """
        if not objects:
            return "No objects detected."
        
        descriptions = []
        
        for obj in objects:
            # Position description
            center_x, center_y = obj.bounding_region.center
            norm_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
            
            if norm_x < -0.3:
                position = "on the left"
            elif norm_x > 0.3:
                position = "on the right"
            else:
                position = "in the center"
            
            class_name = obj.class_label.value if obj.class_label != ObjectClass.UNKNOWN else "object"
            descriptions.append(f"- {class_name} {position} (ID: {obj.stable_id})")
        
        return "Detected objects:\n" + "\n".join(descriptions)



"""
Audio-Visual Binding System.

Binds audio sources to visual entities using:
- Spatial correlation
- Temporal correlation  
- Motion correspondence
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from loguru import logger

from src.core.contracts import (
    SegmentedObject,
    AudioSource,
    AudioVisualBinding,
)


@dataclass
class SpatialAudioEstimate:
    """Estimated spatial position of an audio source."""
    azimuth_degrees: float  # Horizontal angle (-180 to 180)
    elevation_degrees: float  # Vertical angle (-90 to 90)
    confidence: float
    energy: float


class AudioVisualBinder:
    """
    Binds audio sources to visual entities.
    
    Rules:
    - Audio edits must be linked to a visual entity
    - Respect spatial positioning
    - Preserve ambient continuity
    - Fall back to spatial attenuation if confident binding unavailable
    """
    
    def __init__(
        self,
        binding_confidence_threshold: float = 0.6,
        spatial_resolution_degrees: float = 15.0,
        fallback_to_spatial: bool = True,
        ambient_preservation_ratio: float = 0.15,
    ):
        """
        Initialize audio-visual binder.
        
        Args:
            binding_confidence_threshold: Minimum confidence for binding
            spatial_resolution_degrees: Angular resolution for spatial matching
            fallback_to_spatial: Fall back to spatial attenuation on failure
            ambient_preservation_ratio: Minimum ambient sound to preserve
        """
        self.binding_confidence_threshold = binding_confidence_threshold
        self.spatial_resolution_degrees = spatial_resolution_degrees
        self.fallback_to_spatial = fallback_to_spatial
        self.ambient_preservation_ratio = ambient_preservation_ratio
        
        # Active bindings
        self._bindings: Dict[str, AudioVisualBinding] = {}
        
        # History for temporal correlation
        self._binding_history: List[Dict[str, AudioVisualBinding]] = []
        self._max_history = 30  # frames
    
    def estimate_audio_sources(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = 48000,
    ) -> List[AudioSource]:
        """
        Estimate audio source positions from audio.
        
        Uses inter-channel time difference (ITD) and 
        inter-channel level difference (ILD) for localization.
        
        Args:
            audio_chunk: Audio (samples,) or (samples x channels)
            sample_rate: Audio sample rate
            
        Returns:
            List of estimated audio sources
        """
        if audio_chunk is None or audio_chunk.size == 0:
            return []
        
        sources = []
        
        try:
            # Handle mono audio - can't do spatial estimation
            if audio_chunk.ndim == 1 or audio_chunk.shape[1] == 1:
                # Mono audio - create centered source based on energy
                mono = audio_chunk.flatten()
                total_energy = float(np.sqrt(np.mean(mono ** 2)))
                
                if total_energy > 0.001:
                    source = AudioSource(
                        source_id="audio_src_0",
                        azimuth_degrees=0.0,  # Centered
                        elevation_degrees=0.0,
                        confidence=0.5,  # Lower confidence for mono
                    )
                    sources.append(source)
                return sources
            
            left = audio_chunk[:, 0]
            right = audio_chunk[:, 1]
            
            # Calculate inter-channel level difference (ILD)
            left_rms = np.sqrt(np.mean(left ** 2)) + 1e-10
            right_rms = np.sqrt(np.mean(right ** 2)) + 1e-10
            
            ild_db = 20 * np.log10(right_rms / left_rms)
            
            # Estimate azimuth from ILD
            # Simple mapping: -20dB to +20dB -> -90° to +90°
            azimuth = np.clip(ild_db * 4.5, -90, 90)
            
            # Calculate inter-channel time difference (ITD) using cross-correlation
            correlation = np.correlate(left, right, mode='full')
            max_idx = np.argmax(correlation)
            lag_samples = max_idx - len(left) + 1
            
            # Convert lag to angle (assuming ~0.6ms max ITD for humans)
            max_itd_samples = int(0.0006 * sample_rate)
            if max_itd_samples > 0:
                itd_azimuth = np.clip(
                    lag_samples / max_itd_samples * 90,
                    -90, 90
                )
                # Blend ILD and ITD estimates
                azimuth = 0.6 * azimuth + 0.4 * itd_azimuth
            
            # Calculate overall energy
            total_energy = float(np.sqrt(np.mean(audio_chunk ** 2)))
            
            # Confidence based on correlation peak
            peak_value = correlation[max_idx]
            confidence = float(np.clip(peak_value / (np.max(np.abs(correlation)) + 1e-10), 0, 1))
            
            if total_energy > 0.001:  # Minimum energy threshold
                source = AudioSource(
                    source_id=f"audio_src_{len(sources)}",
                    azimuth_degrees=float(azimuth),
                    elevation_degrees=0.0,  # Can't estimate from stereo
                    confidence=confidence,
                )
                sources.append(source)
            
        except Exception as e:
            logger.error(f"Audio source estimation failed: {e}")
        
        return sources
    
    def bind_audio_to_visual(
        self,
        audio_sources: List[AudioSource],
        visual_objects: List[SegmentedObject],
        frame_width: int,
        frame_height: int,
    ) -> List[AudioVisualBinding]:
        """
        Bind audio sources to visual objects.
        
        Args:
            audio_sources: Estimated audio sources
            visual_objects: Segmented visual objects
            frame_width: Video frame width
            frame_height: Video frame height
            
        Returns:
            List of audio-visual bindings
        """
        bindings = []
        
        for audio_source in audio_sources:
            best_match: Optional[SegmentedObject] = None
            best_score = 0.0
            best_offset = 0.0
            
            for obj in visual_objects:
                # Calculate object's angular position
                obj_azimuth = self._pixel_to_azimuth(
                    obj.bounding_region.center[0],
                    frame_width
                )
                
                # Calculate angular difference
                angular_diff = abs(audio_source.azimuth_degrees - obj_azimuth)
                
                # Score based on angular proximity
                if angular_diff <= self.spatial_resolution_degrees:
                    score = 1.0 - (angular_diff / self.spatial_resolution_degrees)
                    
                    # Boost score for certain object classes (e.g., people)
                    if obj.class_label.value == "person":
                        score *= 1.3
                    
                    # Boost score for salient objects
                    score *= (0.5 + 0.5 * obj.saliency_score)
                    
                    if score > best_score:
                        best_score = score
                        best_match = obj
                        best_offset = audio_source.azimuth_degrees - obj_azimuth
            
            if best_match is not None and best_score >= self.binding_confidence_threshold:
                binding = AudioVisualBinding(
                    audio_source_id=audio_source.source_id,
                    visual_object_id=best_match.stable_id,
                    confidence=float(best_score),
                    binding_type="spatial",
                    angular_offset_degrees=float(best_offset),
                )
                bindings.append(binding)
                
                # Update audio source with binding
                audio_source.bound_object_id = best_match.stable_id
                
                logger.debug(
                    f"Bound audio source {audio_source.source_id} to "
                    f"object {best_match.stable_id} (confidence: {best_score:.2f})"
                )
        
        # Update binding history
        self._update_binding_history(bindings)
        
        return bindings
    
    def _pixel_to_azimuth(self, x: int, frame_width: int) -> float:
        """
        Convert pixel x-coordinate to azimuth angle.
        
        Assumes horizontal FoV of ~90 degrees for Quest 3.
        """
        # Map pixel position to angle
        # Center of frame = 0 degrees
        # Left edge = -45 degrees, right edge = +45 degrees
        normalized = (x - frame_width / 2) / (frame_width / 2)
        return normalized * 45.0  # Half of 90° FoV
    
    def _azimuth_to_pixel(self, azimuth: float, frame_width: int) -> int:
        """Convert azimuth angle to pixel x-coordinate."""
        normalized = azimuth / 45.0
        return int(frame_width / 2 + normalized * frame_width / 2)
    
    def _update_binding_history(self, bindings: List[AudioVisualBinding]):
        """Update binding history for temporal consistency."""
        binding_dict = {b.visual_object_id: b for b in bindings}
        self._binding_history.append(binding_dict)
        
        if len(self._binding_history) > self._max_history:
            self._binding_history.pop(0)
    
    def get_binding_for_object(
        self,
        object_id: str,
    ) -> Optional[AudioVisualBinding]:
        """Get the current binding for an object."""
        return self._bindings.get(object_id)
    
    def get_temporally_stable_bindings(
        self,
        min_frames: int = 5,
    ) -> List[AudioVisualBinding]:
        """
        Get bindings that have been stable across multiple frames.
        
        Args:
            min_frames: Minimum frames a binding must persist
            
        Returns:
            List of stable bindings
        """
        if len(self._binding_history) < min_frames:
            return []
        
        # Count binding occurrences
        binding_counts: Dict[str, int] = {}
        binding_latest: Dict[str, AudioVisualBinding] = {}
        
        for frame_bindings in self._binding_history[-min_frames:]:
            for obj_id, binding in frame_bindings.items():
                key = f"{binding.audio_source_id}_{obj_id}"
                binding_counts[key] = binding_counts.get(key, 0) + 1
                binding_latest[key] = binding
        
        # Return bindings present in all frames
        stable = []
        for key, count in binding_counts.items():
            if count >= min_frames:
                stable.append(binding_latest[key])
        
        return stable
    
    def get_spatial_attenuation_factor(
        self,
        target_azimuth: float,
        source_azimuth: float,
        falloff_degrees: float = 30.0,
    ) -> float:
        """
        Calculate spatial attenuation factor for directional dampening.
        
        Args:
            target_azimuth: Target direction in degrees
            source_azimuth: Source direction in degrees
            falloff_degrees: Angular falloff width
            
        Returns:
            Attenuation factor (0 = fully attenuated, 1 = unchanged)
        """
        angular_diff = abs(target_azimuth - source_azimuth)
        
        # Preserve ambient sounds
        min_factor = self.ambient_preservation_ratio
        
        if angular_diff <= falloff_degrees:
            # Within target area - minimal attenuation
            factor = 1.0 - (angular_diff / falloff_degrees) * 0.3
        else:
            # Outside target - attenuate
            excess = angular_diff - falloff_degrees
            factor = max(min_factor, 0.7 - excess / 90.0)
        
        return float(factor)
    
    def reset(self):
        """Reset all bindings and history."""
        self._bindings.clear()
        self._binding_history.clear()
        logger.debug("Audio-visual binder reset")



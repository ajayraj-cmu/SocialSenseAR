"""
Video Capture from Quest 3 Passthrough.

Handles:
- Webcam/virtual camera acquisition
- Frame timing and synchronization
- Format conversion
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger


class VideoCapture:
    """
    Video capture for Quest 3 passthrough (exposed as webcam).
    
    Guarantees:
    - Consistent frame timing
    - RGB format output
    - Thread-safe access
    """
    
    def __init__(
        self,
        device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_frames: int = 3,
    ):
        """
        Initialize video capture.
        
        Args:
            device_index: Camera device index
            width: Capture width
            height: Capture height
            fps: Target frames per second
            buffer_frames: Number of frames to buffer
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_frames = buffer_frames
        
        # State
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._lock = threading.Lock()
        
        # Latest frame
        self._latest_frame: Optional[NDArray[np.uint8]] = None
        self._frame_timestamp: float = 0.0
        self._frame_count: int = 0
        
        # Performance tracking
        self._frame_times: list[float] = []
        self._actual_fps: float = 0.0
    
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if started successfully
        """
        if self._is_running:
            return True
        
        try:
            self._capture = cv2.VideoCapture(self.device_index)
            
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
            
            # Configure capture
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_frames)
            
            # Read actual settings
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(
                f"Video capture started: {actual_width}x{actual_height} @ {actual_fps}fps"
            )
            
            self._is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video capture: {e}")
            return False
    
    def stop(self):
        """Stop video capture."""
        self._is_running = False
        
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        
        logger.info("Video capture stopped")
    
    def read_frame(self) -> Tuple[Optional[NDArray[np.uint8]], float, int]:
        """
        Read a frame from the capture.
        
        Returns:
            Tuple of (frame, timestamp_ms, frame_id)
            - frame: RGB frame or None on failure
            - timestamp_ms: Capture timestamp in milliseconds
            - frame_id: Sequential frame number
        """
        if not self._is_running or self._capture is None:
            return (None, 0.0, 0)
        
        start_time = time.perf_counter()
        
        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            return (None, 0.0, self._frame_count)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with self._lock:
            self._frame_count += 1
            self._frame_timestamp = time.time() * 1000  # ms
            self._latest_frame = frame_rgb
            
            # Track frame timing
            self._frame_times.append(start_time)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            self._update_fps()
        
        return (frame_rgb, self._frame_timestamp, self._frame_count)
    
    def get_latest_frame(self) -> Tuple[Optional[NDArray[np.uint8]], float, int]:
        """
        Get the most recently captured frame.
        
        Returns:
            Tuple of (frame, timestamp_ms, frame_id)
        """
        with self._lock:
            if self._latest_frame is None:
                return (None, 0.0, 0)
            return (
                self._latest_frame.copy(),
                self._frame_timestamp,
                self._frame_count
            )
    
    def _update_fps(self):
        """Calculate actual FPS from frame times."""
        if len(self._frame_times) < 2:
            return
        
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration > 0:
            self._actual_fps = (len(self._frame_times) - 1) / duration
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def actual_fps(self) -> float:
        return self._actual_fps
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get actual frame size (width, height)."""
        if self._capture is not None:
            return (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (self.width, self.height)



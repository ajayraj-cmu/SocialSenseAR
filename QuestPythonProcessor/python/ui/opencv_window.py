"""
OpenCV window UI backend.

Simple UI using OpenCV's imshow for display.
"""
from typing import Optional
import cv2
import numpy as np

from .base import BaseUI


class OpenCVUI(BaseUI):
    """OpenCV-based UI using imshow.

    Simple and portable UI that works on most platforms.
    Uses cv2.imshow() for display and cv2.waitKey() for input.
    """

    def __init__(self, config=None):
        """Initialize OpenCV UI.

        Args:
            config: Configuration object with display settings
        """
        self.config = config
        self.window_name = "Quest Processor"
        self.max_width = 1920
        if config:
            self.max_width = getattr(config, 'display_max_width', 1920)

    def setup(self, title: str = "Quest Processor") -> None:
        """Initialize the display window.

        Args:
            title: Window title
        """
        self.window_name = title
        # Window will be created on first show()

    def show(self, frame: np.ndarray, stats: Optional[dict] = None) -> None:
        """Display a frame.

        Args:
            frame: BGR frame to display
            stats: Optional stats dict (not used currently, could add overlay)
        """
        display_frame = frame

        # Downscale if too wide
        h, w = display_frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            display_frame = cv2.resize(
                display_frame, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_NEAREST
            )

        cv2.imshow(self.window_name, display_frame)

    def poll_input(self) -> Optional[int]:
        """Poll for keyboard input.

        Returns:
            Key code (0-255) or None if no key pressed
        """
        key = cv2.waitKey(1) & 0xFF
        if key == 255:  # No key pressed
            return None
        return key

    def cleanup(self) -> None:
        """Destroy the window."""
        cv2.destroyAllWindows()

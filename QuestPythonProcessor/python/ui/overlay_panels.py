#!/usr/bin/env python3
"""
Overlay panels for context window and emotion display.

Renders glassmorphism-style panels directly onto the OpenCV frame
on the left side, following head position with smoothing.
Context on top, emotion on bottom.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class OverlayRenderer:
    """Renders context and emotion panels directly onto frames."""

    def __init__(self):
        # Panel config
        self.panel_width = 280
        self.margin = 20
        self.panel_gap = 12
        self.corner_radius = 16

        # Colors (BGR format for OpenCV)
        self.glass_bg = (196, 132, 86)  # Calm blue
        self.glass_alpha = 0.25
        self.border_color = (252, 220, 200)  # Light blue border
        self.border_alpha = 0.72
        self.text_color = (255, 250, 244)  # Near white
        self.label_color = (244, 224, 208)  # Muted label
        self.dim_text_color = (200, 200, 200)  # Dimmed text

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.label_scale = 0.35
        self.text_scale = 0.45
        self.emotion_scale = 0.6

        # Smoothing for position (higher = more responsive)
        self._smooth_x = 0.5
        self._smooth_y = 0.5
        self._smooth_factor = 0.25  # Pretty responsive but still smooth

        # Offset from person (panel appears to the left of them)
        self.offset_from_person = 50  # pixels to the left of person

        # State file path
        env_path = os.getenv("CONVO_STATE_PATH")
        if env_path:
            self.state_file = Path(env_path).expanduser()
        else:
            home_guess = Path("~/Downloads/Nex/conve_context/latest_state.json").expanduser()
            rel_guess = (
                Path(__file__).resolve().parent.parent.parent
                / "conve_context"
                / "latest_state.json"
            )
            self.state_file = home_guess if home_guess.exists() else rel_guess

        # Cached state
        self._cached_state = None
        self._last_state_check = 0
        self._state_check_interval = 0.3  # Check every 300ms

    def _load_state(self) -> dict:
        """Load conversation state from file (with caching)."""
        now = time.time()
        if now - self._last_state_check < self._state_check_interval:
            return self._cached_state or {}

        self._last_state_check = now
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    self._cached_state = json.load(f)
                    return self._cached_state
        except:
            pass
        return self._cached_state or {}

    def _draw_rounded_rect(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                           bg_color: Tuple[int, int, int], bg_alpha: float,
                           border_color: Tuple[int, int, int], border_alpha: float,
                           radius: int = 16) -> None:
        """Draw a rounded rectangle with transparency (glassmorphism effect)."""
        # Create overlay for transparency
        overlay = frame.copy()

        # Draw filled rounded rectangle
        # Top-left corner
        cv2.ellipse(overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, bg_color, -1)
        # Top-right corner
        cv2.ellipse(overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, bg_color, -1)
        # Bottom-left corner
        cv2.ellipse(overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, bg_color, -1)
        # Bottom-right corner
        cv2.ellipse(overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, bg_color, -1)

        # Fill rectangles
        cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), bg_color, -1)
        cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), bg_color, -1)

        # Blend with original
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Draw border
        border_overlay = frame.copy()
        # Draw rounded border using arcs and lines
        cv2.ellipse(border_overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, border_color, 2)
        cv2.ellipse(border_overlay, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, border_color, 2)
        cv2.ellipse(border_overlay, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, border_color, 2)
        cv2.ellipse(border_overlay, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, border_color, 2)
        cv2.line(border_overlay, (x + radius, y), (x + w - radius, y), border_color, 2)
        cv2.line(border_overlay, (x + radius, y + h), (x + w - radius, y + h), border_color, 2)
        cv2.line(border_overlay, (x, y + radius), (x, y + h - radius), border_color, 2)
        cv2.line(border_overlay, (x + w, y + radius), (x + w, y + h - radius), border_color, 2)

        cv2.addWeighted(border_overlay, border_alpha, frame, 1 - border_alpha, 0, frame)

    def _wrap_text(self, text: str, max_width: int, font_scale: float) -> list:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            (w, _), _ = cv2.getTextSize(test_line, self.font, font_scale, 1)
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def _draw_text_with_shadow(self, frame: np.ndarray, text: str, pos: Tuple[int, int],
                                font_scale: float, color: Tuple[int, int, int],
                                thickness: int = 1) -> None:
        """Draw text with subtle shadow for depth."""
        x, y = pos
        # Shadow
        cv2.putText(frame, text, (x + 1, y + 1), self.font, font_scale, (30, 30, 30), thickness, cv2.LINE_AA)
        # Main text
        cv2.putText(frame, text, (x, y), self.font, font_scale, color, thickness, cv2.LINE_AA)

    def render(self, frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5) -> np.ndarray:
        """Render overlay panels onto frame.

        Args:
            frame: BGR frame to render onto
            head_x: Normalized head X position (0-1)
            head_y: Normalized head Y position (0-1)

        Returns:
            Frame with overlay rendered
        """
        h, w = frame.shape[:2]
        half_w = w // 2  # For stereo, left eye is first half

        # Smooth the head position
        self._smooth_x += (head_x - self._smooth_x) * self._smooth_factor
        self._smooth_y += (head_y - self._smooth_y) * self._smooth_factor

        # Load current state
        state = self._load_state()
        convo_summary = state.get("convo_state_summary", "--")
        question = state.get("question", "")
        utterance = state.get("recent_utterance", "--")
        emotion = state.get("emotion") or state.get("emotion_label") or "Calm"

        # Calculate panel positions
        # Context panel (top) - taller
        context_height = 180
        emotion_height = 70

        total_height = context_height + self.panel_gap + emotion_height
        margin_top_bottom = 40

        # Fixed X position - pinned to left side with margin for headset visibility
        panel_x = 80  # Fixed distance from left edge

        # Fixed Y position - vertically centered
        base_y = (h - total_height) // 2

        # === CONTEXT PANEL ===
        context_y = base_y
        self._draw_rounded_rect(
            frame, panel_x, context_y, self.panel_width, context_height,
            self.glass_bg, self.glass_alpha,
            self.border_color, self.border_alpha,
            self.corner_radius
        )

        # Conversation label
        label_y = context_y + 22
        self._draw_text_with_shadow(frame, "CONVERSATION", (panel_x + 12, label_y),
                                    self.label_scale, self.label_color)

        # Conversation summary (wrapped)
        text_y = label_y + 20
        summary_lines = self._wrap_text(convo_summary, self.panel_width - 24, self.text_scale)
        for i, line in enumerate(summary_lines[:2]):  # Max 2 lines
            self._draw_text_with_shadow(frame, line, (panel_x + 12, text_y + i * 18),
                                        self.text_scale, self.text_color)

        # Question section
        question_y = context_y + 70
        self._draw_text_with_shadow(frame, "QUESTION", (panel_x + 12, question_y),
                                    self.label_scale, self.label_color)
        q_text = question if question else "No question detected"
        q_color = self.text_color if question else self.dim_text_color
        q_lines = self._wrap_text(q_text, self.panel_width - 24, self.text_scale)
        for i, line in enumerate(q_lines[:2]):
            self._draw_text_with_shadow(frame, line, (panel_x + 12, question_y + 18 + i * 18),
                                        self.text_scale, q_color)

        # Utterance section
        utterance_y = context_y + 130
        self._draw_text_with_shadow(frame, "JUST SAID", (panel_x + 12, utterance_y),
                                    self.label_scale, self.label_color)
        utt_text = f'"{utterance}"' if utterance != "--" else "--"
        utt_lines = self._wrap_text(utt_text, self.panel_width - 24, self.text_scale * 0.9)
        for i, line in enumerate(utt_lines[:2]):
            self._draw_text_with_shadow(frame, line, (panel_x + 12, utterance_y + 18 + i * 16),
                                        self.text_scale * 0.9, self.dim_text_color)

        # === EMOTION PANEL ===
        emotion_y = context_y + context_height + self.panel_gap
        self._draw_rounded_rect(
            frame, panel_x, emotion_y, self.panel_width, emotion_height,
            self.glass_bg, self.glass_alpha,
            self.border_color, self.border_alpha,
            self.corner_radius
        )

        # Emotion label
        self._draw_text_with_shadow(frame, "EMOTION", (panel_x + 12, emotion_y + 22),
                                    self.label_scale, self.label_color)

        # Emotion text (larger)
        self._draw_text_with_shadow(frame, emotion, (panel_x + 12, emotion_y + 50),
                                    self.emotion_scale, self.text_color, thickness=2)

        return frame


# Global renderer instance
_renderer: Optional[OverlayRenderer] = None


def get_renderer() -> OverlayRenderer:
    """Get or create the overlay renderer."""
    global _renderer
    if _renderer is None:
        _renderer = OverlayRenderer()
    return _renderer


def render_overlay(frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5) -> np.ndarray:
    """Render overlay panels onto frame.

    Args:
        frame: BGR frame to render onto
        head_x: Normalized head X position (0-1)
        head_y: Normalized head Y position (0-1)

    Returns:
        Frame with overlay rendered
    """
    renderer = get_renderer()
    return renderer.render(frame, head_x, head_y)


# Keep these for backwards compatibility but they're no-ops now
def start_overlay():
    """No-op - overlay is now rendered directly onto frames."""
    pass


def stop_overlay():
    """No-op - overlay is now rendered directly onto frames."""
    pass


def update_head_position(head_y: float):
    """No-op - head position is passed directly to render_overlay."""
    pass

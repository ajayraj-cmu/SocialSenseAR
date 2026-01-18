#!/usr/bin/env python3
"""
Modern glassmorphism overlay panels with smooth animations.

Features:
- Only shows when a person is being tracked
- Smooth fade + slide animations
- Frosted glass effect with blur
- Clean, modern design
"""
import json
import os
import time
import math
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter


class OverlayRenderer:
    """Renders modern glassmorphism overlay panels with animations."""

    def __init__(self):
        # Panel config
        self.panel_width = 340
        self.panel_gap = 14
        self.corner_radius = 16
        self.padding_x = 24
        self.padding_y = 20

        # Animation settings
        self.animation_duration = 0.4  # seconds
        self._visible = False
        self._animation_progress = 0.0  # 0 = hidden, 1 = visible
        self._last_update = time.time()

        # Glassmorphism colors
        self.bg_color = (18, 18, 22)  # Near black
        self.bg_alpha = 0.75
        self.border_color = (255, 255, 255)  # White border
        self.border_alpha = 0.12
        self.blur_radius = 20

        # Text colors (RGB for PIL)
        self.white = (255, 255, 255)
        self.label_color = (255, 255, 255, 140)  # Semi-transparent white
        self.text_color = (255, 255, 255)
        self.dim_color = (180, 180, 180)

        # Font settings
        self.label_size = 11
        self.text_size = 18
        self.emotion_size = 26

        # Load fonts
        self._font_label = self._load_font(self.label_size, bold=True)
        self._font_text = self._load_font(self.text_size)
        self._font_emotion = self._load_font(self.emotion_size, bold=True)

        # State file
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
        self._state_check_interval = 0.3

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Load a TrueType font with fallbacks."""
        if bold:
            font_names = ["segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"]
        else:
            font_names = ["segoeui.ttf", "arial.ttf", "calibri.ttf"]

        font_dirs = ["C:/Windows/Fonts/", "/usr/share/fonts/truetype/", "/System/Library/Fonts/"]

        for font_dir in font_dirs:
            for font_name in font_names:
                font_path = os.path.join(font_dir, font_name)
                if os.path.exists(font_path):
                    try:
                        return ImageFont.truetype(font_path, size)
                    except:
                        continue
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

    def _load_state(self) -> dict:
        """Load conversation state from file."""
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

    def _ease_out_cubic(self, t: float) -> float:
        """Smooth ease-out animation curve."""
        return 1 - pow(1 - t, 3)

    def _ease_in_cubic(self, t: float) -> float:
        """Smooth ease-in animation curve."""
        return t * t * t

    def _update_animation(self, should_show: bool) -> float:
        """Update animation state and return current progress."""
        now = time.time()
        dt = now - self._last_update
        self._last_update = now

        # Clamp dt to reasonable range (handles first frame after long delay)
        dt = min(dt, 0.1)

        # Animation speed
        speed = 1.0 / self.animation_duration

        if should_show:
            self._animation_progress = min(1.0, self._animation_progress + dt * speed)
            return self._ease_out_cubic(self._animation_progress)
        else:
            self._animation_progress = max(0.0, self._animation_progress - dt * speed)
            return self._ease_out_cubic(self._animation_progress)

    def _draw_glassmorphism_panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                                   alpha: float = 1.0) -> np.ndarray:
        """Draw a glassmorphism panel with blur effect."""
        if alpha <= 0:
            return frame

        radius = self.corner_radius

        # Create mask for rounded rectangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        cv2.ellipse(mask, (radius, radius), (radius, radius), 180, 0, 90, 255, -1)
        cv2.ellipse(mask, (w - radius, radius), (radius, radius), 270, 0, 90, 255, -1)
        cv2.ellipse(mask, (radius, h - radius), (radius, radius), 90, 0, 90, 255, -1)
        cv2.ellipse(mask, (w - radius, h - radius), (radius, radius), 0, 0, 90, 255, -1)

        # Extract region and apply blur (frosted glass effect)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Adjust mask to actual region size
        mask_x1, mask_y1 = x1 - x, y1 - y
        mask_x2, mask_y2 = mask_x1 + (x2 - x1), mask_y1 + (y2 - y1)
        region_mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

        # Get region and blur it
        region = frame[y1:y2, x1:x2].copy()
        blurred = cv2.GaussianBlur(region, (21, 21), 0)

        # Create dark tinted overlay
        dark_overlay = np.full_like(region, self.bg_color, dtype=np.uint8)

        # Blend: blurred background + dark tint
        blended = cv2.addWeighted(blurred, 0.3, dark_overlay, 0.7, 0)

        # Apply with mask and alpha
        mask_3ch = cv2.merge([region_mask, region_mask, region_mask]) / 255.0
        mask_3ch = mask_3ch * (self.bg_alpha * alpha)

        frame[y1:y2, x1:x2] = (blended * mask_3ch + region * (1 - mask_3ch)).astype(np.uint8)

        # Draw subtle border
        border_overlay = frame.copy()
        border_pts = []

        # Top edge
        cv2.line(border_overlay, (x + radius, y), (x + w - radius, y), self.border_color, 1)
        # Bottom edge
        cv2.line(border_overlay, (x + radius, y + h - 1), (x + w - radius, y + h - 1), self.border_color, 1)
        # Left edge
        cv2.line(border_overlay, (x, y + radius), (x, y + h - radius), self.border_color, 1)
        # Right edge
        cv2.line(border_overlay, (x + w - 1, y + radius), (x + w - 1, y + h - radius), self.border_color, 1)
        # Corners
        cv2.ellipse(border_overlay, (x + radius, y + radius), (radius, radius), 180, 0, 90, self.border_color, 1)
        cv2.ellipse(border_overlay, (x + w - radius - 1, y + radius), (radius, radius), 270, 0, 90, self.border_color, 1)
        cv2.ellipse(border_overlay, (x + radius, y + h - radius - 1), (radius, radius), 90, 0, 90, self.border_color, 1)
        cv2.ellipse(border_overlay, (x + w - radius - 1, y + h - radius - 1), (radius, radius), 0, 0, 90, self.border_color, 1)

        cv2.addWeighted(border_overlay, self.border_alpha * alpha, frame, 1 - self.border_alpha * alpha, 0, frame)

        return frame

    def _wrap_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def render(self, frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5,
               person_tracked: bool = False) -> np.ndarray:
        """Render overlay panels with animation."""
        h, w = frame.shape[:2]

        # Update animation
        anim_progress = self._update_animation(person_tracked)

        # Skip rendering if fully hidden
        if anim_progress <= 0.01:
            return frame

        # Load state
        state = self._load_state()
        convo_summary = state.get("convo_state_summary") or state.get("conversation_summary") or "Listening..."
        question = state.get("question", "")
        utterance = state.get("recent_utterance", "")
        emotion = state.get("emotion") or state.get("emotion_label") or "Neutral"

        # Calculate panel dimensions
        line_height_text = 26
        line_height_label = 20
        section_spacing = 18

        # Calculate context panel height dynamically
        context_sections = 3  # conversation, question, just said
        context_height = (self.padding_y * 2) + (context_sections * (line_height_label + line_height_text * 2)) + (section_spacing * 2)
        emotion_height = self.padding_y * 2 + line_height_label + 36

        total_height = context_height + self.panel_gap + emotion_height

        # Animated position (slide in from left)
        base_x = 40
        slide_offset = int((1 - anim_progress) * -80)  # Slide from -80px
        panel_x = base_x + slide_offset

        base_y = (h - total_height) // 2

        # Draw glassmorphism panels
        context_y = base_y
        frame = self._draw_glassmorphism_panel(frame, panel_x, context_y,
                                                self.panel_width, context_height, anim_progress)

        emotion_y = context_y + context_height + self.panel_gap
        frame = self._draw_glassmorphism_panel(frame, panel_x, emotion_y,
                                                self.panel_width, emotion_height, anim_progress)

        # Render text with PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Text alpha based on animation
        text_alpha = int(255 * anim_progress)
        white_a = (255, 255, 255, text_alpha)
        label_a = (160, 160, 160, text_alpha)
        dim_a = (140, 140, 140, text_alpha)

        text_x = panel_x + self.padding_x
        max_text_width = self.panel_width - (self.padding_x * 2)

        # === CONTEXT PANEL ===
        y = context_y + self.padding_y

        # CONVERSATION
        draw.text((text_x, y), "CONVERSATION", font=self._font_label, fill=label_a)
        y += line_height_label

        summary_lines = self._wrap_text(draw, convo_summary, self._font_text, max_text_width)
        for line in summary_lines[:2]:
            draw.text((text_x, y), line, font=self._font_text, fill=white_a)
            y += line_height_text
        y += section_spacing

        # QUESTION
        draw.text((text_x, y), "QUESTION", font=self._font_label, fill=label_a)
        y += line_height_label

        q_text = question if question else "No question detected"
        q_fill = white_a if question else dim_a
        q_lines = self._wrap_text(draw, q_text, self._font_text, max_text_width)
        for line in q_lines[:2]:
            draw.text((text_x, y), line, font=self._font_text, fill=q_fill)
            y += line_height_text
        y += section_spacing

        # JUST SAID
        draw.text((text_x, y), "JUST SAID", font=self._font_label, fill=label_a)
        y += line_height_label

        utt_text = f'"{utterance}"' if utterance else "..."
        utt_lines = self._wrap_text(draw, utt_text, self._font_text, max_text_width)
        for line in utt_lines[:2]:
            draw.text((text_x, y), line, font=self._font_text, fill=white_a)
            y += line_height_text

        # === EMOTION PANEL ===
        y = emotion_y + self.padding_y

        draw.text((text_x, y), "EMOTION", font=self._font_label, fill=label_a)
        y += line_height_label + 4

        draw.text((text_x, y), emotion, font=self._font_emotion, fill=white_a)

        # Convert back to BGR
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return frame


# Global renderer
_renderer: Optional[OverlayRenderer] = None


def get_renderer() -> OverlayRenderer:
    """Get or create the overlay renderer."""
    global _renderer
    if _renderer is None:
        _renderer = OverlayRenderer()
    return _renderer


def render_overlay(frame: np.ndarray, head_x: float = 0.5, head_y: float = 0.5,
                   person_tracked: bool = False) -> np.ndarray:
    """Render overlay panels onto frame."""
    renderer = get_renderer()
    return renderer.render(frame, head_x, head_y, person_tracked)


def start_overlay():
    """No-op for backwards compatibility."""
    pass


def stop_overlay():
    """No-op for backwards compatibility."""
    pass


def update_head_position(head_y: float):
    """No-op for backwards compatibility."""
    pass

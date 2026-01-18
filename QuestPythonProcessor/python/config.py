"""
Configuration module for Quest Processor.

This module contains all configuration options, presets, and settings.
Modify ACTIVE_CONFIG to change default behavior, or use command-line args.

To add a new preset:
1. Add entry to PRESETS dict with your settings
2. Optionally set as ACTIVE_PRESET default
"""
from dataclasses import dataclass, field
from typing import Optional
import argparse


# === QUALITY PRESETS ===
# Each preset balances quality vs performance differently
PRESETS = {
    "QUALITY": {
        "processor_width": 640,     # Input size for ML model
        "effect_scale": 1.0,        # Effect resolution multiplier
        "process_skip": 1,          # Process every Nth frame
        "display_skip": 1,          # Display every Nth frame
        "mask_blur": 15,            # Mask edge smoothing
    },
    "STREAM": {
        "processor_width": 320,
        "effect_scale": 1.0,
        "process_skip": 1,
        "display_skip": 3,
        "mask_blur": 15,
    },
    "BALANCED": {
        "processor_width": 320,
        "effect_scale": 0.5,
        "process_skip": 2,
        "display_skip": 2,
        "mask_blur": 11,
    },
    "FAST": {
        "processor_width": 256,
        "effect_scale": 0.25,
        "process_skip": 3,
        "display_skip": 3,
        "mask_blur": 7,
    },
    "EXTREME": {
        "processor_width": 160,
        "effect_scale": 0.25,
        "process_skip": 4,
        "display_skip": 4,
        "mask_blur": 5,
    },
}

# Default preset
ACTIVE_PRESET = "QUALITY"


@dataclass
class Config:
    """Main configuration for the processing pipeline.

    Attributes:
        source: Video source type ("quest", "webcam", "file")
        processor: ML processor type ("yolo", "mediapipe", etc.)
        effect: Visual effect type ("focus", "blur", etc.)
        ui: UI backend ("opencv", "qt", "headless")
        preset: Quality preset name
        auto_start: Start processing automatically
        show_profiling: Show performance stats
        display_max_width: Max display window width
        headless: Run without display
        capture_max_size: Max capture resolution (0 = full)
        transition_duration: Effect transition time in seconds
    """
    # Component selection
    source: str = "quest"
    processor: str = "yolo"
    effect: str = "focus"
    ui: str = "opencv"

    # Preset (overrides individual settings)
    preset: str = ACTIVE_PRESET

    # Behavior
    auto_start: bool = True
    show_profiling: bool = True
    headless: bool = False

    # Display
    display_max_width: int = 1920
    profile_interval: int = 30

    # Capture
    capture_max_size: int = 0  # 0 = full resolution
    capture_fps: int = 60

    # Transition
    transition_duration: float = 1.0

    # Computed from preset (set in __post_init__)
    processor_width: int = field(default=640, init=False)
    effect_scale: float = field(default=1.0, init=False)
    process_skip: int = field(default=1, init=False)
    display_skip: int = field(default=1, init=False)
    mask_blur: int = field(default=15, init=False)

    def __post_init__(self):
        """Apply preset settings."""
        if self.preset in PRESETS:
            preset = PRESETS[self.preset]
            self.processor_width = preset["processor_width"]
            self.effect_scale = preset["effect_scale"]
            self.process_skip = preset["process_skip"]
            self.display_skip = preset["display_skip"]
            self.mask_blur = preset["mask_blur"]


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from command-line args or defaults.

    Args:
        args: Parsed command-line arguments, or None for defaults

    Returns:
        Config object with all settings
    """
    if args is None:
        return Config()

    return Config(
        source=getattr(args, 'source', 'quest'),
        processor=getattr(args, 'processor', 'yolo'),
        effect=getattr(args, 'effect', 'focus'),
        ui=getattr(args, 'ui', 'opencv'),
        preset=getattr(args, 'preset', ACTIVE_PRESET),
        auto_start=getattr(args, 'auto_start', True),
        headless=getattr(args, 'headless', False),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Quest Processor - Modular ML video processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Default settings
  python main.py --preset FAST             # Use FAST preset
  python main.py --processor yolo          # Explicit processor
  python main.py --headless                # No display window
        """
    )

    parser.add_argument(
        '--source', '-s',
        choices=['quest', 'quest_tcp', 'webcam', 'file'],
        default='quest',
        help='Video input source (default: quest). Use quest_tcp for Unity passthrough.'
    )

    parser.add_argument(
        '--processor', '-p',
        choices=['yolo'],  # Add more as implemented
        default='yolo',
        help='ML processor for frame analysis (default: yolo)'
    )

    parser.add_argument(
        '--effect', '-e',
        choices=['focus'],  # Add more as implemented
        default='focus',
        help='Visual effect to apply (default: focus)'
    )

    parser.add_argument(
        '--ui', '-u',
        choices=['opencv', 'headless', 'webview', 'quest_tcp'],
        default='opencv',
        help='UI backend (default: opencv). Use quest_tcp to send frames back to Unity.'
    )

    parser.add_argument(
        '--preset',
        choices=list(PRESETS.keys()),
        default=ACTIVE_PRESET,
        help=f'Quality preset (default: {ACTIVE_PRESET})'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display window'
    )

    parser.add_argument(
        '--no-auto-start',
        action='store_true',
        help='Do not start processing automatically'
    )

    return parser.parse_args()

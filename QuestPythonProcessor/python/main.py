#!/usr/bin/env python3
"""
Quest Processor - Modular ML Video Processing Pipeline

A modular, plugin-based video processing system for Quest VR headsets.
Easily extensible with new ML models, visual effects, and UI backends.

Usage:
    python main.py                        # Default settings (QUALITY preset)
    python main.py --preset FAST          # Use FAST preset for speed
    python main.py --processor yolo       # Explicit processor selection
    python main.py --headless             # Run without display

To add a new component:
    - Processor: See processors/base.py
    - Effect: See effects/base.py
    - Source: See sources/base.py
    - UI: See ui/base.py

Architecture:
    main.py
    ├── config.py           # Configuration and presets
    ├── core/
    │   ├── pipeline.py     # Main processing loop
    │   └── transition.py   # Effect transition manager
    ├── sources/            # Video input sources
    ├── processors/         # ML models (YOLO, etc.)
    ├── effects/            # Visual effects
    ├── controls/           # Input controls
    └── ui/                 # Display backends
"""
from config import Config, parse_args, load_config
from core.pipeline import Pipeline
from sources import get_source
from processors import get_processor
from effects import get_effect
from ui import get_ui


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args)

    # Handle headless mode
    if args.headless:
        config.headless = True
        config.ui = 'headless'

    # Auto-select quest_tcp UI when using quest_tcp source (for bidirectional communication)
    if config.source == 'quest_tcp' and config.ui == 'opencv':
        config.ui = 'quest_tcp'
        print("[AUTO] Using quest_tcp UI for bidirectional Quest communication")

    if args.no_auto_start:
        config.auto_start = False

    # Note: Overlay panels (context/emotion) are now integrated directly
    # into the UI backends (OpenCVUI, QuestTCPUI) and launched automatically.
    # The separate test_webview.py helper is no longer needed.

    # Initialize components
    print(f"Initializing with preset: {config.preset}")
    print(f"  Source: {config.source}")
    print(f"  Processor: {config.processor}")
    print(f"  Effect: {config.effect}")
    print(f"  UI: {config.ui}")
    print()

    source = get_source(config.source, config)
    processor = get_processor(config.processor, config)
    effect = get_effect(config.effect, config)
    ui = get_ui(config.ui, config)

    # Create and run pipeline
    pipeline = Pipeline(source, processor, effect, ui, config)
    pipeline.run()


if __name__ == "__main__":
    main()

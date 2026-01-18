# Perceptual Modulation Engine for Meta Quest 3 Passthrough

A real-time perceptual modulation system for neurodivergent and neurodegenerative users. This system modifies live passthrough perception (video + audio) from Meta Quest 3 in a semantically grounded, reversible, and sensory-safe way.

## 🎯 Design Principles

### Top Priorities (Strict Order)
1. **Sensory safety and user comfort** - No sudden changes, no full darkness, no full silence
2. **Deterministic, explainable behavior** - Every transformation can be understood and predicted
3. **Object-level semantic control** - Never pixel hacks, always mask-based operations
4. **Temporal stability** - No flicker, no abrupt transitions, smooth easing everywhere
5. **Local-first inference** - All processing happens on-device with modular extensibility

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Acquire        2. Segment      3. Track        4. Depth    │
│  ┌─────────┐      ┌─────────┐     ┌─────────┐    ┌─────────┐  │
│  │ Video   │──────│  SAM-3  │─────│ Kalman  │────│  MiDaS  │  │
│  │ + Audio │      │         │     │ Filter  │    │         │  │
│  └─────────┘      └─────────┘     └─────────┘    └─────────┘  │
│       │                │               │              │        │
│       v                v               v              v        │
│  5. Bind AV       6. Parse       7. Resolve      8. Transform │
│  ┌─────────┐      ┌─────────┐     ┌─────────┐    ┌─────────┐  │
│  │ Spatial │──────│  NLP    │─────│ Target  │────│ Visual  │  │
│  │ Binding │      │ Intent  │     │ Resolve │    │ + Audio │  │
│  └─────────┘      └─────────┘     └─────────┘    └─────────┘  │
│                                                       │        │
│                    9. Safety Layer                    │        │
│                    ┌────────────────────────────────┐│        │
│                    │ • Rate limiting                ││        │
│                    │ • Minimum transitions          ││        │
│                    │ • Sensory load monitoring      ││        │
│                    │ • Emergency revert             ││        │
│                    └────────────────────────────────┘│        │
│                                                       │        │
│                    10. Render Output ◄───────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Meta Quest 3 with passthrough enabled as virtual webcam

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/SAMIntegrationAdvanced.git
cd SAMIntegrationAdvanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SAM2 model (optional - system works with placeholder)
# Follow instructions at https://github.com/facebookresearch/segment-anything-2

# Download spaCy model for NLP
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

```bash
# Run with default settings
python main.py

# Run with custom config
python main.py --config config/settings.yaml

# Run with specific camera device
python main.py --device 1
```

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `R` | Toggle recording |
| `P` | Process/playback last recording |
| `ESC` | Emergency revert (restore unmodified passthrough) |
| `/` | Enter command mode |
| `Q` | Quit |

## 💬 Voice/Text Commands

The system understands natural language commands:

```
"dim the person to my right"
"mute the screen on the left"
"blur the bright light in the center"
"make everything slightly less saturated"
"reduce the volume of the person speaking"
```

### Command Structure
- **Target**: object, person, region (with spatial relations like left/right/near/far)
- **Modality**: visual or audio
- **Attribute**: brightness, color, volume, texture
- **Magnitude**: slightly, much, a lot, completely

## 🔒 Safety Constraints (Non-Negotiable)

### Hard Constraints
- All visual edits MUST originate from segmentation masks
- All audio edits MUST be spatially or identity grounded
- No global brightness/contrast/volume changes unless explicitly requested
- All changes MUST be temporally eased (no step functions)

### Latency Budgets
- Video: ≤ 120ms end-to-end
- Audio: ≤ 40ms end-to-end

### Sensory Guardrails
- **Minimum brightness**: 10% (never fully dark)
- **Minimum volume**: 5% (never fully silent)
- **Maximum blur radius**: 30px
- **Minimum transition duration**: 300ms
- **Maximum change rate**: 15-20% per second

## 📁 Project Structure

```
SAMIntegrationAdvanced/
├── config/
│   └── settings.yaml          # Configuration
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── contracts.py       # Data structures and types
│   ├── capture/
│   │   ├── video_capture.py   # Quest 3 video input
│   │   └── frame_buffer.py    # Recording buffer
│   ├── segmentation/
│   │   ├── sam_segmenter.py   # SAM-3 integration
│   │   └── mask_processor.py  # Mask cleaning/smoothing
│   ├── tracking/
│   │   ├── object_tracker.py  # Persistent ID tracking
│   │   └── kalman_tracker.py  # Trajectory smoothing
│   ├── depth/
│   │   └── depth_estimator.py # MiDaS depth estimation
│   ├── audio/
│   │   ├── audio_processor.py # Audio capture
│   │   ├── audio_visual_binder.py
│   │   └── audio_transformer.py
│   ├── intent/
│   │   ├── intent_parser.py   # NLP command parsing
│   │   └── target_resolver.py # Spatial target resolution
│   ├── transforms/
│   │   └── visual_transformer.py
│   ├── safety/
│   │   ├── safety_layer.py    # Core guardrails
│   │   └── sensory_monitor.py # Load tracking
│   └── pipeline/
│       └── orchestrator.py    # Main pipeline
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

```yaml
# Latency budgets
latency:
  video_max_ms: 120
  audio_max_ms: 40

# Safety constraints
safety:
  max_brightness_delta_per_second: 0.15
  min_transition_seconds: 0.3
  min_brightness: 0.1
  min_volume: 0.05

# Visual transforms
visual_transforms:
  allowed:
    - brightness_attenuation
    - color_temperature_shift
    - saturation_reduction
    - texture_simplification
    - edge_preserving_blur
```

## 🔌 Allowed Operations

### Visual (Mask-Confined)
| Operation | Description | Parameters |
|-----------|-------------|------------|
| `brightness_attenuation` | Dim/brighten region | factor (0.1-1.0+) |
| `color_temperature_shift` | Warm/cool color | shift (-1 to 1) |
| `saturation_reduction` | Reduce color intensity | factor (0-1) |
| `texture_simplification` | Reduce visual detail | strength (0-1) |
| `edge_preserving_blur` | Smooth while keeping edges | radius (0-30px) |

### Audio (Identity-Grounded)
| Operation | Description | Parameters |
|-----------|-------------|------------|
| `selective_muting` | Reduce volume of source | factor (0-1) |
| `volume_attenuation` | Adjust source volume | factor (0.05-1) |
| `frequency_filtering` | Filter harsh/bass/mids | preset, strength |
| `directional_dampening` | Attenuate from direction | azimuth, strength |

### Disallowed by Default
- Object removal
- Geometry distortion
- Flashing effects
- High-contrast overlays
- Full darkness/silence

## 🧪 Development

### Running Tests
```bash
pytest tests/ -v
```

### Adding New Transforms

1. Define the operation in `src/core/contracts.py`
2. Implement in `src/transforms/visual_transformer.py` or `src/audio/audio_transformer.py`
3. Add safety constraints in `src/safety/safety_layer.py`
4. Add NLP patterns in `src/intent/intent_parser.py`

### Extensibility Contract

All components must be:
- **Modular**: Single responsibility
- **Swappable**: Interface-based
- **Testable**: Independent unit tests

New models must:
- Expose confidence scores
- Respect latency constraints
- Integrate through defined interfaces

## 📊 Performance

Target performance on recommended hardware:

| Metric | Target | Typical |
|--------|--------|---------|
| Frame latency | <120ms | 40-80ms |
| Audio latency | <40ms | 10-20ms |
| Segmentation | <50ms | 30-40ms |
| Tracking | <5ms | 2-3ms |
| Transforms | <10ms | 5-8ms |

## 🤝 Contributing

When contributing, remember the core behavioral rule:

> **When uncertain: Do less, not more. Ask for clarification. Preserve the user's baseline perception.**

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [Meta Quest 3](https://www.meta.com/quest/quest-3/)



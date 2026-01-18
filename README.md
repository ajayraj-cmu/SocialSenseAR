# SocialSenseAR

Real-time AR environment modifier with voice control, using SAM (Segment Anything Model), Gemini Vision, and sensory modulation features.

## 🚀 Quick Start

### Main Application (Voice-Controlled)

```bash
python scripts/sam_gemini_voice.py
```

**Voice Commands:**
- Say **"hey vibe"** to start recording
- Say your command (e.g., "blur my face", "dim the ceiling")
- Say **"thanks"** to process

### Alternative Entry Point

```bash
python main.py
```

## 📁 Project Structure

```
SocialSenseAR/
├── main.py                 # Main entry point (perceptual modulation engine)
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
├── src/                     # Core source code
│   ├── audio/              # Audio processing
│   ├── capture/            # Video capture
│   ├── core/               # Core contracts and types
│   ├── depth/              # Depth estimation
│   ├── intent/             # NLP and intent parsing
│   ├── pipeline/           # Main pipeline orchestrator
│   ├── safety/             # Safety layer and monitoring
│   ├── segmentation/       # SAM segmentation
│   ├── tracking/           # Object tracking
│   ├── transforms/         # Visual transformations
│   └── voice/              # Voice command processing
├── scripts/                # Standalone scripts and demos
│   ├── sam_gemini_voice.py # Main voice-controlled app ⭐
│   ├── sam_*.py            # Various SAM demo scripts
│   └── fast_*.py           # FastSAM scripts
├── docs/                   # Documentation
│   ├── README.md           # Main documentation
│   ├── PIPELINE_DOCUMENTATION.md
│   ├── FEEDBACK_LOOP_DOCUMENTATION.md
│   └── *.md                # Other documentation
├── models/                 # Model weights
│   ├── FastSAM-s.pt
│   ├── yolov8*.pt
│   └── sam_*.pth
├── assets/                 # Images, HTML, etc.
├── logs/                   # Log files
└── recordings/             # Audio/video recordings
```

## 🎯 Features

- **Voice Control**: Wake word activation ("hey vibe" / "thanks")
- **Real-time Segmentation**: FastSAM for object segmentation
- **Smart Labeling**: Gemini Vision API for open-vocabulary detection
- **Sensory Modulation**: Blur, brightness, color, motion dampening
- **Persistent Tracking**: Masks track objects during movement
- **Clean View Mode**: Toggle between full view and effects-only

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` (FastSAM, YOLO)
- `google-generativeai` (Gemini API)
- `speech_recognition` (Voice commands)
- `mediapipe` (Body part segmentation)
- `opencv-python` (Video processing)

## 🔧 Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your-gemini-api-key
OVERSHOOT_API_KEY=your-overshoot-api-key  # Optional
```

## 📖 Documentation

See `docs/` folder for detailed documentation:
- `PIPELINE_DOCUMENTATION.md` - Full pipeline architecture
- `FEEDBACK_LOOP_DOCUMENTATION.md` - Self-correction system
- `USAGE_GUIDE.md` - Usage instructions

## ⌨️ Controls

- **V** - Toggle clean/full view
- **C** - Clear all effects
- **L** - List all detected labels
- **S** - Screenshot
- **Q** - Quit

## 🤝 Contributing

See the main documentation in `docs/` for architecture details and contribution guidelines.

## 📄 License

MIT License


# Advanced Sensory Modulation System - Complete Pipeline Documentation

## Overview

This system provides real-time, object-local sensory modulation for neurodivergent users experiencing sensory overload. It combines optimized computer vision (YOLO/SAM), continuous scene understanding (Gemini Vision), and natural language processing to dynamically adjust the visual environment.

---

## System Architecture

### 1. Detection Pipeline (Optimized YOLO/SAM)

**Components:**
- **YOLO v8**: Object detection with optimized confidence thresholds (0.2)
- **FastSAM**: Instance segmentation with 512px resolution
- **MediaPipe**: Body part segmentation (face, hands, arms, legs, torso)
- **Matching Algorithm**: Multi-factor matching (IoU + center overlap + pixel overlap)

**Improvements:**
- Lower YOLO threshold (0.2) for better recall
- Higher SAM resolution (512px) for accuracy
- Improved matching algorithm with 3-factor scoring
- Semantic fallback labeling for unmatched objects
- Window detection (bright, rectangular, mid-height)

**Output:** List of `SegmentedObject` with:
- Mask (numpy array)
- Label (from YOLO or semantic)
- Confidence score
- Bounding box
- Center coordinates
- Feature vector (for modulation)

---

### 2. Continuous Scene Analysis (Gemini Vision)

**Process:**
1. **Every 1 second**: Capture current frame
2. **Send to Gemini Vision** with:
   - Current frame (640x480 thumbnail)
   - List of all detected objects with positions
   - Context about scene
3. **Gemini analyzes**:
   - What objects are visible
   - Their relationships
   - Environmental context (lighting, motion, etc.)
4. **Store context** for natural language understanding

**Benefits:**
- Real-time scene understanding
- Context-aware object identification
- Better handling of complex requests

---

### 3. Natural Language Understanding

**Capabilities:**
- **Simple requests**: "Make the wall blue"
- **Complex requests**: "I'm overstimulated by the sunlight coming through the windows"
- **Multi-object**: "Dim the laptop and reduce the brightness of the screen"
- **Emotional/Contextual**: "The lights are too bright", "I need less visual noise"

**Processing Flow:**
1. User speaks → Google Speech-to-Text
2. Gemini Text model extracts:
   - Target objects (windows, lights, screens, etc.)
   - Desired modulation (dim, reduce brightness, soften, etc.)
   - Urgency/context (overstimulated, too bright, etc.)
3. Gemini Vision maps to actual objects in scene
4. Generate feature vector modifications
5. Apply to correct masks

**Example Processing:**
```
Input: "I'm overstimulated by the sunlight coming through the windows"
  ↓
Gemini Text: Identifies "windows" + "sunlight" + "overstimulated"
  ↓
Gemini Vision: Finds window objects in scene
  ↓
Modulation: brightness: 0.3, highlight_suppression: 0.8, saturation: 0.5
  ↓
Apply to window masks
```

---

### 4. Sensory Modulation Features

All features are **object-local** (never global) and **reversible**.

#### 4.1 Luminance & Energy Features

**Brightness (Local Luminance Attenuation)**
- **What**: Reduces light intensity of specific objects
- **Why**: Bright lights are #1 sensory overload trigger
- **Implementation**: HSV/LAB space luminance scaling
- **Constraints**: Max 15% change/sec, never pure black
- **Targets**: Windows, screens, lights, reflective surfaces

**Contrast (Local Contrast Compression)**
- **What**: Compresses high-contrast edges
- **Why**: Reduces visual "vibration"
- **Implementation**: CLAHE with capped parameters
- **Targets**: Screens, printed text, patterned clothing

**Highlight Suppression (Specular Dampening)**
- **What**: Reduces glare and specular highlights
- **Why**: Glare is disproportionately overstimulating
- **Implementation**: Saturated pixel detection + soft rolloff
- **Targets**: Windows, screens, glossy surfaces

#### 4.2 Color Features

**Color Temperature Shift**
- **What**: Moves light toward warmer/cooler tones
- **Why**: Warm light is calming, cool light can be alerting
- **Implementation**: White balance adjustment in mask
- **Constraints**: Limited Kelvin shift, smooth transitions
- **Range**: -2000K to +2000K

**Saturation Reduction**
- **What**: Reduces color intensity
- **Why**: Highly saturated colors are overstimulating
- **Implementation**: HSV saturation scaling
- **Targets**: LED lights, clothing, advertisements

**Hue Softening (Selective)**
- **What**: Nudges problematic hues (harsh reds, etc.)
- **Why**: Certain hues trigger anxiety
- **Constraints**: Small shifts only, no cycling

#### 4.3 Detail & Texture Features

**Texture Simplification**
- **What**: Reduces fine texture detail
- **Why**: Busy textures increase cognitive load
- **Implementation**: Bilateral filtering, guided smoothing
- **Targets**: Carpets, walls, clothing, furniture

**Edge Softening (Edge-Preserving Blur)**
- **What**: Softens harsh edges without losing shape
- **Why**: Sharp edges create visual tension
- **Implementation**: Guided filter / domain transform
- **Never**: Full Gaussian blur (causes nausea)

**Pattern Attenuation**
- **What**: Dampens repeating patterns
- **Why**: Patterns cause visual overstimulation
- **Implementation**: Frequency-domain suppression

#### 4.4 Motion Features

**Motion Dampening (Per Object)**
- **What**: Reduces perceived motion intensity
- **Why**: Sudden movement is highly triggering
- **Implementation**: Optical flow estimation + magnitude scaling
- **Targets**: Moving people, swinging objects, screens

**Temporal Smoothing**
- **What**: Reduces flicker and jitter
- **Why**: Flicker is exhausting
- **Implementation**: Rolling temporal filters, mask persistence

#### 4.5 Saliency & Attention Features

**Saliency Reduction**
- **What**: Makes objects less attention-grabbing
- **Why**: Prevents fixation or distraction
- **Implementation**: Combined brightness + saturation + contrast reduction

**Depth-Based Attenuation**
- **What**: Objects closer to user are softened more
- **Why**: Near-field stimuli are more intense
- **Implementation**: Depth-weighted blending

#### 4.6 Specialized Object Features

**Faces (People)**
- **Allowed**: Brightness reduction, texture smoothing, edge softening
- **Disallowed**: Face warping, identity alteration
- **Reason**: Avoid uncanny valley, preserve social trust

**Screens (Phones, TVs, Monitors)**
- **Allowed**: Brightness clamp, blue light reduction, contrast compression
- **Highly recommended**: Aggressive safety limits

**Light Sources**
- **Allowed**: Intensity attenuation, color temperature shift, bloom suppression
- **Never**: Turn off entirely without confirmation

---

## Feature Vector Structure

Each segmented object has a feature vector:

```python
{
  "brightness": 0.7,           # 0.0-1.0 (1.0 = no change)
  "contrast": 0.85,            # 0.0-1.0
  "saturation": 0.6,           # 0.0-1.0
  "color_temp": -500.0,        # Kelvin shift (-2000 to +2000)
  "texture_detail": 0.4,       # 0.0-1.0 (1.0 = full detail)
  "motion_scale": 0.5,         # 0.0-1.0 (1.0 = full motion)
  "transition_time": 1.2,      # seconds
  "edge_softness": 0.3,        # 0.0-1.0
  "highlight_suppression": 0.8  # 0.0-1.0
}
```

**Default values**: All 1.0 (no change) or 0.0 (neutral)

---

## Design Principles

### Core Rules:
1. **Everything is object-local** (never global by default)
2. **All changes are continuous** (not discrete)
3. **All features must be reversible**
4. **No geometry hallucination**
5. **No sudden luminance or contrast spikes**

### Safety Constraints:
- **Max brightness delta**: 15% per second
- **Max volume delta**: 20% per second (for audio)
- **Max saturation delta**: 15% per second
- **Min brightness**: 0.1 (never pure black)
- **Min transition time**: 0.3 seconds
- **Default transition**: 0.5 seconds
- **Max transition**: 3.0 seconds

---

## Processing Pipeline

### Frame-by-Frame Processing:

```
1. Capture Frame
   ↓
2. Detect Objects (YOLO + SAM + MediaPipe)
   ↓
3. Match Objects to Labels
   ↓
4. Update Object Tracking
   ↓
5. Check for Voice Commands
   ↓
6. Process Natural Language Request
   ↓
7. Gemini Vision Maps to Objects
   ↓
8. Generate Feature Vector Modifications
   ↓
9. Apply Sensory Modulations
   ↓
10. Render Frame with Outlines
   ↓
11. Display
```

### Continuous Background Tasks:

- **Every 1 second**: Gemini Vision scene analysis
- **Every 5 frames**: Update SAM masks
- **Every frame**: Update MediaPipe body parts
- **On voice command**: Process request immediately

---

## Natural Language Examples

### Simple Requests:
- "Make the wall blue"
- "Dim the laptop"
- "Turn my face green"

### Complex Requests:
- "I'm overstimulated by the sunlight coming through the windows"
  → Windows: brightness↓, highlight_suppression↑, saturation↓

- "The screen is too bright and the pattern on the carpet is distracting"
  → Screen: brightness↓, contrast↓
  → Carpet: texture_detail↓, pattern_attenuation↑

- "I need less visual noise from the lights and the moving people"
  → Lights: brightness↓, saturation↓
  → People: motion_scale↓, saliency↓

- "The colors are too intense and the edges are too sharp"
  → All objects: saturation↓, edge_softness↑

---

## Object Detection Accuracy Improvements

### Before:
- Generic labels: `object_1`, `item_center`
- Poor YOLO matching (high threshold)
- No window detection
- Limited body part segmentation

### After:
- **YOLO labels**: `laptop`, `cup`, `chair`, `window`, etc.
- **Lower threshold** (0.2) for better recall
- **Window detection**: Bright, rectangular, mid-height objects
- **Full body parts**: face, hands, arms, legs, torso
- **Improved matching**: 3-factor scoring (IoU + center + pixels)
- **Semantic fallback**: Better labeling when YOLO doesn't match

---

## Implementation Status

✅ **Completed:**
- Optimized YOLO/SAM pipeline
- Object feature vector structure
- Basic sensory modulation framework
- Gemini Vision integration structure

🚧 **In Progress:**
- Complete sensory modulation implementations
- Natural language understanding for complex requests
- Continuous scene analysis (every second)
- Object tracking and persistence

📋 **To Do:**
- Motion detection and dampening
- Temporal smoothing
- Depth estimation integration
- Audio-visual binding (future)

---

## Usage

### Running the System:

```bash
cd /Users/ajayraj/SAMIntegrationAdvanced
export GOOGLE_API_KEY="your-api-key"
python sam_advanced_sensory.py
```

### Voice Commands:
- **SPACE**: Record voice command
- **L**: List all detected objects
- **C**: Clear all modulations
- **Q**: Quit

### Example Commands:
1. "I'm overstimulated by the sunlight" → Windows dimmed
2. "The screen is too bright" → Screen brightneess reduced
3. "Make the wall blue" → Wall colored blue
4. "Reduce visual noise" → All objects: saturation↓, texture↓

---

## Technical Details

### Models Used:
- **YOLO v8n**: Object detection
- **FastSAM-s**: Instance segmentation
- **MediaPipe**: Body part segmentation
- **Gemini 1.5 Flash**: Vision and text understanding

### Performance:
- **Detection**: ~30 FPS (with frame skipping)
- **Scene Analysis**: Every 1 second (async)
- **Voice Processing**: Real-time (push-to-talk)

### Safety:
- All modulations are reversible
- No permanent changes
- Smooth transitions only
- Respects sensory limits

---

## Future Enhancements

1. **Audio Modulation**: Selective muting, volume attenuation
2. **Depth Integration**: Use Quest 3 depth sensor
3. **Learning**: Remember user preferences
4. **Presets**: Save/load modulation profiles
5. **Multi-modal**: Combine visual + audio + haptic feedback

---

## References

- FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
- YOLO: https://github.com/ultralytics/ultralytics
- MediaPipe: https://mediapipe.dev/
- Gemini: https://ai.google.dev/

---

*Last Updated: 2025-01-16*


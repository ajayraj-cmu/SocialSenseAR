# 🎙️ Advanced Sensory Modulation System - Complete Usage Guide

## 🚀 How to Run

### Step 1: Set Your API Key
```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

**To make it permanent**, add to your `~/.zshrc`:
```bash
echo 'export GOOGLE_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Run the Application
```bash
cd /Users/ajayraj/SAMIntegrationAdvanced
python sam_gemini_voice.py
```

**Or with conda environment:**
```bash
conda activate JuneBrainEyeTracker
cd /Users/ajayraj/SAMIntegrationAdvanced
python sam_gemini_voice.py
```

### Step 3: Wait for Initialization
You'll see:
```
============================================================
  🎙️ SAM + GEMINI VOICE CONTROLLER
============================================================

Loading models...
  ✓ FastSAM
  ✓ YOLO
  ✓ MediaPipe (selfie + face + hands + pose)
✅ Gemini initialized
🎤 Calibrating microphone...
✅ Voice listener ready (press SPACE to speak)
✅ Ready!
```

A camera window will open showing your live feed with object outlines.

---

## 🎮 Controls & Interaction

### Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| **SPACE** | Record Voice Command | Press and hold, speak your command, release |
| **L** | List All Objects | Shows all detected objects in terminal |
| **C** | Clear All Effects | Removes all color/modulation effects |
| **S** | Screenshot | Saves current frame as PNG |
| **Q** | Quit | Exit the application |

---

## 🗣️ Voice Commands You Can Use

### Simple Color Commands

**Basic syntax:** `"[object] [color]"` or `"make [object] [color]"`

| Command | What It Does |
|---------|--------------|
| `"Make the wall blue"` | Colors the wall blue |
| `"Turn the chair red"` | Colors the chair red |
| `"Make my face green"` | Colors your face green |
| `"Turn the laptop yellow"` | Colors the laptop yellow |
| `"Make the floor purple"` | Colors the floor purple |

### Multiple Objects at Once

| Command | What It Does |
|---------|--------------|
| `"Make the wall blue and the chair red"` | Colors both objects |
| `"Turn the laptop green and dim the screen"` | Colors laptop, dims screen |
| `"Make my face blue and my hands yellow"` | Colors face and hands |

### Dimming & Brightness

| Command | What It Does |
|---------|--------------|
| `"Dim the wall"` | Darkens the wall |
| `"Darken the background"` | Darkens the background |
| `"Make the screen darker"` | Reduces screen brightness |
| `"Dim everything"` | Darkens all objects |

### Complex Natural Language Requests

**The system understands context and emotion!**

| Command | What It Does |
|---------|--------------|
| `"I'm overstimulated by the sunlight coming through the windows"` | Finds windows, dims them, reduces glare |
| `"The screen is too bright and distracting"` | Reduces screen brightness and contrast |
| `"I need less visual noise"` | Reduces saturation and texture detail |
| `"The lights are too bright"` | Finds light sources, dims them |
| `"The colors are too intense"` | Reduces saturation across objects |
| `"I'm overwhelmed by the moving people"` | Reduces motion visibility |

### Body Parts

| Command | What It Does |
|---------|--------------|
| `"Make my face blue"` | Colors your face |
| `"Turn my left hand red"` | Colors left hand |
| `"Make my right hand green"` | Colors right hand |
| `"Dim my torso"` | Darkens your torso |
| `"Color my arms yellow"` | Colors your arms |

### Clearing Effects

| Command | What It Does |
|---------|--------------|
| `"Clear everything"` | Clears all effects |
| `"Reset"` | Resets all modulations |
| `"Turn off effects"` | Removes all changes |

---

## 📺 What You'll See on Screen

### Visual Elements:

1. **Green Boxes**: YOLO object detections with labels and confidence
   - Example: `laptop 87%`, `chair 92%`, `person 95%`

2. **Colored Outlines**: SAM segmentation masks
   - Each detected object has a colored outline
   - Different colors for different objects

3. **Yellow Outlines**: Active effects
   - Objects with active modulations show bright yellow outlines

4. **Labels**: Object names at center of each mask
   - Shows what each object is (e.g., "wall", "laptop", "face")

5. **Status Bar** (top of screen):
   - FPS counter
   - Object count
   - Voice status
   - Last command heard

6. **Active Effects Panel** (left side):
   - Lists all active modulations
   - Format: `object_name -> effect`

7. **Help Bar** (bottom):
   - Keyboard shortcuts reminder

---

## 🎯 Step-by-Step Usage Example

### Example Session:

1. **Launch the app** → Camera window opens

2. **See what's detected**:
   - Press **L** to see all objects in terminal
   - Example output:
     ```
     📋 ALL AVAILABLE LABELS:
       1. person
       2. face
       3. left_hand
       4. right_hand
       5. wall
       6. laptop
       7. chair
       8. window
     ```

3. **Give a voice command**:
   - Press and **hold SPACE**
   - Say: *"I'm overstimulated by the sunlight coming through the windows"*
   - **Release SPACE**
   - Status shows: `🔄 Processing...`
   - Then: `✅ Got: I'm overstimulated...`

4. **See the result**:
   - Terminal shows: `👁️ Gemini Vision: Sees 1 target(s): ['window']`
   - Terminal shows: `✅ Applied brightness reduction to 'window'`
   - Window outline turns **yellow** (active effect)
   - Window becomes dimmer on screen

5. **Add more effects**:
   - Press **SPACE** again
   - Say: *"Dim the laptop too"*
   - Both window and laptop now have effects

6. **Clear when done**:
   - Press **C** to clear all effects
   - Or say: *"Clear everything"*

---

## 🔍 Understanding the Output

### Terminal Messages:

**When you speak:**
```
🎯 Processing: 'make the wall blue'
📋 Available labels: ['person', 'face', 'wall', 'laptop', 'chair']
👁️ Using Gemini Vision to analyze scene...
👁️ Gemini Vision: Sees 1 target(s): ['wall']
  🎯 Gemini picked: 'wall' -> blue (conf: 90%)
  ✅ Applied blue to 'wall'
🎨 Applied 1 effect(s)
```

**When scene is analyzed (every second):**
```
📊 Scene context: ['bright sunlight', 'moving people']
```

**If object not found:**
```
❓ Couldn't determine target(s). Available: ['person', 'wall', 'laptop']
```

---

## 💡 Tips for Best Results

### 1. **Be Specific**
- ✅ Good: "Make the laptop blue"
- ❌ Vague: "Make it blue" (what is "it"?)

### 2. **Use Natural Language**
- ✅ Good: "I'm overstimulated by the bright screen"
- ✅ Good: "The sunlight is too intense"
- ✅ Good: "Dim the distracting lights"

### 3. **Wait for Processing**
- After speaking, wait for `✅ Got: ...` before speaking again
- Status bar shows current state

### 4. **Check Available Objects**
- Press **L** to see what objects are detected
- Use exact object names from the list

### 5. **Multiple Commands**
- Each SPACE press = new command
- Effects accumulate (don't replace each other)
- Press **C** to clear all

### 6. **Complex Requests Work Best**
- The system understands context
- "I'm overstimulated by X" → finds X and applies appropriate modulations
- More descriptive = better results

---

## 🐛 Troubleshooting

### Camera Not Opening?
- Check camera permissions
- Make sure no other app is using the camera
- Try: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### Voice Not Working?
- Check microphone permissions
- Make sure microphone is not muted
- Try speaking louder or closer to mic
- Check terminal for: `✅ Voice listener ready`

### Objects Not Detected?
- Ensure good lighting
- Objects should be clearly visible
- Press **L** to see what's detected
- Some objects may need better lighting or closer view

### Effects Not Applying?
- Check terminal for error messages
- Make sure you're using exact object names (press **L** to see)
- Try simpler commands first: "make wall blue"
- Check that Gemini API key is set correctly

### API Errors?
- Verify: `echo $GOOGLE_API_KEY` shows your key
- Check internet connection
- Gemini API may have rate limits

---

## 📊 What the System Does Automatically

### Every Frame:
- Updates MediaPipe body parts (face, hands, arms, legs)
- Tracks person mask
- Renders effects and outlines

### Every 5 Frames:
- Updates SAM segmentation masks
- Updates YOLO object detections
- Matches objects to labels

### Every 1 Second:
- **Gemini Vision analyzes scene** (background)
- Identifies light sources, motion, sensory triggers
- Updates scene context for better understanding

### On Voice Command:
- Processes speech → text
- Gemini Vision maps request to objects
- Applies modulations
- Updates display

---

## 🎨 Example Use Cases

### Use Case 1: Reducing Screen Brightness
1. Say: *"The screen is too bright"*
2. System finds screen/monitor
3. Applies: brightness↓, contrast↓
4. Screen becomes more comfortable

### Use Case 2: Dimming Windows
1. Say: *"I'm overstimulated by the sunlight"*
2. System finds windows
3. Applies: brightness↓, highlight_suppression↑
4. Windows become less intense

### Use Case 3: Reducing Visual Noise
1. Say: *"I need less visual noise"*
2. System applies to all objects:
   - Saturation↓
   - Texture detail↓
   - Contrast↓
3. Scene becomes calmer

### Use Case 4: Multiple Adjustments
1. Say: *"Dim the laptop and reduce the brightness of the screen"*
2. System finds both objects
3. Applies appropriate modulations
4. Both become more comfortable

---

## 🎯 Quick Reference Card

```
┌─────────────────────────────────────────┐
│  KEYBOARD SHORTCUTS                     │
├─────────────────────────────────────────┤
│  SPACE  → Record voice command          │
│  L      → List all objects              │
│  C      → Clear all effects             │
│  S      → Screenshot                    │
│  Q      → Quit                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  EXAMPLE COMMANDS                       │
├─────────────────────────────────────────┤
│  "Make the wall blue"                  │
│  "Dim the laptop"                      │
│  "I'm overstimulated by sunlight"      │
│  "The screen is too bright"            │
│  "Make my face green"                  │
│  "Clear everything"                    │
└─────────────────────────────────────────┘
```

---

## 🚀 Ready to Use!

The application should now be running. Try these commands:

1. **Press L** → See what objects are detected
2. **Press SPACE** → Say "make the wall blue"
3. **Watch** → Wall turns blue with yellow outline
4. **Press SPACE** → Say "I'm overstimulated by the screen"
5. **Watch** → Screen dims automatically

Enjoy your sensory modulation system! 🎉

---

*For technical details, see `PIPELINE_DOCUMENTATION.md`*


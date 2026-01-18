# SocialSenseAR - Complete Tech Stack Architecture (with Overshoot AI)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    SOCIALSENSEAR TECH STACK - COMBINED AI APPROACH                           │
│         Real-time Voice-Controlled AR with Overshoot AI + Gemini Vision                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│              │
│   CAMERA     │──┐
│  (OpenCV)    │  │
│  VideoCapture│  │
└──────────────┘  │
                  │
┌──────────────┐  │     ┌──────────────────────────────────────────────────────────────┐
│              │  │     │                                                                │
│  MICROPHONE  │──┼────▶│                    PC (Python Runtime)                        │
│  (PyAudio)   │  │     │                    ┌──────────────────────────────┐          │
└──────────────┘  │     │                    │  AsyncCamera Class           │          │
                  │     │                    │  - Frame capture thread       │          │
                  │     │                    │  - Non-blocking frame buffer  │          │
                  │     │                    └──────────────────────────────┘          │
                  │     │                                                                │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                    AUDIO PROCESSING PATH                    │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                                                              │
                  │     │         Wake Word Detection                                  │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  "hey vibe" /      │                              │
                  │     │         │  "thanks"           │                              │
                  │     │         │  (SpeechRecognition)│                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  VoiceCommand      │                              │
                  │     │         │  Class             │                              │
                  │     │         │  - Record audio   │                              │
                  │     │         │  - Process command │                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  Command Text       │                              │
                  │     │         │  (String)          │                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Gemini Text API                     │    │
                  │     │         │         (google-generativeai)                │    │
                  │     │         │         ┌──────────────────────┐            │    │
                  │     │         │         │  GeminiAgent        │            │    │
                  │     │         │         │  - process_request() │            │    │
                  │     │         │         │  - Rate limiting     │            │    │
                  │     │         │         └──────────────────────┘            │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  Parsed Command    │                              │
                  │     │         │  - target_label   │                              │
                  │     │         │  - effect_type    │                              │
                  │     │         │  - parameters     │                              │
                  │     │         └────────────────────┘                              │
                  │     │                                                              │
                  │     └──────────────────────────────────────────────────────────────┘
                  │
                  │
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                    VISION PROCESSING PATH                   │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                                                              │
                  │     │         Frame Processing (OpenCV)                            │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  BGR Frame        │                              │
                  │     │         │  (640x480)         │                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ├──────────────────┐                       │
                  │     │                    │                  │                       │
                  │     │                    ▼                  ▼                       │
                  │     │     ┌──────────────────┐    ┌──────────────────┐          │
                  │     │     │                  │    │                  │          │
                  │     │     │   FastSAM        │    │   MediaPipe      │          │
                  │     │     │   (ultralytics)  │    │   (mediapipe)    │          │
                  │     │     │                  │    │                  │          │
                  │     │     │  ┌────────────┐  │    │  ┌────────────┐  │          │
                  │     │     │  │ FastSAM-s  │  │    │  │ Selfie     │  │          │
                  │     │     │  │ Model      │  │    │  │ Segmentation│ │          │
                  │     │     │  │ (.pt file) │  │    │  │            │  │          │
                  │     │     │  └────────────┘  │    │  └────────────┘  │          │
                  │     │     │                  │    │                  │          │
                  │     │     │  - Instance     │    │  - Person mask   │          │
                  │     │     │    segmentation │    │  - Face mesh      │          │
                  │     │     │  - 512px res    │    │  - Hands         │          │
                  │     │     │  - Retina masks │    │  - Pose          │          │
                  │     │     │                  │    │                  │          │
                  │     │     └──────────────────┘    └──────────────────┘          │
                  │     │                    │                  │                     │
                  │     │                    └──────────┬───────┘                     │
                  │     │                               ▼                              │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Mask Collection                      │    │
                  │     │         │         ┌──────────────────────┐            │    │
                  │     │         │         │  EnvironmentController│            │    │
                  │     │         │         │  - _update_masks()  │            │    │
                  │     │         │         │  - Combine SAM + MP  │            │    │
                  │     │         │         └──────────────────────┘            │    │
                  │     │         │                                              │    │
                  │     │         │  Output: List of (mask, label, center)      │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Mask Tracking                        │    │
                  │     │         │         ┌──────────────────────┐            │    │
                  │     │         │         │  _update_tracked_   │            │    │
                  │     │         │         │  masks()            │            │    │
                  │     │         │         │                      │            │    │
                  │     │         │         │  - Velocity tracking │            │    │
                  │     │         │         │  - Interpolation    │            │    │
                  │     │         │         │  - Persistence       │            │    │
                  │     │         │         └──────────────────────┘            │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │     ┌────────────────────────────────────────────────────┐    │
                  │     │     │                                                    │    │
                  │     │     │         COMBINED AI LABELING SYSTEM                │    │
                  │     │     │         (Overshoot AI + Gemini Vision)              │    │
                  │     │     │                                                    │    │
                  │     │     │                    │                                │    │
                  │     │     │                    ├──────────────┐                 │    │
                  │     │     │                    │              │                 │    │
                  │     │     │                    ▼              ▼                 │    │
                  │     │     │     ┌────────────────────┐  ┌────────────────────┐ │    │
                  │     │     │     │                    │  │                    │ │    │
                  │     │     │     │  Overshoot AI      │  │  Gemini Vision API │ │    │
                  │     │     │     │  (WebRTC Streaming)│  │  (google-genai)    │ │    │
                  │     │     │     │                    │  │                    │ │    │
                  │     │     │     │  ┌──────────────┐  │  │  ┌──────────────┐  │ │    │
                  │     │     │     │  │OvershootLabel│  │  │  │GeminiAgent  │  │ │    │
                  │     │     │     │  │er            │  │  │  │             │  │ │    │
                  │     │     │     │  │              │  │  │  │- label_all_ │  │ │    │
                  │     │     │     │  │- WebRTC      │  │  │  │  segments() │  │ │    │
                  │     │     │     │  │  streaming   │  │  │  │             │  │ │    │
                  │     │     │     │  │- Real-time   │  │  │  │- Rate limit │  │ │    │
                  │     │     │     │  │  inference   │  │  │  │  (3/min)    │  │ │    │
                  │     │     │     │  │- Fast labels │  │  │  │             │  │ │    │
                  │     │     │     │  └──────────────┘  │  │  └──────────────┘  │ │    │
                  │     │     │     │                    │  │                    │ │    │
                  │     │     │     │  Output:           │  │  Output:           │ │    │
                  │     │     │     │  {idx: "label"}   │  │  {idx: "label"}   │ │    │
                  │     │     │     │                    │  │                    │ │    │
                  │     │     │     └────────────────────┘  └────────────────────┘ │    │
                  │     │     │                    │              │               │    │
                  │     │     │                    └──────┬───────┘               │    │
                  │     │     │                           ▼                        │    │
                  │     │     │         ┌──────────────────────────────┐          │    │
                  │     │     │         │                              │          │    │
                  │     │     │         │  Label Merging Logic        │          │    │
                  │     │     │         │  ┌──────────────────────┐   │          │    │
                  │     │     │         │  │  Combined Labels   │   │          │    │
                  │     │     │         │  │                     │   │          │    │
                  │     │     │         │  │  1. Start with      │   │          │    │
                  │     │     │         │  │     Overshoot       │   │          │    │
                  │     │     │         │  │  2. Add Gemini for  │   │          │    │
                  │     │     │         │  │     missing indices │   │          │    │
                  │     │     │         │  │  3. Prefer more     │   │          │    │
                  │     │     │         │  │     specific labels │   │          │    │
                  │     │     │         │  └──────────────────────┘   │          │    │
                  │     │     │         │                              │          │    │
                  │     │     │         └──────────────────────────────┘          │    │
                  │     │     │                           │                        │    │
                  │     │     │                           ▼                        │    │
                  │     │     │         ┌──────────────────────────────┐          │    │
                  │     │     │         │                              │          │    │
                  │     │     │         │  Combined Label Dictionary   │          │    │
                  │     │     │         │  {idx: "best_label"}         │          │    │
                  │     │     │         │                              │          │    │
                  │     │     │         └──────────────────────────────┘          │    │
                  │     │     │                                                    │    │
                  │     │     └────────────────────────────────────────────────────┘    │
                  │     │                                                              │
                  │     └──────────────────────────────────────────────────────────────┘
                  │
                  │
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                    EFFECT APPLICATION                        │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                                                              │
                  │     │         Active Effects Dictionary                           │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  active_effects   │                              │
                  │     │         │  {                │                              │
                  │     │         │    "face": "red", │                              │
                  │     │         │    "screen": "mod_│                              │
                  │     │         │      {blur: true}"│                              │
                  │     │         │  }                │                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Effect Matching                      │    │
                  │     │         │         ┌──────────────────────┐            │    │
                  │     │         │         │  _match_effect()    │            │    │
                  │     │         │         │  - Exact match      │            │    │
                  │     │         │         │  - Base match      │            │    │
                  │     │         │         │  - Synonym groups  │            │    │
                  │     │         │         └──────────────────────┘            │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                    │                                          │
                  │     │                    ├──────────────────┐                       │
                  │     │                    │                  │                       │
                  │     │                    ▼                  ▼                       │
                  │     │     ┌──────────────────┐    ┌──────────────────┐          │
                  │     │     │                  │    │                  │          │
                  │     │     │   Color Mode     │    │  Modulation Mode │          │
                  │     │     │                  │    │                  │          │
                  │     │     │  - Color overlay │    │  - Blur          │          │
                  │     │     │  - Alpha blend   │    │  - Brightness    │          │
                  │     │     │  - Edge feather  │    │  - Contrast      │          │
                  │     │     │                  │    │  - Saturation    │          │
                  │     │     │                  │    │  - Motion dampen │          │
                  │     │     │                  │    │                  │          │
                  │     │     └──────────────────┘    └──────────────────┘          │
                  │     │                    │                  │                     │
                  │     │                    └──────────┬───────┘                     │
                  │     │                               ▼                              │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Modified Frame                       │    │
                  │     │         │         (OpenCV BGR)                        │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                                                              │
                  │     └──────────────────────────────────────────────────────────────┘
                  │
                  │
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                    DISPLAY & UI                            │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                                                              │
                  │     │         Display Frame (OpenCV)                               │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  cv2.imshow()     │                              │
                  │     │         │  - Borders        │                              │
                  │     │         │  - Labels         │                              │
                  │     │         │  - Effects        │                              │
                  │     │         │  - UI overlay     │                              │
                  │     │         │  - API status     │                              │
                  │     │         │    (Overshoot/Gemini)│                            │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  Display Window   │                              │
                  │     │         │  "Voice Environment│                              │
                  │     │         │   Controller"      │                              │
                  │     │         └────────────────────┘                              │
                  │     │                                                              │
                  │     └──────────────────────────────────────────────────────────────┘
                  │
                  │
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                    FEEDBACK LOOP                            │
                  │     └──────────────────────────────────────────────────────────────┘
                  │                                    │
                  │                                    ▼
                  │     ┌──────────────────────────────────────────────────────────────┐
                  │     │                                                              │
                  │     │         Comprehensive Feedback Loop                         │
                  │     │         ┌──────────────────────┐                          │
                  │     │         │  GeminiAgent         │                          │
                  │     │         │  - comprehensive_    │                          │
                  │     │         │    feedback_loop()   │                          │
                  │     │         │                      │                          │
                  │     │         │  Validates:          │                          │
                  │     │         │  - Detection accuracy│                          │
                  │     │         │  - Label correctness │                          │
                  │     │         │  - Threshold tuning  │                          │
                  │     │         │  - API performance   │                          │
                  │     │         └──────────────────────┘                          │
                  │     │                    │                                          │
                  │     │                    ▼                                          │
                  │     │         ┌────────────────────┐                              │
                  │     │         │  Label Corrections │                              │
                  │     │         │  Optimal Thresholds│                              │
                  │     │         │  API Usage Stats   │                              │
                  │     │         └────────────────────┘                              │
                  │     │                    │                                          │
                  │     │                    └──────────────┐                          │
                  │     │                                     │                          │
                  │     │                                     ▼                          │
                  │     │         ┌──────────────────────────────────────────────┐    │
                  │     │         │                                              │    │
                  │     │         │         Adaptive Thresholds                 │    │
                  │     │         │         ┌──────────────────────┐            │    │
                  │     │         │         │  - yolo_conf        │            │    │
                  │     │         │         │  - sam_conf         │            │    │
                  │     │         │         │  - matching_iou    │            │    │
                  │     │         │         │  - api_preference  │            │    │
                  │     │         │         └──────────────────────┘            │    │
                  │     │         │                                              │    │
                  │     │         └──────────────────────────────────────────────┘    │
                  │     │                                                              │
                  │     └──────────────────────────────────────────────────────────────┘
                  │
                  │
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                    KEY COMPONENTS                                         │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  HARDWARE:                                                                               │
│  • Camera (USB/Webcam) - OpenCV VideoCapture                                            │
│  • Microphone - PyAudio                                                                 │
│                                                                                           │
│  VISION MODELS:                                                                          │
│  • FastSAM-s (ultralytics) - Instance segmentation                                      │
│  • MediaPipe - Body part segmentation (face, hands, pose)                               │
│  • OpenCV - Image processing, drawing, effects                                          │
│                                                                                           │
│  AI SERVICES (COMBINED):                                                                 │
│  • Overshoot AI (WebRTC Streaming) - Primary labeling, real-time inference              │
│    - API: https://api.overshoot.ai/api/v0.2                                             │
│    - WebRTC streaming for low latency                                                    │
│    - Fast, continuous labeling                                                          │
│                                                                                           │
│  • Google Gemini 1.5 Flash - Vision API (validation & enhancement)                      │
│    - Validates Overshoot labels                                                          │
│    - Fills missing labels                                                               │
│    - Provides more specific labels when both agree                                      │
│    - Rate limited: 3 calls/minute (conservative)                                        │
│                                                                                           │
│  • Google Gemini 1.5 Flash - Text API (command parsing)                                 │
│    - Natural language command understanding                                              │
│    - Intent extraction                                                                   │
│                                                                                           │
│  AUDIO PROCESSING:                                                                       │
│  • SpeechRecognition - Wake word detection ("hey vibe", "thanks")                       │
│  • PyAudio - Audio input capture                                                        │
│                                                                                           │
│  DATA STRUCTURES:                                                                        │
│  • Mask Objects - (mask, label, center) tuples                                         │
│  • Active Effects Dictionary - {label: effect} mapping                                 │
│  • Tracked Masks - Persistent mask tracking with velocity                              │
│  • Scene Context - Continuous scene understanding                                       │
│  • Combined Labels - Merged results from Overshoot + Gemini                            │
│                                                                                           │
│  EFFECT TYPES:                                                                           │
│  • Color Remapping - Overlay colors on objects                                          │
│  • Blur - Gaussian blur for visual noise cancellation                                   │
│  • Brightness/Contrast - Reduce glare and overstimulation                               │
│  • Saturation - Reduce color intensity                                                  │
│  • Motion Dampening - Temporal smoothing for predictability                             │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                          COMBINED AI LABELING STRATEGY                                   │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  STEP 1: OVERSHOOT AI (Primary)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐               │
│  │ • WebRTC streaming connection                                        │               │
│  │ • Real-time frame analysis                                           │               │
│  │ • Fast labeling of all segments                                      │               │
│  │ • Output: {0: "wall", 1: "face", 2: "screen", ...}                  │               │
│  └─────────────────────────────────────────────────────────────────────┘               │
│                                                                                           │
│  STEP 2: GEMINI VISION (Enhancement)                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐               │
│  │ • Validates Overshoot labels                                         │               │
│  │ • Fills missing indices                                              │               │
│  │ • Provides more specific labels                                      │               │
│  │ • Rate limited to 3 calls/minute                                     │               │
│  │ • Output: {1: "person_face", 3: "laptop_screen", ...}              │               │
│  └─────────────────────────────────────────────────────────────────────┘               │
│                                                                                           │
│  STEP 3: LABEL MERGING                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐               │
│  │ • Start with Overshoot labels (fast, comprehensive)                  │               │
│  │ • Add Gemini labels for missing indices                              │               │
│  │ • When both label same object: prefer more specific                  │               │
│  │ • Example: Overshoot="object" + Gemini="laptop" → "laptop"          │               │
│  │ • Final: {0: "wall", 1: "person_face", 2: "laptop_screen", ...}     │               │
│  └─────────────────────────────────────────────────────────────────────┘               │
│                                                                                           │
│  STEP 4: SMART FALLBACK                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐               │
│  │ • Fills any remaining unlabeled segments                            │               │
│  │ • Uses position, size, brightness heuristics                        │               │
│  │ • Preserves MediaPipe labels (face, hands, etc.)                    │               │
│  └─────────────────────────────────────────────────────────────────────┘               │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA FLOW SUMMARY                                      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  1. INPUT: Camera frames + Microphone audio                                             │
│                                                                                           │
│  2. AUDIO PATH:                                                                          │
│     Mic → Wake Word Detection → Command Recording → Gemini Text API → Parsed Command   │
│                                                                                           │
│  3. VISION PATH:                                                                         │
│     Frame → FastSAM + MediaPipe → Mask Collection → Mask Tracking →                      │
│     ┌─────────────────────────────────────────────────────────────┐                   │
│     │ COMBINED AI LABELING:                                         │                   │
│     │ • Overshoot AI (WebRTC streaming) → Primary labels            │                   │
│     │ • Gemini Vision API → Validation & enhancement                │                   │
│     │ • Label Merging → Best combined labels                        │                   │
│     │ • Smart Fallback → Fill remaining                             │                   │
│     └─────────────────────────────────────────────────────────────┘                   │
│     → Labeled Masks                                                                      │
│                                                                                           │
│  4. EFFECT APPLICATION:                                                                  │
│     Active Effects → Effect Matching → Color/Modulation → Modified Frame                │
│                                                                                           │
│  5. DISPLAY:                                                                             │
│     Modified Frame → OpenCV Window → User View                                          │
│                                                                                           │
│  6. FEEDBACK:                                                                            │
│     Display → Gemini Feedback Loop → Label Corrections → Adaptive Thresholds           │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                              API USAGE & PERFORMANCE                                      │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                           │
│  OVERSHOOT AI:                                                                            │
│  • Connection: WebRTC streaming (persistent connection)                                  │
│  • Latency: ~100-200ms (real-time)                                                       │
│  • Rate: Continuous (no per-call limits)                                                │
│  • Cost: Based on streaming duration                                                     │
│  • Use Case: Primary labeling, fast response                                             │
│                                                                                           │
│  GEMINI VISION:                                                                          │
│  • Connection: HTTP REST API (per-request)                                               │
│  • Latency: ~500-2000ms (API call overhead)                                              │
│  • Rate: 3 calls/minute (conservative limit)                                             │
│  • Cost: Per API call                                                                    │
│  • Use Case: Validation, enhancement, filling gaps                                      │
│                                                                                           │
│  COMBINED BENEFITS:                                                                       │
│  • Faster overall labeling (Overshoot primary)                                            │
│  • Higher accuracy (Gemini validation)                                                   │
│  • Better coverage (both APIs complement each other)                                     │
│  • Cost optimization (Gemini only when needed)                                            │
│                                                                                           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack Details

### Core Libraries
- **Python 3.12+** - Runtime environment
- **OpenCV (cv2)** - Computer vision, image processing, display
- **NumPy** - Array operations, mask processing
- **PyTorch** - Deep learning framework (for FastSAM)

### Vision Models
- **FastSAM-s** (ultralytics) - Fast instance segmentation
- **MediaPipe** - Real-time body part segmentation
  - Selfie Segmentation (person mask)
  - Face Mesh (face detection)
  - Hands (hand tracking)
  - Pose (body part detection)

### AI Services (Combined Approach)
- **Overshoot AI** (Primary)
  - WebRTC streaming API
  - Real-time vision inference
  - Low latency labeling
  - API Key: `ovs_ddae62a151997de6e694fa4345611a0d`
  
- **Google Gemini 1.5 Flash** (Enhancement)
  - Vision API - Label validation and enhancement
  - Text API - Natural language command parsing
  - Rate limiting and feedback loop integration

### Audio Processing
- **SpeechRecognition** - Wake word detection ("hey vibe", "thanks")
- **PyAudio** - Audio input capture

### Data Processing
- **Threading** - Async camera capture, background processing
- **Queue** - Command queue management
- **JSON** - Effect parameter serialization
- **WebRTC (aiortc)** - Overshoot streaming connection
- **aiohttp/websockets** - Async HTTP/WebSocket for Overshoot

### Visual Effects
- **OpenCV Gaussian Blur** - Visual noise cancellation
- **NumPy array operations** - Color remapping, brightness/contrast
- **HSV color space** - Saturation control
- **Temporal smoothing** - Motion dampening

### Performance Optimizations
- Async camera capture (non-blocking)
- Frame skipping for detection
- Adaptive resolution (320-512px for SAM)
- Rate limiting for Gemini API calls
- WebRTC streaming for Overshoot (low latency)
- Mask tracking with velocity prediction
- Combined AI labeling (Overshoot + Gemini)


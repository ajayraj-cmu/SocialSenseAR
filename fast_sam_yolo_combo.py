#!/usr/bin/env python3
"""
FastSAM + YOLO Combo - Real-time Object Detection & Segmentation

Features:
- FastSAM for segmentation
- YOLO for object identification
- Dev/Non-Dev mode toggle (press 'D')
- Real-time visualization

Controls:
- D: Toggle Dev/Non-Dev mode
- Q: Quit
- S: Save screenshot
"""

import cv2
import numpy as np
import time
from ultralytics import FastSAM, YOLO

class FastSAMYOLOCombo:
    def __init__(self):
        print("\n" + "="*60)
        print("  FastSAM + YOLO Combo")
        print("="*60)
        print("\nLoading models...")
        
        # Load FastSAM
        self.sam = FastSAM("FastSAM-s.pt")
        print("  ✓ FastSAM loaded")
        
        # Load YOLO
        self.yolo = YOLO("yolov8n.pt")
        print("  ✓ YOLO loaded")
        
        print("✅ All models ready!\n")
        
        # Dev mode toggle
        self.dev_mode = True  # Start in dev mode
        
        # FPS tracking
        self.fps_times = []
        
    def process_frame(self, frame):
        """Process frame with FastSAM and YOLO."""
        h, w = frame.shape[:2]
        
        # Run YOLO for object detection
        yolo_results = self.yolo(frame, verbose=False, conf=0.3)
        yolo_detections = []
        
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                label = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                yolo_detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'label': label,
                    'confidence': conf
                })
        
        # Run FastSAM for segmentation
        sam_results = self.sam(
            frame,
            device="cpu",
            retina_masks=True,
            imgsz=320,
            conf=0.4,
            iou=0.9,
            verbose=False
        )
        
        masks = []
        if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
            masks_data = sam_results[0].masks.data.cpu().numpy()
            for mask_data in masks_data:
                mask_resized = cv2.resize(mask_data.astype(np.float32), (w, h))
                masks.append(mask_resized)
        
        return yolo_detections, masks
    
    def match_masks_to_detections(self, masks, yolo_detections, h, w):
        """Match FastSAM masks to YOLO detections."""
        matched = []
        
        for i, mask in enumerate(masks):
            # Get mask bbox
            mask_bool = mask > 0.5
            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                continue
            
            mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            
            # Find best matching YOLO detection
            best_match = None
            best_iou = 0
            
            for det in yolo_detections:
                iou = self._calc_iou(mask_bbox, det['bbox'])
                if iou > best_iou and iou > 0.2:
                    best_iou = iou
                    best_match = det
            
            matched.append({
                'mask': mask,
                'bbox': mask_bbox,
                'label': best_match['label'] if best_match else 'object',
                'confidence': best_match['confidence'] if best_match else 0.5,
                'yolo_match': best_match is not None
            })
        
        return matched
    
    def _calc_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def draw_dev_mode(self, frame, matched_objects, yolo_detections):
        """Draw dev mode visualization with all debug info."""
        display = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw all masks with colors
        np.random.seed(42)
        for i, obj in enumerate(matched_objects):
            mask = obj['mask']
            mask_bool = mask > 0.5
            
            # Generate color
            hue = int((i * 137.5) % 180)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = tuple(map(int, color))
            
            # Draw mask overlay
            overlay = display.copy()
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            display = overlay.astype(np.uint8)
            
            # Draw contour
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color_tuple, 2)
            
            # Draw label at center
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    label = f"{i+1}: {obj['label']}"
                    if obj['yolo_match']:
                        label += f" ({obj['confidence']:.2f})"
                    
                    # Background for text
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display, (cx - 5, cy - th - 5), (cx + tw + 5, cy + 5), (0, 0, 0), -1)
                    cv2.putText(display, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw YOLO bounding boxes
        for det in yolo_detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display
    
    def draw_non_dev_mode(self, frame, matched_objects):
        """Draw clean non-dev mode visualization."""
        display = frame.copy()
        
        # Only draw subtle outlines
        for obj in matched_objects:
            mask = obj['mask']
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Subtle white outline
            cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
        
        return display
    
    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("✅ Camera ready!")
        print("\nControls:")
        print("  D - Toggle Dev/Non-Dev mode")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("="*60 + "\n")
        
        cv2.namedWindow("FastSAM + YOLO Combo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FastSAM + YOLO Combo", 1280, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                start_time = time.time()
                
                # Process frame
                yolo_detections, masks = self.process_frame(frame)
                
                # Match masks to detections
                h, w = frame.shape[:2]
                matched_objects = self.match_masks_to_detections(masks, yolo_detections, h, w)
                
                # Draw based on mode
                if self.dev_mode:
                    display = self.draw_dev_mode(frame, matched_objects, yolo_detections)
                else:
                    display = self.draw_non_dev_mode(frame, matched_objects)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                self.fps_times.append(elapsed)
                if len(self.fps_times) > 30:
                    self.fps_times.pop(0)
                avg_time = sum(self.fps_times) / len(self.fps_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Info overlay
                h, w = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                
                mode_text = "DEV MODE" if self.dev_mode else "NON-DEV MODE"
                mode_color = (0, 255, 255) if self.dev_mode else (0, 255, 0)
                
                cv2.putText(display, f"FastSAM + YOLO Combo", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"Mode: {mode_text}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
                cv2.putText(display, f"Objects: {len(matched_objects)} | YOLO: {len(yolo_detections)}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, f"FPS: {fps:.1f} ({elapsed*1000:.0f}ms)", (20, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("FastSAM + YOLO Combo", display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('d') or key == ord('D'):
                    self.dev_mode = not self.dev_mode
                    mode = "DEV" if self.dev_mode else "NON-DEV"
                    print(f"🔄 Switched to {mode} MODE")
                elif key == ord('s'):
                    filename = f"fastsam_yolo_{int(time.time())}.png"
                    cv2.imwrite(filename, display)
                    print(f"💾 Saved: {filename}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n👋 Done!")


def main():
    app = FastSAMYOLOCombo()
    app.run()


if __name__ == "__main__":
    main()


"""
Overshoot Vision Client for Python
Real-time AI vision analysis using Overshoot's hosted API
"""
import asyncio
import json
import time
import threading
import queue
import base64
import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, List
import os
from fractions import Fraction

try:
    import aiohttp
    import websockets
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("⚠️ aiohttp/websockets not installed. Run: pip install aiohttp websockets")

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaStreamTrack
    from av import VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("⚠️ aiortc not available. Using HTTP fallback for Overshoot.")


class FrameVideoTrack(VideoStreamTrack if WEBRTC_AVAILABLE else object):
    """Custom video track that sends frames from a queue."""
    
    def __init__(self):
        if WEBRTC_AVAILABLE:
            super().__init__()
        self.frame_queue = queue.Queue(maxsize=5)
        self._timestamp = 0
    
    def put_frame(self, frame: np.ndarray):
        """Add a frame to the queue (BGR format from OpenCV)."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except:
                pass
    
    async def recv(self):
        """Get the next frame for WebRTC."""
        if not WEBRTC_AVAILABLE:
            return None
            
        # Wait for a frame
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                break
            except queue.Empty:
                await asyncio.sleep(0.01)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = self._timestamp
        video_frame.time_base = Fraction(1, 30)
        self._timestamp += 1
        
        return video_frame


class OvershootVisionClient:
    """
    Python client for Overshoot real-time vision API.
    
    Handles video streaming and inference results without blocking
    the main application thread.
    """
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.overshoot.ai/api/v0.2",
        prompt: str = "Describe what you see",
        backend: str = "overshoot",
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        on_result: Optional[Callable[[Dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        debug: bool = False
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip('/')
        self.prompt = prompt
        self.backend = backend
        self.model = model
        self.on_result = on_result
        self.on_error = on_error
        self.debug = debug
        
        # State
        self.stream_id: Optional[str] = None
        self.is_running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._frame_track: Optional[FrameVideoTrack] = None
        self._pc: Optional[RTCPeerConnection] = None
        self._ws: Optional[Any] = None
        
        # Results queue for main thread to consume
        self.results_queue = queue.Queue(maxsize=100)
        self.last_result: Optional[Dict] = None
        self.last_result_time = 0
        
        # Rate limiting
        self.min_result_interval = 0.5  # Minimum time between processing results
        
    def _log(self, *args):
        if self.debug:
            print("[Overshoot]", *args)
    
    def _log_error(self, *args):
        print("[Overshoot ERROR]", *args)
    
    async def _create_stream_http(self, frame: np.ndarray) -> Optional[str]:
        """Create stream using HTTP (fallback when WebRTC unavailable)."""
        if not ASYNC_AVAILABLE:
            return None
            
        try:
            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # This is a simplified approach - actual API may differ
            # The real Overshoot API uses WebRTC for streaming
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/streams",
                    headers=headers,
                    json={
                        "inference": {
                            "prompt": self.prompt,
                            "backend": self.backend,
                            "model": self.model
                        },
                        "processing": {
                            "sampling_ratio": 0.5,
                            "fps": 10,
                            "clip_length_seconds": 1.0,
                            "delay_seconds": 1.0
                        }
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._log(f"Stream created: {data.get('stream_id')}")
                        return data.get('stream_id')
                    else:
                        error_text = await resp.text()
                        self._log_error(f"Failed to create stream: {resp.status} - {error_text}")
                        return None
        except Exception as e:
            self._log_error(f"HTTP stream creation error: {e}")
            return None
    
    async def _connect_websocket(self, stream_id: str):
        """Connect to WebSocket for results."""
        if not ASYNC_AVAILABLE:
            return
            
        ws_url = self.api_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/ws/streams/{stream_id}"
        
        try:
            async with websockets.connect(ws_url) as ws:
                self._ws = ws
                
                # Authenticate
                await ws.send(json.dumps({"api_key": self.api_key}))
                self._log("WebSocket connected and authenticated")
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        result = json.loads(message)
                        
                        self.last_result = result
                        self.last_result_time = time.time()
                        
                        # Add to queue
                        try:
                            self.results_queue.put_nowait(result)
                        except queue.Full:
                            self.results_queue.get_nowait()
                            self.results_queue.put_nowait(result)
                        
                        # Callback
                        if self.on_result:
                            self.on_result(result)
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._log_error(f"WebSocket receive error: {e}")
                        break
                        
        except Exception as e:
            self._log_error(f"WebSocket connection error: {e}")
            if self.on_error:
                self.on_error(e)
    
    def start(self, initial_frame: Optional[np.ndarray] = None):
        """Start the Overshoot vision client in a background thread."""
        if self.is_running:
            self._log("Already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        self._log("Started background thread - WebRTC streaming will begin")
        
        # Wait a moment for connection to establish
        time.sleep(2)
    
    def _run_async_loop(self):
        """Run the async event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            self._log_error(f"Async loop error: {e}")
        finally:
            self._loop.close()
    
    async def _main_loop(self):
        """Main async loop for handling WebRTC streaming."""
        self._log("Async loop started - setting up WebRTC")
        
        if not WEBRTC_AVAILABLE:
            self._log_error("WebRTC not available - cannot stream")
            return
        
        try:
            # Create video track
            self._frame_track = FrameVideoTrack()
            
            # Create peer connection
            self._pc = RTCPeerConnection()
            self._pc.addTrack(self._frame_track)
            
            # Create offer
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            
            # Wait for ICE gathering
            await asyncio.sleep(1)
            
            if not self._pc.localDescription:
                self._log_error("Failed to create local description")
                return
            
            # Create stream on server
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "webrtc": {
                    "type": "offer",
                    "sdp": self._pc.localDescription.sdp
                },
                "processing": {
                    "sampling_ratio": 0.1,
                    "fps": 10,
                    "clip_length_seconds": 1.0,
                    "delay_seconds": 1.0
                },
                "inference": {
                    "prompt": self.prompt,
                    "backend": self.backend,
                    "model": self.model
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/streams",
                    headers=headers,
                    json=request_data
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.stream_id = data.get("stream_id")
                        self._log(f"Stream created: {self.stream_id}")
                        
                        # Set remote description
                        answer = RTCSessionDescription(
                            sdp=data["webrtc"]["sdp"],
                            type=data["webrtc"]["type"]
                        )
                        await self._pc.setRemoteDescription(answer)
                        
                        # Connect WebSocket for results in background
                        asyncio.create_task(self._connect_websocket(self.stream_id))
                    else:
                        error_text = await resp.text()
                        self._log_error(f"Failed to create stream: {resp.status} - {error_text[:200]}")
                        return
            
            # Keep connection alive
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self._log_error(f"WebRTC setup error: {e}")
            import traceback
            self._log_error(traceback.format_exc())
    
    def send_frame(self, frame: np.ndarray):
        """Send a frame for analysis (queued for async processing)."""
        if self._frame_track:
            self._frame_track.put_frame(frame)
    
    def get_result(self) -> Optional[Dict]:
        """Get the latest result (non-blocking)."""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get the most recent result without removing from queue."""
        return self.last_result
    
    async def analyze_frame_once(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Analyze a single frame using Overshoot API.
        This is a simpler HTTP-based approach for occasional analysis.
        """
        if not ASYNC_AVAILABLE:
            self._log_error("aiohttp not available")
            return None
        
        try:
            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Note: This assumes Overshoot has a single-image endpoint
            # The actual API primarily uses WebRTC streaming
            # You may need to adjust based on actual Overshoot API capabilities
            
            async with aiohttp.ClientSession() as session:
                # Try the streaming API with a single frame approach
                async with session.post(
                    f"{self.api_url}/analyze",  # Hypothetical endpoint
                    headers=headers,
                    json={
                        "image": frame_b64,
                        "prompt": self.prompt,
                        "backend": self.backend,
                        "model": self.model
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 404:
                        # Endpoint doesn't exist - Overshoot is streaming-only
                        self._log("Single-frame endpoint not available (streaming-only API)")
                        return None
                    else:
                        error_text = await resp.text()
                        self._log_error(f"Analysis failed: {resp.status} - {error_text[:100]}")
                        return None
                        
        except Exception as e:
            self._log_error(f"Frame analysis error: {e}")
            return None
    
    def analyze_frame_sync(self, frame: np.ndarray) -> Optional[Dict]:
        """Synchronous wrapper for analyze_frame_once."""
        if not ASYNC_AVAILABLE:
            return None
            
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self.analyze_frame_once(frame))
            loop.close()
            return result
        except Exception as e:
            self._log_error(f"Sync analysis error: {e}")
            return None
    
    async def update_prompt(self, new_prompt: str):
        """Update the analysis prompt."""
        self.prompt = new_prompt
        
        if self.stream_id and ASYNC_AVAILABLE:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.patch(
                        f"{self.api_url}/streams/{self.stream_id}/config/prompt",
                        headers=headers,
                        json={"prompt": new_prompt}
                    ) as resp:
                        if resp.status == 200:
                            self._log(f"Prompt updated")
                        else:
                            self._log_error(f"Failed to update prompt: {resp.status}")
            except Exception as e:
                self._log_error(f"Prompt update error: {e}")
    
    def stop(self):
        """Stop the client."""
        self.is_running = False
        
        if self._ws:
            try:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            except:
                pass
        
        if self._pc:
            try:
                asyncio.run_coroutine_threadsafe(self._pc.close(), self._loop)
            except:
                pass
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        self._log("Stopped")


class OvershootLabeler:
    """
    High-level interface for using Overshoot to label SAM segments.
    
    Integrates with the existing GeminiAgent pattern but uses Overshoot API.
    """
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        self.api_key = api_key or os.environ.get("OVERSHOOT_API_KEY", "")
        self.debug = debug
        self.available = bool(self.api_key) and ASYNC_AVAILABLE
        
        self.client: Optional[OvershootVisionClient] = None
        self._last_label_time = 0
        self._label_cooldown = 2.0  # Seconds between API calls
        self._cached_labels: Dict[str, tuple] = {}  # spatial_key -> (label, timestamp)
        
        if self.available:
            print("✅ Overshoot client initialized")
        else:
            if not self.api_key:
                print("⚠️ No OVERSHOOT_API_KEY found")
            if not ASYNC_AVAILABLE:
                print("⚠️ Missing dependencies: pip install aiohttp websockets")
    
    def _log(self, *args):
        if self.debug:
            print("[OvershootLabeler]", *args)
    
    def label_segments(
        self,
        frame: np.ndarray,
        masks_with_centers: List[tuple]
    ) -> Optional[Dict[int, str]]:
        """
        Label segments using Overshoot Vision API (streaming mode).
        
        Args:
            frame: BGR image from camera
            masks_with_centers: List of (mask, existing_label, center) tuples
            
        Returns:
            Dict mapping mask index -> label string, or None if unavailable
        """
        if not self.available:
            return None
        
        try:
            # Build position descriptions for prompt
            h, w = frame.shape[:2]
            positions = []
            for idx, (mask, existing_label, center) in enumerate(masks_with_centers):
                if center and center[0] > 0:
                    cx, cy = center
                    h_pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                    v_pos = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"
                    positions.append(f"{idx+1}: {v_pos}-{h_pos}")
            
            prompt = f"""Look at this image. Name each numbered region with ONE simple label.

Regions:
{chr(10).join(positions[:15])}

LABELS TO USE:
- Body: face, hand, arm, body, head, person
- Structure: wall, ceiling, floor, door, window
- Furniture: chair, desk, table, shelf
- Objects: light, lamp, monitor, screen, plant

Return ONLY JSON array: [{{"region":1,"label":"wall"}},{{"region":2,"label":"face"}}]"""

            # Create and start client if needed
            if not self.client:
                self.client = OvershootVisionClient(
                    api_key=self.api_key,
                    prompt=prompt,
                    debug=self.debug,
                    on_result=lambda r: None  # Results come via queue
                )
                self.client.start()
                # Wait for connection
                time.sleep(3)
            else:
                # Update prompt if changed
                if self.client.prompt != prompt:
                    asyncio.run_coroutine_threadsafe(
                        self.client.update_prompt(prompt),
                        self.client._loop
                    )
            
            # Send frame to stream
            if self.client._frame_track:
                self.client.send_frame(frame)
            
            # Try to get result from queue (non-blocking)
            result = self.client.get_result()
            
            if not result:
                # Check latest result
                result = self.client.get_latest_result()
            
            if not result:
                return None
            
            # Parse result
            result_text = result.get("result", "")
            if not result_text:
                return None
            
            # Extract JSON
            if "[" in result_text and "]" in result_text:
                start = result_text.find("[")
                end = result_text.rfind("]") + 1
                json_str = result_text[start:end]
                
                items = json.loads(json_str)
                labels = {}
                for item in items:
                    region = item.get("region", 0) - 1
                    label = item.get("label", "").lower().strip()
                    if 0 <= region < len(masks_with_centers) and label:
                        labels[region] = label
                
                if labels:
                    self._log(f"Labeled {len(labels)} segments: {list(labels.values())[:5]}")
                    return labels
            
            return None
            
        except Exception as e:
            self._log(f"Labeling error: {e}")
            import traceback
            if self.debug:
                self._log(traceback.format_exc())
            return None
    
    def close(self):
        """Clean up resources."""
        if self.client:
            self.client.stop()


# Convenience function to check if Overshoot is available
def is_overshoot_available() -> bool:
    """Check if Overshoot client can be used."""
    api_key = os.environ.get("OVERSHOOT_API_KEY", "")
    if not api_key or api_key == "your-overshoot-api-key-here":
        return False
    return ASYNC_AVAILABLE and WEBRTC_AVAILABLE


if __name__ == "__main__":
    # Test the client
    print("Testing Overshoot client...")
    
    api_key = os.environ.get("OVERSHOOT_API_KEY", "")
    if not api_key:
        print("Set OVERSHOOT_API_KEY environment variable to test")
        exit(1)
    
    labeler = OvershootLabeler(api_key=api_key, debug=True)
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.putText(test_frame, "TEST", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Test labeling
    result = labeler.label_segments(test_frame, [
        (np.ones((480, 640)), "test", (200, 200))
    ])
    
    print(f"Result: {result}")
    labeler.close()


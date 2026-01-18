"""
Audio Capture and Processing.

Handles:
- Audio stream capture
- Buffering for synchronization
- Basic audio analysis
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Callable, List
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from loguru import logger

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not available, audio capture disabled")


@dataclass
class AudioBuffer:
    """Circular buffer for audio samples."""
    data: NDArray[np.float32]
    sample_rate: int
    channels: int
    write_index: int = 0
    read_index: int = 0
    
    @property
    def available_samples(self) -> int:
        """Number of samples available to read."""
        if self.write_index >= self.read_index:
            return self.write_index - self.read_index
        return len(self.data) - self.read_index + self.write_index


class AudioProcessor:
    """
    Audio capture and basic processing.
    
    Guarantees:
    - Synchronized audio capture with video
    - Low-latency buffering
    - Thread-safe operations
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,  # Use mono - more compatible
        chunk_size: int = 1024,
        buffer_seconds: float = 0.5,
        device_index: Optional[int] = None,
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            chunk_size: Samples per processing chunk
            buffer_seconds: Buffer duration in seconds
            device_index: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.device_index = device_index
        
        # Buffer
        buffer_size = int(sample_rate * buffer_seconds)
        self._buffer = AudioBuffer(
            data=np.zeros((buffer_size, channels), dtype=np.float32),
            sample_rate=sample_rate,
            channels=channels,
        )
        
        # State
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_audio_callback: Optional[Callable[[NDArray[np.float32]], None]] = None
        
        # Performance tracking
        self._capture_latency_ms: float = 0.0
        self._dropped_frames: int = 0
    
    def start(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            True if started successfully
        """
        if sd is None:
            logger.error("sounddevice not available")
            return False
        
        if self._is_running:
            return True
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device_index,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            logger.info(f"Audio capture started: {self.sample_rate}Hz, {self.channels}ch")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self):
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False
        logger.info("Audio capture stopped")
    
    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info,
        status,
    ):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        with self._lock:
            # Write to circular buffer
            buffer_size = len(self._buffer.data)
            
            for i in range(frames):
                idx = (self._buffer.write_index + i) % buffer_size
                self._buffer.data[idx] = indata[i]
            
            self._buffer.write_index = (self._buffer.write_index + frames) % buffer_size
        
        # Notify callback if set
        if self._on_audio_callback is not None:
            self._on_audio_callback(indata.copy())
    
    def get_latest_chunk(
        self,
        num_samples: Optional[int] = None,
    ) -> Optional[NDArray[np.float32]]:
        """
        Get latest audio samples.
        
        Args:
            num_samples: Number of samples to retrieve (default: chunk_size)
            
        Returns:
            Audio samples or None if not enough data
        """
        if num_samples is None:
            num_samples = self.chunk_size
        
        with self._lock:
            if self._buffer.available_samples < num_samples:
                return None
            
            buffer_size = len(self._buffer.data)
            result = np.zeros((num_samples, self.channels), dtype=np.float32)
            
            for i in range(num_samples):
                idx = (self._buffer.read_index + i) % buffer_size
                result[i] = self._buffer.data[idx]
            
            self._buffer.read_index = (self._buffer.read_index + num_samples) % buffer_size
            
            return result
    
    def peek_latest_chunk(
        self,
        num_samples: Optional[int] = None,
    ) -> Optional[NDArray[np.float32]]:
        """
        Peek at latest audio samples without consuming them.
        """
        if num_samples is None:
            num_samples = self.chunk_size
        
        with self._lock:
            if self._buffer.available_samples < num_samples:
                return None
            
            buffer_size = len(self._buffer.data)
            result = np.zeros((num_samples, self.channels), dtype=np.float32)
            
            # Read from most recent data
            start_idx = (self._buffer.write_index - num_samples) % buffer_size
            
            for i in range(num_samples):
                idx = (start_idx + i) % buffer_size
                result[i] = self._buffer.data[idx]
            
            return result
    
    def set_callback(self, callback: Callable[[NDArray[np.float32]], None]):
        """Set callback for real-time audio processing."""
        self._on_audio_callback = callback
    
    def get_rms_level(self, audio: NDArray[np.float32]) -> float:
        """Calculate RMS level of audio chunk."""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def get_peak_level(self, audio: NDArray[np.float32]) -> float:
        """Get peak level of audio chunk."""
        return float(np.max(np.abs(audio)))
    
    def analyze_frequency_content(
        self,
        audio: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Analyze frequency content of audio.
        
        Returns:
            Power spectrum
        """
        # Mix to mono for analysis
        if audio.ndim > 1:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio
        
        # FFT
        spectrum = np.abs(np.fft.rfft(mono))
        
        return spectrum.astype(np.float32)
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def latency_ms(self) -> float:
        """Estimated capture latency in milliseconds."""
        if self._stream is not None:
            return float(self._stream.latency * 1000)
        return 0.0



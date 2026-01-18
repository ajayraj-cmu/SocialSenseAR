"""
Voice Command Handler.

Listens for voice commands and sends them to the pipeline.
Uses Google Speech Recognition for simplicity.
"""

from __future__ import annotations

import threading
import queue
import time
from typing import Optional, Callable
from loguru import logger

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    logger.warning("speech_recognition not available")


class VoiceCommander:
    """
    Voice command handler using speech recognition.
    
    Listens continuously for voice commands and sends them
    to a callback function.
    """
    
    def __init__(
        self,
        on_command: Optional[Callable[[str], None]] = None,
        wake_word: Optional[str] = None,
        energy_threshold: int = 100,  # Lower = more sensitive
        pause_threshold: float = 0.5,  # Faster response
    ):
        """
        Initialize voice commander.
        
        Args:
            on_command: Callback function when command is recognized
            wake_word: Optional wake word to trigger listening (e.g., "hey assistant")
            energy_threshold: Minimum audio energy to consider for recording
            pause_threshold: Seconds of silence before phrase is considered complete
        """
        self.on_command = on_command
        self.wake_word = wake_word.lower() if wake_word else None
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        
        self._recognizer: Optional[sr.Recognizer] = None
        self._microphone: Optional[sr.Microphone] = None
        self._is_running = False
        self._listen_thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue()
        
        # State
        self._is_listening = False
        self._last_command_time = 0.0
        self._cooldown_seconds = 1.0  # Prevent duplicate commands
        
    def start(self) -> bool:
        """
        Start voice command listening.
        
        Returns:
            True if started successfully
        """
        if not SPEECH_AVAILABLE:
            logger.error("speech_recognition module not available")
            return False
        
        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self.energy_threshold
            self._recognizer.pause_threshold = self.pause_threshold
            self._recognizer.dynamic_energy_threshold = True
            
            self._microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            logger.info("Calibrating microphone for ambient noise...")
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info(f"Microphone calibrated. Energy threshold: {self._recognizer.energy_threshold}")
            
            # Start listening thread
            self._is_running = True
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
            
            logger.info("Voice commander started - speak commands!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice commander: {e}")
            return False
    
    def stop(self):
        """Stop voice command listening."""
        self._is_running = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        logger.info("Voice commander stopped")
    
    def _listen_loop(self):
        """Main listening loop (runs in background thread)."""
        while self._is_running:
            try:
                with self._microphone as source:
                    self._is_listening = True
                    
                    # Listen for audio
                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=5.0,
                            phrase_time_limit=10.0
                        )
                    except sr.WaitTimeoutError:
                        continue
                    
                    self._is_listening = False
                    
                    # Recognize speech
                    logger.debug("Audio captured, sending to Google...")
                    try:
                        text = self._recognizer.recognize_google(audio)
                        logger.info(f"🎤 RECOGNIZED: \"{text}\"")
                        self._process_recognition(text)
                    except sr.UnknownValueError:
                        # Speech was unintelligible
                        logger.debug("Speech unintelligible")
                    except sr.RequestError as e:
                        logger.error(f"Speech recognition service error: {e}")
                        
            except Exception as e:
                logger.error(f"Voice commander error: {e}")
                time.sleep(1.0)
    
    def _process_recognition(self, text: str):
        """Process recognized text."""
        text = text.lower().strip()
        
        if not text:
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - self._last_command_time < self._cooldown_seconds:
            return
        
        # Check wake word if configured
        if self.wake_word:
            if not text.startswith(self.wake_word):
                return
            # Remove wake word from command
            text = text[len(self.wake_word):].strip()
        
        if not text:
            return
        
        logger.info(f"Voice command: \"{text}\"")
        self._last_command_time = current_time
        
        # Send to callback
        if self.on_command:
            self.on_command(text)
        
        # Also queue it
        self._command_queue.put(text)
    
    def get_command(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get a command from the queue (non-blocking).
        
        Returns:
            Command string or None
        """
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening for speech."""
        return self._is_listening
    
    @property
    def is_running(self) -> bool:
        """Check if voice commander is running."""
        return self._is_running


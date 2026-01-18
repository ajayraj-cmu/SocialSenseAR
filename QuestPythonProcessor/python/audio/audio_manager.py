"""
Audio Manager - Orchestrates all audio services.

Manages starting/stopping audio services alongside the video pipeline.
"""
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseAudioService


class AudioManager:
    """Manages audio services that run parallel to the video pipeline."""

    def __init__(self, config, services: Optional[List[str]] = None):
        """Initialize audio manager.

        Args:
            config: Configuration object with audio settings
            services: List of service names to enable, or None for defaults
        """
        self.config = config
        self.services: Dict[str, BaseAudioService] = {}
        self.running = False

        # Get state directory from config
        state_dir = getattr(config, 'audio_state_dir', '~/Downloads/Nex/conve_context')
        self.state_dir = Path(state_dir).expanduser()

        # Initialize requested services
        requested = services or getattr(config, 'audio_services', ['context'])
        self._init_services(requested)

    def _init_services(self, service_names: List[str]) -> None:
        """Initialize requested audio services.

        Args:
            service_names: List of service names to initialize
        """
        from . import AUDIO_SERVICES

        for name in service_names:
            if name in AUDIO_SERVICES:
                try:
                    service = AUDIO_SERVICES[name](self.config, self.state_dir)
                    self.services[name] = service
                    print(f"[AUDIO] Initialized {name} service")
                except Exception as e:
                    print(f"[AUDIO] Failed to initialize {name} service: {e}")
            else:
                print(f"[AUDIO] Unknown service: {name}")

    def start(self) -> None:
        """Start all audio services."""
        if self.running:
            return

        self.running = True

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        for name, service in self.services.items():
            try:
                if service.start():
                    print(f"[AUDIO] Started {name} service")
                else:
                    print(f"[AUDIO] Failed to start {name} service")
            except Exception as e:
                print(f"[AUDIO] Error starting {name} service: {e}")

    def stop(self) -> None:
        """Stop all audio services."""
        if not self.running:
            return

        self.running = False

        for name, service in self.services.items():
            try:
                service.stop()
                print(f"[AUDIO] Stopped {name} service")
            except Exception as e:
                print(f"[AUDIO] Error stopping {name} service: {e}")

    def get_state(self, service_name: str = "context") -> dict:
        """Get current state from a service.

        Args:
            service_name: Name of the service to query

        Returns:
            State dictionary, or empty dict if service not found
        """
        if service_name in self.services:
            return self.services[service_name].get_state()
        return {}

    def list_microphones(self) -> None:
        """Print available microphones for configuration."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()

            print("\n" + "=" * 60)
            print("AVAILABLE MICROPHONES")
            print("=" * 60)

            default_index = p.get_default_input_device_info()['index']

            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    default = " (DEFAULT)" if i == default_index else ""
                    print(f"  [{i}] {info['name']}{default}")

            print("=" * 60)
            print("Use --mic1 and --mic2 to specify microphone indices")
            print()

            p.terminate()
        except Exception as e:
            print(f"[AUDIO] Could not list microphones: {e}")

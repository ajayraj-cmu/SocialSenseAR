"""
Audio processing module for QuestPythonProcessor.

Provides audio services that run in parallel with the video pipeline,
including conversation transcription, summarization, and context tracking.
"""
from .base import BaseAudioService
from .audio_manager import AudioManager
from .context_service import ContextService
from .voice_isolation_service import VoiceIsolationService

# Registry of available audio services
AUDIO_SERVICES = {
    "context": ContextService,
    "voice_isolation": VoiceIsolationService,
}

__all__ = [
    'BaseAudioService',
    'AudioManager',
    'ContextService',
    'VoiceIsolationService',
    'AUDIO_SERVICES',
]

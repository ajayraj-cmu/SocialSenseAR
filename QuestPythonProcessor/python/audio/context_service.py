"""
Context Service - Dual-mic audio capture, VAD, transcription, and summarization.

Adapted from Audio_Merge/context_window.py for integration with QuestPythonProcessor.
Captures audio from two microphones, transcribes speech using OpenAI Whisper,
and generates conversation summaries using GPT-4o-mini.
"""
import pyaudio
import webrtcvad
import wave
import tempfile
import os
import time
import json
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import OpenAI

from .base import BaseAudioService


class ContextService(BaseAudioService):
    """Captures dual-mic audio, transcribes, and generates conversation summaries."""

    # Audio settings
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

    # VAD settings
    VAD_AGGRESSIVENESS = 3
    SILENCE_THRESHOLD = 20
    MIN_SPEECH_FRAMES = 8

    # Processing settings
    MAX_UTTERANCES_BEFORE_UPDATE = 3
    MAX_WAIT_TIME = 3.0
    MIN_WORDS_FOR_UPDATE = 15

    def __init__(self, config, state_dir: Path):
        """Initialize context service.

        Args:
            config: Configuration object with audio settings
            state_dir: Directory for state output files
        """
        super().__init__(config, state_dir)

        # OpenAI client - uses OPENAI_API_KEY from environment
        self.client = OpenAI()

        # PyAudio instance
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.streams: List = []

        # Audio queues for async processing
        self.audio_queue_user: Optional[Queue] = None
        self.audio_queue_other: Optional[Queue] = None

        # Processing threads
        self.threads: List[threading.Thread] = []

        # Context directories
        self.user_dir = state_dir / "user_conv_context"
        self.other_dir = state_dir / "person_talking_to_context"

        # Single mic mode flag
        self.single_mic_mode = False

        # Current state
        self.current_state: Dict[str, Any] = {
            "convo_state_summary": "--",
            "question": "",
            "recent_utterance": "--",
            "emotion": "Calm",
            "timestamp": ""
        }

    def start(self) -> bool:
        """Start audio capture with configured microphones.

        Returns:
            True if started successfully
        """
        try:
            # Create directories
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.user_dir.mkdir(exist_ok=True)
            self.other_dir.mkdir(exist_ok=True)

            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()

            # Get microphone indices
            mic1_index = getattr(self.config, 'audio_mic1_index', None)
            mic2_index = getattr(self.config, 'audio_mic2_index', None)

            # Auto-detect if not specified
            if mic1_index is None:
                mic1_index = self.pyaudio.get_default_input_device_info()['index']

            # Check if we're in single mic mode
            if mic2_index is None or mic2_index == mic1_index:
                self.single_mic_mode = True
                mic2_index = mic1_index

            # Print device info
            mic1_name = self.pyaudio.get_device_info_by_index(mic1_index)['name']
            if self.single_mic_mode:
                print(f"[AUDIO] Single mic mode: [{mic1_index}] {mic1_name}")
                print("[AUDIO] All speech will be processed as conversation context")
            else:
                mic2_name = self.pyaudio.get_device_info_by_index(mic2_index)['name']
                print(f"[AUDIO] Mic 1 (User): [{mic1_index}] {mic1_name}")
                print(f"[AUDIO] Mic 2 (Other): [{mic2_index}] {mic2_name}")

            # Setup queues
            self.audio_queue_user = Queue()
            self.audio_queue_other = Queue()

            # Set running BEFORE starting any threads!
            self.running = True

            if self.single_mic_mode:
                # Single mic mode - one stream, process as general conversation
                print("[AUDIO] Creating transcription processor thread...", flush=True)
                thread1 = threading.Thread(
                    target=self._transcription_processor,
                    args=(self.audio_queue_other, "[MIC]", self.other_dir),
                    daemon=True
                )
                print("[AUDIO] Starting transcription processor thread...", flush=True)
                thread1.start()
                print(f"[AUDIO] Thread started, is_alive: {thread1.is_alive()}", flush=True)
                self.threads.append(thread1)

                # Setup VAD
                vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)

                # Open single audio stream
                stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    input_device_index=mic1_index,
                    frames_per_buffer=self.FRAME_SIZE
                )
                self.streams = [stream]

                # Start single mic capture thread
                capture_thread = threading.Thread(
                    target=self._capture_loop_single,
                    args=(stream, vad),
                    daemon=True
                )
                capture_thread.start()
                self.threads.append(capture_thread)
            else:
                # Dual mic mode - two streams for user and other person
                thread1 = threading.Thread(
                    target=self._transcription_processor,
                    args=(self.audio_queue_user, "[USER]", self.user_dir),
                    daemon=True
                )
                thread2 = threading.Thread(
                    target=self._transcription_processor,
                    args=(self.audio_queue_other, "[OTHER]", self.other_dir),
                    daemon=True
                )
                thread1.start()
                thread2.start()
                self.threads.extend([thread1, thread2])

                # Setup VAD
                vad_user = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
                vad_other = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)

                # Open audio streams
                stream_user = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    input_device_index=mic1_index,
                    frames_per_buffer=self.FRAME_SIZE
                )
                stream_other = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    input_device_index=mic2_index,
                    frames_per_buffer=self.FRAME_SIZE
                )
                self.streams = [stream_user, stream_other]

                # Start dual mic capture thread
                capture_thread = threading.Thread(
                    target=self._capture_loop_dual,
                    args=(stream_user, stream_other, vad_user, vad_other),
                    daemon=True
                )
                capture_thread.start()
                self.threads.append(capture_thread)

            print("[AUDIO] Context service started")
            return True

        except Exception as e:
            print(f"[AUDIO] Failed to start context service: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Stop audio capture and cleanup."""
        self.running = False

        # Close streams
        for stream in self.streams:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        self.streams.clear()

        # Terminate PyAudio
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except:
                pass
            self.pyaudio = None

        print("[AUDIO] Context service stopped")

    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        return self.current_state.copy()

    def _capture_loop_single(self, stream, vad) -> None:
        """Single mic audio capture loop."""
        print("[AUDIO] Single mic capture loop started")
        state = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}

        while self.running:
            try:
                frame = stream.read(self.FRAME_SIZE, exception_on_overflow=False)
                self._process_frame(frame, vad, state, self.audio_queue_other, "[MIC]")
            except Exception as e:
                if self.running:
                    print(f"[AUDIO] Capture error: {e}")
                break

    def _capture_loop_dual(self, stream_user, stream_other, vad_user, vad_other) -> None:
        """Dual mic audio capture loop processing both microphones."""
        # State tracking
        state_user = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}
        state_other = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}

        while self.running:
            try:
                # Process user mic
                frame_user = stream_user.read(self.FRAME_SIZE, exception_on_overflow=False)
                self._process_frame(
                    frame_user, vad_user, state_user,
                    self.audio_queue_user, "[USER]"
                )

                # Process other mic
                frame_other = stream_other.read(self.FRAME_SIZE, exception_on_overflow=False)
                self._process_frame(
                    frame_other, vad_other, state_other,
                    self.audio_queue_other, "[OTHER]"
                )

            except Exception as e:
                if self.running:
                    print(f"[AUDIO] Capture error: {e}")
                break

    def _process_frame(self, frame: bytes, vad, state: dict, queue: Queue, label: str) -> None:
        """Process a single audio frame with VAD."""
        is_speech = vad.is_speech(frame, self.SAMPLE_RATE)

        if is_speech:
            if not state['speaking']:
                print(f"[AUDIO] {label} Speech detected")
                state['speaking'] = True
            state['frames'].append(frame)
            state['speech_count'] += 1
            state['silence'] = 0
        elif state['speaking']:
            state['silence'] += 1
            state['frames'].append(frame)

            if state['silence'] >= self.SILENCE_THRESHOLD:
                if state['speech_count'] >= self.MIN_SPEECH_FRAMES:
                    duration_ms = len(state['frames']) * self.FRAME_DURATION_MS
                    audio_data = b''.join(state['frames'])
                    print(f"[AUDIO] {label} Speech complete ({duration_ms}ms) - queuing {len(audio_data)} bytes")
                    queue.put({
                        'audio': audio_data,
                        'timestamp': time.time()
                    })
                    print(f"[AUDIO] {label} Audio queued, queue size: {queue.qsize()}")
                # Reset state
                state['frames'] = []
                state['silence'] = 0
                state['speaking'] = False
                state['speech_count'] = 0

    def _transcription_processor(self, audio_queue: Queue, label: str, directory: Path) -> None:
        """Background processor for transcriptions."""
        try:
            print(f"[AUDIO] {label} Transcription processor thread STARTING...", flush=True)
            utterances_batch = []
            last_utterance_time = None
            print(f"[AUDIO] {label} Transcription processor ready, waiting for audio...", flush=True)

            while self.running:
                try:
                    utterance_data = audio_queue.get(timeout=0.1)
                    print(f"[AUDIO] {label} Got audio from queue, transcribing...")
                    text = self._transcribe_audio(utterance_data['audio'])

                    if text:
                        utterances_batch.append(text)
                        last_utterance_time = time.time()

                        # Update recent utterance in state
                        self.current_state['recent_utterance'] = text

                        # Check for questions
                        if '?' in text:
                            self.current_state['question'] = text

                        word_count = sum(len(u.split()) for u in utterances_batch)

                        should_process = (
                            len(utterances_batch) >= self.MAX_UTTERANCES_BEFORE_UPDATE or
                            word_count >= self.MIN_WORDS_FOR_UPDATE
                        )

                        if should_process:
                            self._process_utterances(utterances_batch, label, directory)
                            utterances_batch = []
                            last_utterance_time = None

                except Empty:
                    # Check if we should process due to timeout
                    if utterances_batch and last_utterance_time:
                        if time.time() - last_utterance_time >= self.MAX_WAIT_TIME:
                            self._process_utterances(utterances_batch, label, directory)
                            utterances_batch = []
                            last_utterance_time = None
                except Exception as e:
                    if self.running:
                        print(f"[AUDIO] {label} Processor error: {e}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"[AUDIO] {label} THREAD CRASHED: {e}")
            import traceback
            traceback.print_exc()

    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio using OpenAI Whisper."""
        print(f"[WHISPER] Sending {len(audio_bytes)} bytes to Whisper API...", flush=True)
        tmp_path = None
        try:
            # Create temp file (closed immediately so Windows doesn't lock it)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # Write WAV data
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(audio_bytes)

            # Send to Whisper
            with open(tmp_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            text = transcript.text.strip()
            print(f"[WHISPER] Got transcription: \"{text}\"", flush=True)
            return text
        except Exception as e:
            print(f"[WHISPER] ERROR: {e}", flush=True)
            return ""
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Ignore cleanup errors on Windows

    def _process_utterances(self, utterances_batch: List[str], label: str, directory: Path) -> None:
        """Process batch of utterances - save context and update summary."""
        if not utterances_batch:
            return

        combined_text = " ".join(utterances_batch)
        word_count = len(combined_text.split())

        print(f"[AUDIO] {label} \"{combined_text[:80]}...\"" if len(combined_text) > 80 else f"[AUDIO] {label} \"{combined_text}\"")

        # Save context file
        context_file = self._save_context(directory, combined_text, len(utterances_batch), word_count)
        print(f"[AUDIO] Saved {context_file.name}")

        # Update final state with summary
        try:
            self._update_final_state()
            print("[AUDIO] Updated conversation summary")
        except Exception as e:
            print(f"[AUDIO] Failed to update summary: {e}")

    def _save_context(self, directory: Path, text: str, utterance_count: int, word_count: int) -> Path:
        """Save context to numbered JSON file (cycles 1-5)."""
        num = self._get_next_context_number(directory)
        filename = directory / f"context_{num:03d}.json"

        data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "text": text,
            "utterance_count": utterance_count,
            "word_count": word_count
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return filename

    def _get_next_context_number(self, directory: Path) -> int:
        """Get next context file number (cycles 1-5)."""
        existing = list(directory.glob("context_*.json"))

        if not existing:
            return 1

        nums = []
        for f in existing:
            try:
                num = int(f.stem.split('_')[1])
                if 1 <= num <= 5:
                    nums.append(num)
            except:
                f.unlink()

        if not nums:
            return 1

        # If we haven't used all 5 slots, use the next available
        if len(nums) < 5:
            for i in range(1, 6):
                if i not in nums:
                    return i

        # Otherwise, replace the oldest
        file_ages = [(directory / f"context_{n:03d}.json", n) for n in nums]
        file_ages = [(f, n) for f, n in file_ages if f.exists()]

        if file_ages:
            oldest = min(file_ages, key=lambda x: x[0].stat().st_mtime)
            return oldest[1]

        return 1

    def _get_most_recent_context_files(self, directory: Path, count: int = 2) -> List[dict]:
        """Get the most recent N context files from a directory."""
        existing = list(directory.glob("context_*.json"))

        if not existing:
            return []

        existing.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        recent_files = []
        for f in existing[:count]:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    recent_files.append({
                        'filename': f.name,
                        'number': int(f.stem.split('_')[1]),
                        'data': data
                    })
            except:
                continue

        return recent_files

    def _update_final_state(self) -> None:
        """Update state with summary based on recent contexts."""
        user_contexts = self._get_most_recent_context_files(self.user_dir, count=2)
        other_contexts = self._get_most_recent_context_files(self.other_dir, count=2)

        if not user_contexts and not other_contexts:
            return

        # Build prompt based on mode
        if self.single_mic_mode:
            # Single mic mode - just summarize the conversation
            context_text = "Recent conversation:\n"
            for ctx in sorted(other_contexts, key=lambda x: x['data']['timestamp']):
                context_text += f"- [{ctx['data']['timestamp']}] {ctx['data']['text']}\n"

            prompt = f"""Based on the following conversation snippets, provide a concise summary of what is being discussed.

{context_text}

Provide a brief summary (2-3 sentences max) of the conversation topic and key points."""
        else:
            # Dual mic mode - summarize both speakers
            user_text = ""
            if user_contexts:
                user_text = "USER's recent statements:\n"
                for ctx in sorted(user_contexts, key=lambda x: x['data']['timestamp']):
                    user_text += f"- [{ctx['data']['timestamp']}] {ctx['data']['text']}\n"

            other_text = ""
            if other_contexts:
                other_text = "\nOTHER PERSON's recent statements:\n"
                for ctx in sorted(other_contexts, key=lambda x: x['data']['timestamp']):
                    other_text += f"- [{ctx['data']['timestamp']}] {ctx['data']['text']}\n"

            prompt = f"""Based on the following conversation snippets, provide a concise summary of the current conversation state - what they're discussing and the key points so far.

{user_text}{other_text}

Provide a brief summary (2-3 sentences max) of what the conversation is about and its current state."""

        try:
            print(f"[GPT] Sending to GPT-4o-mini for summary...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a conversation analyst. Summarize conversation states concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            summary = response.choices[0].message.content.strip()
            print(f"[GPT] Got summary: \"{summary}\"")

            # Update current state
            self.current_state['convo_state_summary'] = summary
            self.current_state['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Write to file for overlay
            self.write_state(self.current_state)

        except Exception as e:
            print(f"[GPT] ERROR: {e}")

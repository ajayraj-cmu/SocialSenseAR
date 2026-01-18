#!/usr/bin/env python3
"""
Process Video Audio - Extract audio and generate transcriptions for demo rendering.

Pass 1 of the two-pass demo workflow:
1. Extract audio from video
2. Run Whisper transcription with timestamps
3. Run GPT summarization
4. Generate timed state JSON file

Then use render_demo.py --script output.json for smooth rendering.

Usage:
    python process_video_audio.py input.mp4 output_script.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def find_ffmpeg() -> str:
    """Find ffmpeg executable, checking common installation locations."""
    # Check PATH first
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    # Common Windows locations
    common_paths = [
        r"C:\Program Files\ShareX\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video as 16kHz mono WAV for Whisper."""
    print(f"[AUDIO] Extracting audio from {video_path.name}...")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[AUDIO] Error: ffmpeg not found. Please install ffmpeg.")
        return False

    print(f"[AUDIO] Using ffmpeg: {ffmpeg}")

    try:
        cmd = [
            ffmpeg, '-y', '-i', str(video_path),
            '-vn',                    # No video
            '-acodec', 'pcm_s16le',   # PCM 16-bit
            '-ar', '16000',           # 16kHz for Whisper
            '-ac', '1',               # Mono
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[AUDIO] ffmpeg error: {result.stderr}")
            return False
        print(f"[AUDIO] Audio extracted to {audio_path}")
        return True
    except FileNotFoundError:
        print("[AUDIO] Error: ffmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"[AUDIO] Error extracting audio: {e}")
        return False


def transcribe_audio(audio_path: Path) -> list:
    """Transcribe audio using OpenAI Whisper API with timestamps."""
    print("[WHISPER] Transcribing audio...")

    try:
        from openai import OpenAI
        client = OpenAI()

        with open(audio_path, 'rb') as audio_file:
            # Use whisper-1 model with timestamp granularities
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                # Handle both dict and object formats
                start = seg.start if hasattr(seg, 'start') else seg['start']
                end = seg.end if hasattr(seg, 'end') else seg['end']
                text = seg.text if hasattr(seg, 'text') else seg['text']
                text = text.strip()
                segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                print(f"  [{start:.1f}s - {end:.1f}s] {text}")

        print(f"[WHISPER] Transcribed {len(segments)} segments")
        return segments

    except Exception as e:
        print(f"[WHISPER] Error: {e}")
        return []


def generate_summaries(segments: list) -> list:
    """Generate conversation summaries using GPT for each segment."""
    print("[GPT] Generating summaries...")

    try:
        from openai import OpenAI
        client = OpenAI()

        # Group segments into chunks for summarization
        states = []
        context_window = []

        for i, seg in enumerate(segments):
            context_window.append(seg['text'])

            # Keep last 3 utterances for context
            if len(context_window) > 3:
                context_window.pop(0)

            # Generate summary
            context_text = "\n".join([f"- {t}" for t in context_window])

            prompt = f"""Recent conversation:
{context_text}

Respond with ONLY a JSON object:
{{"summary": "<max 60 chars: what they're discussing>", "question": "<if they asked a question, otherwise empty string>", "vibe": "<one word: engaged/bored/upset/excited/neutral>"}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You summarize conversations concisely for AR display. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )

                result_text = response.choices[0].message.content.strip()
                # Clean up markdown if present
                if result_text.startswith('```'):
                    result_text = result_text.split('\n', 1)[1].rsplit('\n', 1)[0]

                result = json.loads(result_text)

                states.append({
                    'time': seg['start'],
                    'state': {
                        'convo_state_summary': result.get('summary', 'Listening...'),
                        'recent_utterance': seg['text'],
                        'question': result.get('question', ''),
                        'emotion': 'neutral',  # Will be filled by emotion detection during render
                        'emotion_display': 'Neutral',
                        'is_other_speaking': True,
                        'vibe': result.get('vibe', 'neutral')
                    }
                })

                print(f"  [{seg['start']:.1f}s] {result.get('summary', '')[:50]}...")

            except json.JSONDecodeError:
                # Fallback if GPT doesn't return valid JSON
                states.append({
                    'time': seg['start'],
                    'state': {
                        'convo_state_summary': 'Listening...',
                        'recent_utterance': seg['text'],
                        'question': '',
                        'emotion': 'neutral',
                        'emotion_display': 'Neutral',
                        'is_other_speaking': True
                    }
                })

        print(f"[GPT] Generated {len(states)} state entries")
        return states

    except Exception as e:
        print(f"[GPT] Error: {e}")
        # Return basic states without summaries
        return [{
            'time': seg['start'],
            'state': {
                'convo_state_summary': 'Listening...',
                'recent_utterance': seg['text'],
                'question': '',
                'emotion': 'neutral',
                'emotion_display': 'Neutral',
                'is_other_speaking': True
            }
        } for seg in segments]


def add_speaking_indicators(states: list) -> list:
    """Add speaking/pause indicators between segments."""
    if not states:
        return states

    enhanced = []

    # Add initial state
    enhanced.append({
        'time': 0.0,
        'state': {
            'convo_state_summary': 'Starting...',
            'recent_utterance': '',
            'question': '',
            'emotion': 'neutral',
            'emotion_display': 'Neutral',
            'is_other_speaking': False
        }
    })

    for i, state in enumerate(states):
        # Add speaking indicator
        state['state']['is_other_speaking'] = True
        enhanced.append(state)

        # Add pause after speech if there's a gap
        if i < len(states) - 1:
            next_start = states[i + 1]['time']
            current_end = state['time'] + 2.0  # Assume ~2s per utterance

            if next_start - current_end > 1.0:
                # Add pause state
                enhanced.append({
                    'time': current_end,
                    'state': {
                        'convo_state_summary': state['state']['convo_state_summary'],
                        'recent_utterance': state['state']['recent_utterance'],
                        'question': state['state']['question'],
                        'emotion': 'neutral',
                        'emotion_display': 'Neutral',
                        'is_other_speaking': False
                    }
                })

    return enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Process video audio for demo rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-pass demo workflow:
  1. python process_video_audio.py input.mp4 script.json
  2. python render_demo.py input.mp4 output.mp4 --script script.json
        """
    )

    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output JSON script file')
    parser.add_argument('--keep-audio', action='store_true',
                        help='Keep extracted audio file')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Video Audio Processor")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Extract audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        if not extract_audio(input_path, audio_path):
            sys.exit(1)

        # Transcribe
        segments = transcribe_audio(audio_path)
        if not segments:
            print("Warning: No speech detected in audio")

        # Generate summaries
        states = generate_summaries(segments)

        # Add speaking indicators
        states = add_speaking_indicators(states)

        # Save script
        with open(output_path, 'w') as f:
            json.dump(states, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Script saved to: {output_path}")
        print(f"Total state changes: {len(states)}")
        print(f"\nNext step:")
        print(f"  python render_demo.py {input_path} output.mp4 --script {output_path}")
        print(f"{'='*60}\n")

    finally:
        # Cleanup
        if not args.keep_audio and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()

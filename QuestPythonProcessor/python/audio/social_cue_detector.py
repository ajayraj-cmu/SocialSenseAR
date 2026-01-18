"""
Social Cue Detector - Rule-based detection of social cues for neurodivergent users.

Detects patterns in speech that may indicate:
- Sarcasm
- Rhetorical questions
- Conversation ending signals
- Topic changes
- Polite disagreement
- Disengagement
- And more...
"""
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class CueType(Enum):
    """Types of social cues we can detect."""
    SARCASM = "sarcasm"
    RHETORICAL = "rhetorical"
    ENDING = "ending"
    TOPIC_CHANGE = "topic_change"
    DISAGREEMENT = "disagreement"
    UNCERTAINTY = "uncertainty"
    FRUSTRATION = "frustration"
    PASSIVE_AGGRESSION = "passive_aggression"
    HUMOR = "humor"
    VALIDATION_SEEKING = "validation_seeking"
    DISENGAGEMENT = "disengagement"
    WAIT_YOUR_TURN = "wait_turn"


@dataclass
class SocialCue:
    """A detected social cue."""
    cue_type: CueType
    message: str
    icon: str
    confidence: float
    timestamp: float
    source_text: str


class SocialCueDetector:
    """Detects social cues from transcribed speech."""

    # Sarcasm patterns - positive words that may be sarcastic
    SARCASM_POSITIVE_WORDS = [
        "great", "wonderful", "fantastic", "amazing", "brilliant",
        "perfect", "lovely", "awesome", "terrific", "marvelous"
    ]
    SARCASM_PHRASES = [
        r"oh,?\s*(great|wonderful|fantastic|perfect)",
        r"yeah,?\s*right",
        r"sure,?\s*(thing|whatever)",
        r"as if",
        r"how\s+nice",
        r"that'?s?\s+just\s+(great|wonderful|perfect)",
        r"well,?\s+isn'?t\s+that\s+(nice|special|something)",
    ]

    # Rhetorical question patterns
    RHETORICAL_PATTERNS = [
        r"who\s+(even\s+)?(does|cares|knows)\s+that",
        r"what\s+did\s+I\s+(just\s+)?tell\s+you",
        r"how\s+hard\s+(can|is)\s+(it|that)",
        r"are\s+you\s+(even\s+)?(listening|kidding|serious)",
        r"do\s+I\s+(really\s+)?(have\s+to|need\s+to|look\s+like)",
        r"why\s+would\s+(I|anyone)",
        r"can\s+you\s+believe",
    ]

    # Conversation ending signals
    ENDING_PHRASES = [
        r"anyway,?\s*$",
        r"I\s+should\s+(go|get\s+going|head\s+out|run)",
        r"(good|nice)\s+talking\s+to\s+you",
        r"take\s+care",
        r"catch\s+(you|ya)\s+later",
        r"gotta\s+(go|run)",
        r"I('?ll|\s+will)\s+let\s+you\s+go",
        r"we('?ll|\s+will)\s+talk\s+(later|soon)",
        r"see\s+you\s+(later|around|soon)",
        r"(well|okay|alright),?\s*$",
    ]

    # Topic change signals
    TOPIC_CHANGE_PHRASES = [
        r"speaking\s+of",
        r"by\s+the\s+way",
        r"that\s+reminds\s+me",
        r"oh,?\s+(also|and)",
        r"anyway,?\s+so",
        r"(moving|getting)\s+on",
        r"on\s+another\s+note",
        r"while\s+(we'?re|I'm)\s+(at\s+it|here)",
        r"changing\s+(the\s+)?subject",
    ]

    # Polite disagreement patterns
    DISAGREEMENT_PHRASES = [
        r"I\s+see\s+what\s+you\s+mean,?\s*but",
        r"that'?s?\s+interesting,?\s*(but|however)",
        r"with\s+(all\s+due\s+)?respect",
        r"I\s+(hear|understand)\s+you,?\s*but",
        r"not\s+to\s+disagree,?\s*but",
        r"I\s+(kind\s+of|kinda)\s+disagree",
        r"actually,?\s*I\s+think",
        r"well,?\s*not\s+(exactly|really|quite)",
        r"I'?m\s+not\s+(so\s+)?sure\s+(about\s+that|I\s+agree)",
    ]

    # Uncertainty/hedging patterns
    UNCERTAINTY_PHRASES = [
        r"I\s+guess",
        r"I\s+think\s+maybe",
        r"(kind\s+of|kinda|sort\s+of|sorta)",
        r"I\s+(don'?t|do\s+not)\s+know",
        r"probably",
        r"might\s+be",
        r"I'?m\s+not\s+(really\s+)?sure",
        r"could\s+be",
        r"maybe",
    ]

    # Frustration signals
    FRUSTRATION_PHRASES = [
        r"look,?\.{0,3}\s",
        r"I\s+already\s+(said|told|explained)",
        r"for\s+the\s+(last|third|fourth)\s+time",
        r"how\s+many\s+times",
        r"I'?ve\s+(said|told)\s+you",
        r"as\s+I\s+(said|mentioned)",
        r"(ugh|argh|jeez)",
        r"(seriously|honestly),?\s*$",
        r"come\s+on,?\s*$",
    ]

    # Passive aggression patterns
    PASSIVE_AGGRESSION_PHRASES = [
        r"^fine\.?\s*$",
        r"whatever\s+(you\s+say)?",
        r"I'?m\s+not\s+mad",
        r"(it'?s|that'?s)\s+fine,?\s*really",
        r"do\s+what(ever)?\s+you\s+want",
        r"if\s+that'?s\s+what\s+you\s+think",
        r"sure,?\s*$",
        r"okay,?\s+then",
        r"^no,?\s+it'?s\s+(okay|fine)",
    ]

    # Humor/joke signals
    HUMOR_PHRASES = [
        r"just\s+kidding",
        r"I'?m\s+(just\s+)?joking",
        r"(haha|hehe|lol|lmao)",
        r"I'?m\s+(just\s+)?messing\s+with\s+you",
        r"gotcha",
        r"pulling\s+your\s+leg",
    ]

    # Validation seeking
    VALIDATION_PHRASES = [
        r"right\s*\?",
        r"you\s+know\s*\?",
        r"don'?t\s+you\s+think\s*\?",
        r"wouldn'?t\s+you\s+agree\s*\?",
        r"isn'?t\s+(it|that)\s+(right|true)\s*\?",
        r"(am|are)\s+I\s+(right|wrong)\s*\?",
        r"makes\s+sense,?\s*right\s*\?",
    ]

    # Disengagement signals (short responses)
    SHORT_RESPONSES = [
        "uh huh", "mm hmm", "mmm", "yeah", "yep", "yup",
        "okay", "ok", "sure", "cool", "right", "interesting"
    ]

    def __init__(self):
        """Initialize the social cue detector."""
        # Recent cues for deduplication
        self.recent_cues: deque = deque(maxlen=10)

        # Compile regex patterns for performance
        self._compiled_patterns = {
            CueType.SARCASM: [re.compile(p, re.IGNORECASE) for p in self.SARCASM_PHRASES],
            CueType.RHETORICAL: [re.compile(p, re.IGNORECASE) for p in self.RHETORICAL_PATTERNS],
            CueType.ENDING: [re.compile(p, re.IGNORECASE) for p in self.ENDING_PHRASES],
            CueType.TOPIC_CHANGE: [re.compile(p, re.IGNORECASE) for p in self.TOPIC_CHANGE_PHRASES],
            CueType.DISAGREEMENT: [re.compile(p, re.IGNORECASE) for p in self.DISAGREEMENT_PHRASES],
            CueType.UNCERTAINTY: [re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_PHRASES],
            CueType.FRUSTRATION: [re.compile(p, re.IGNORECASE) for p in self.FRUSTRATION_PHRASES],
            CueType.PASSIVE_AGGRESSION: [re.compile(p, re.IGNORECASE) for p in self.PASSIVE_AGGRESSION_PHRASES],
            CueType.HUMOR: [re.compile(p, re.IGNORECASE) for p in self.HUMOR_PHRASES],
            CueType.VALIDATION_SEEKING: [re.compile(p, re.IGNORECASE) for p in self.VALIDATION_PHRASES],
        }

        # Cue display info
        self._cue_info = {
            CueType.SARCASM: ("⚠️", "May be sarcastic"),
            CueType.RHETORICAL: ("ℹ️", "No answer expected"),
            CueType.ENDING: ("🚪", "Wrapping up soon"),
            CueType.TOPIC_CHANGE: ("🔄", "New topic"),
            CueType.DISAGREEMENT: ("⚡", "May disagree politely"),
            CueType.UNCERTAINTY: ("🤔", "They seem unsure"),
            CueType.FRUSTRATION: ("😤", "May be frustrated"),
            CueType.PASSIVE_AGGRESSION: ("⚠️", "May not mean it"),
            CueType.HUMOR: ("😄", "Probably joking"),
            CueType.VALIDATION_SEEKING: ("👍", "They want agreement"),
            CueType.DISENGAGEMENT: ("😐", "May be losing interest"),
            CueType.WAIT_YOUR_TURN: ("🛑", "Wait - they're still talking"),
        }

    def detect(self, text: str, emotion: Optional[str] = None,
               is_other_speaking: bool = False) -> List[SocialCue]:
        """Detect social cues in transcribed text.

        Args:
            text: Transcribed text to analyze
            emotion: Current detected emotion (optional, for combined analysis)
            is_other_speaking: Whether the other person is currently speaking

        Returns:
            List of detected social cues
        """
        detected = []
        text_lower = text.lower().strip()
        now = time.time()

        # Check each pattern type
        for cue_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    icon, message = self._cue_info[cue_type]

                    # Adjust message based on emotion context
                    message = self._adjust_message_with_emotion(cue_type, message, emotion)

                    cue = SocialCue(
                        cue_type=cue_type,
                        message=message,
                        icon=icon,
                        confidence=0.8,
                        timestamp=now,
                        source_text=text[:50]
                    )
                    detected.append(cue)
                    break  # Only report each type once per text

        # Check for disengagement (short responses)
        if text_lower in self.SHORT_RESPONSES or len(text.split()) <= 2:
            # Only flag if we've seen multiple short responses recently
            self.recent_cues.append(("short", now))
            short_count = sum(1 for cue, ts in self.recent_cues
                              if cue == "short" and now - ts < 30)
            if short_count >= 3:
                icon, message = self._cue_info[CueType.DISENGAGEMENT]
                detected.append(SocialCue(
                    cue_type=CueType.DISENGAGEMENT,
                    message=message,
                    icon=icon,
                    confidence=0.6,
                    timestamp=now,
                    source_text=text[:50]
                ))

        # Combined detection: words don't match emotion
        detected.extend(self._detect_word_emotion_mismatch(text, emotion, now))

        return detected

    def _adjust_message_with_emotion(self, cue_type: CueType, message: str,
                                      emotion: Optional[str]) -> str:
        """Adjust cue message based on emotion context."""
        if not emotion:
            return message

        emotion = emotion.lower()

        # Sarcasm is more likely with negative emotions
        if cue_type == CueType.SARCASM and emotion in ['angry', 'disgust', 'sad']:
            return "Likely sarcastic (expression doesn't match)"

        # Passive aggression with angry/sad face
        if cue_type == CueType.PASSIVE_AGGRESSION and emotion in ['angry', 'sad']:
            return "Words don't match expression"

        return message

    def _detect_word_emotion_mismatch(self, text: str, emotion: Optional[str],
                                       now: float) -> List[SocialCue]:
        """Detect when words don't match facial expression."""
        if not emotion:
            return []

        emotion = emotion.lower()
        text_lower = text.lower()
        detected = []

        # Positive words with negative expression
        positive_indicators = any(word in text_lower for word in
                                  ['great', 'good', 'fine', 'happy', 'love', 'wonderful', 'amazing'])
        negative_emotions = ['angry', 'disgust', 'sad', 'fear']

        if positive_indicators and emotion in negative_emotions:
            detected.append(SocialCue(
                cue_type=CueType.SARCASM,
                message="Words don't match expression",
                icon="⚠️",
                confidence=0.9,
                timestamp=now,
                source_text=text[:50]
            ))

        # "I'm fine" / "It's okay" with sad/angry expression
        fine_phrases = ['i\'m fine', 'im fine', 'it\'s fine', 'its fine', 'it\'s okay', 'its okay', 'i\'m okay', 'im okay']
        if any(phrase in text_lower for phrase in fine_phrases) and emotion in ['sad', 'angry']:
            detected.append(SocialCue(
                cue_type=CueType.PASSIVE_AGGRESSION,
                message="They may not be fine",
                icon="⚠️",
                confidence=0.85,
                timestamp=now,
                source_text=text[:50]
            ))

        return detected

    def create_interruption_warning(self) -> SocialCue:
        """Create a warning cue when user might be interrupting."""
        icon, message = self._cue_info[CueType.WAIT_YOUR_TURN]
        return SocialCue(
            cue_type=CueType.WAIT_YOUR_TURN,
            message=message,
            icon=icon,
            confidence=1.0,
            timestamp=time.time(),
            source_text=""
        )


# Global detector instance
_detector: Optional[SocialCueDetector] = None


def get_detector() -> SocialCueDetector:
    """Get or create the global social cue detector."""
    global _detector
    if _detector is None:
        _detector = SocialCueDetector()
    return _detector


def detect_social_cues(text: str, emotion: Optional[str] = None) -> List[SocialCue]:
    """Convenience function to detect social cues."""
    detector = get_detector()
    return detector.detect(text, emotion)

"""
Voice Notes Integration Module.

Transcribes audio notes and matches them to training timestamps.
Uses OpenAI Whisper API or local whisper.cpp for transcription.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from io import BytesIO
from datetime import datetime
import json
import os


@dataclass
class TranscribedNote:
    """A single transcribed voice note."""
    text: str
    start_time: float  # seconds in audio
    end_time: float
    confidence: float = 1.0


@dataclass 
class TimedNote:
    """Voice note matched to training timestamp."""
    text: str
    training_time_sec: float  # Time in training session
    audio_time_sec: float      # Time in audio file
    confidence: float = 1.0
    
    @property
    def time_str(self) -> str:
        """Format time as MM:SS."""
        mins, secs = divmod(int(self.training_time_sec), 60)
        return f"{mins:02d}:{secs:02d}"


class VoiceNotesProcessor:
    """Process voice notes for training sessions."""
    
    def __init__(self, api_key: Optional[str] = None, use_local: bool = True):
        """
        Args:
            api_key: OpenAI API key (optional, for cloud transcription)
            use_local: Whether to use local whisper.cpp (default True)
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.use_local = use_local
        self._whisper_available = self._check_whisper()
    
    def _check_whisper(self) -> bool:
        """Check if local whisper is available."""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def transcribe(
        self, 
        audio_data: BytesIO,
        language: str = "pl"
    ) -> List[TranscribedNote]:
        """Transcribe audio file to text with timestamps.
        
        Args:
            audio_data: Audio file bytes
            language: Language code (default Polish)
            
        Returns:
            List of transcribed notes with timestamps
        """
        if self.use_local and self._whisper_available:
            return self._transcribe_local(audio_data, language)
        elif self.api_key:
            return self._transcribe_openai(audio_data, language)
        else:
            # Fallback: return empty
            return []
    
    def _transcribe_local(
        self, 
        audio_data: BytesIO,
        language: str
    ) -> List[TranscribedNote]:
        """Transcribe using local whisper model."""
        try:
            import whisper
            import tempfile
            
            # Save to temp file (whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_data.read())
                temp_path = f.name
            
            # Load model (use tiny for speed)
            model = whisper.load_model("tiny")
            
            # Transcribe with word timestamps
            result = model.transcribe(
                temp_path,
                language=language,
                word_timestamps=True
            )
            
            # Clean up
            os.unlink(temp_path)
            
            # Convert to TranscribedNote objects
            notes = []
            for segment in result.get('segments', []):
                notes.append(TranscribedNote(
                    text=segment['text'].strip(),
                    start_time=segment['start'],
                    end_time=segment['end'],
                    confidence=segment.get('no_speech_prob', 0)
                ))
            
            return notes
            
        except Exception as e:
            print(f"Local transcription failed: {e}")
            return []
    
    def _transcribe_openai(
        self, 
        audio_data: BytesIO,
        language: str
    ) -> List[TranscribedNote]:
        """Transcribe using OpenAI Whisper API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Reset buffer position
            audio_data.seek(0)
            
            # Call API
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
            # Convert to TranscribedNote objects
            notes = []
            for segment in response.segments:
                notes.append(TranscribedNote(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=1.0 - segment.no_speech_prob
                ))
            
            return notes
            
        except Exception as e:
            print(f"OpenAI transcription failed: {e}")
            return []
    
    def match_timestamps(
        self,
        notes: List[TranscribedNote],
        training_start_sec: float,
        audio_offset_sec: float = 0
    ) -> List[TimedNote]:
        """Match transcribed notes to training session timestamps.
        
        Args:
            notes: List of transcribed notes
            training_start_sec: Training session start time (seconds from midnight)
            audio_offset_sec: Offset between audio start and training start
            
        Returns:
            List of TimedNote with training timestamps
        """
        timed_notes = []
        
        for note in notes:
            # Calculate training time
            training_time = note.start_time + audio_offset_sec
            
            timed_notes.append(TimedNote(
                text=note.text,
                training_time_sec=training_time,
                audio_time_sec=note.start_time,
                confidence=note.confidence
            ))
        
        return timed_notes
    
    def detect_keywords(
        self, 
        notes: List[TranscribedNote]
    ) -> dict:
        """Detect training-related keywords in notes.
        
        Returns dict with detected elements:
        - intervals: mentions of intervals/reps
        - intensity: RPE or intensity mentions
        - feelings: physical sensations
        - issues: problems/pain mentions
        """
        keywords = {
            'intervals': ['interwał', 'seria', 'powtórzenie', 'rep', 'interval'],
            'intensity': ['rpe', 'ciężko', 'lekko', 'mocno', 'max', 'tempo'],
            'feelings': ['nogi', 'oddech', 'ból', 'zmęczenie', 'dobrze', 'źle'],
            'issues': ['ból', 'kurcz', 'problem', 'słabo', 'mdłości']
        }
        
        results = {category: [] for category in keywords}
        
        for note in notes:
            text_lower = note.text.lower()
            for category, words in keywords.items():
                for word in words:
                    if word in text_lower:
                        results[category].append({
                            'text': note.text,
                            'time': note.start_time,
                            'keyword': word
                        })
        
        return results
    
    def generate_summary(self, notes: List[TranscribedNote]) -> str:
        """Generate a summary of voice notes.
        
        Args:
            notes: List of transcribed notes
            
        Returns:
            Summary string
        """
        if not notes:
            return "Brak notatek głosowych."
        
        # Combine all text
        full_text = " ".join(n.text for n in notes)
        
        # Count notes
        num_notes = len(notes)
        total_duration = notes[-1].end_time if notes else 0
        
        # Detect keywords
        keywords = self.detect_keywords(notes)
        
        summary_parts = [
            f"📝 {num_notes} notatek głosowych ({total_duration:.0f}s audio)",
        ]
        
        if keywords['intensity']:
            summary_parts.append(f"⚡ Wspomniana intensywność: {len(keywords['intensity'])}x")
        if keywords['issues']:
            summary_parts.append(f"⚠️ Zgłoszone problemy: {len(keywords['issues'])}x")
        
        summary_parts.append(f"\n**Transkrypcja:**\n{full_text[:500]}...")
        
        return "\n".join(summary_parts)

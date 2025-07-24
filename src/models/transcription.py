"""
Transcription data models.

This module defines Pydantic models for transcription services and responses,
supporting multiple providers (OpenAI, Replicate, Local Whisper).
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import uuid


class TranscriptionProvider(str, Enum):
    """
    Supported transcription providers.
    
    OPENAI: OpenAI Whisper API (25MB file limit)
    REPLICATE: Replicate victor-upmeet/whisperx (larger files, word-level timestamps)
    LOCAL: Local Whisper installation
    """
    OPENAI = "openai"
    REPLICATE = "replicate"
    LOCAL = "local"


class WordTimestamp(BaseModel):
    """
    Word-level timestamp information.
    
    Attributes:
        word: The transcribed word
        start: Start time of the word (seconds)
        end: End time of the word (seconds)
        confidence: Confidence score for this word (0.0-1.0)
    """
    word: str = Field(description="The transcribed word")
    start: float = Field(ge=0, description="Start time of the word (seconds)")
    end: float = Field(ge=0, description="End time of the word (seconds)")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this word"
    )
    
    @field_validator("end")
    @classmethod
    def validate_end_time(cls, v, info):
        """Validate end time is after start time, with auto-correction."""
        start_time = info.data.get('start', 0)
        if v <= start_time:
            # Auto-correct invalid timestamps by adding a small increment
            corrected_end = start_time + 0.1
            print(f"🔧 WordTimestamp: Auto-corrected end time from {v} to {corrected_end} (start: {start_time})")
            return corrected_end
        return v
    
    @field_validator("word")
    @classmethod
    def validate_word(cls, v):
        """Validate word is not empty."""
        if not v or not v.strip():
            raise ValueError("Word cannot be empty")
        return v.strip()


class TranscriptionSegment(BaseModel):
    """
    A segment of transcribed audio with timing information.
    
    Attributes:
        id: Unique identifier for the segment
        start: Start time of the segment (seconds)
        end: End time of the segment (seconds)
        text: Transcribed text for this segment
        confidence: Overall confidence score for this segment
        words: Word-level timestamps (if available)
        speaker: Speaker identifier (if diarization is enabled)
        language: Detected language code
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    start: float = Field(ge=0, description="Start time of the segment (seconds)")
    end: float = Field(ge=0, description="End time of the segment (seconds)")
    text: str = Field(description="Transcribed text for this segment")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score for this segment"
    )
    words: Optional[List[WordTimestamp]] = Field(
        default=None,
        description="Word-level timestamps (if available)"
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker identifier (if diarization is enabled)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code"
    )
    
    @field_validator("end")
    @classmethod
    def validate_end_time(cls, v, info):
        """Validate end time is after start time."""
        if info.data.get('start') and v <= info.data['start']:
            raise ValueError("End time must be after start time")
        return v
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Segment text cannot be empty")
        return v.strip()
    
    @field_validator("words")
    @classmethod
    def validate_words_timing(cls, v, info):
        """Validate word timestamps are within segment bounds."""
        if v and info.data.get('start') and info.data.get('end'):
            segment_start = info.data['start']
            segment_end = info.data['end']
            
            for word in v:
                if word.start < segment_start or word.end > segment_end:
                    raise ValueError(f"Word timestamp {word.word} is outside segment bounds")
        return v


class TranscriptionRequest(BaseModel):
    """
    Request for transcription service.
    
    Attributes:
        audio_path: Path to the audio file to transcribe
        provider: Transcription provider to use
        language: Language code for transcription (optional)
        enable_word_timestamps: Whether to request word-level timestamps
        enable_diarization: Whether to enable speaker diarization
        model: Specific model to use (provider-dependent)
        temperature: Temperature for sampling (0.0-1.0)
        max_retries: Maximum number of retry attempts
    """
    audio_path: str = Field(description="Path to the audio file to transcribe")
    provider: TranscriptionProvider = Field(
        default=TranscriptionProvider.REPLICATE,
        description="Transcription provider to use (Replicate preferred for word-level timestamps)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription (optional)"
    )
    enable_word_timestamps: bool = Field(
        default=True,
        description="Whether to request word-level timestamps"
    )
    enable_diarization: bool = Field(
        default=False,
        description="Whether to enable speaker diarization"
    )
    model: Optional[str] = Field(
        default=None,
        description="Specific model to use (provider-dependent)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for sampling"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts"
    )
    
    @field_validator("audio_path")
    @classmethod
    def validate_audio_path(cls, v):
        """Validate audio path is not empty."""
        if not v or not v.strip():
            raise ValueError("Audio path cannot be empty")
        return v.strip()
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language code format."""
        if v and len(v) not in [2, 5]:  # ISO 639-1 (2 chars) or locale (5 chars)
            raise ValueError("Language code must be 2 or 5 characters")
        return v


class TranscriptionResponse(BaseModel):
    """
    Response from transcription service.
    
    Attributes:
        request_id: Unique identifier for this transcription request
        provider: Provider used for transcription
        status: Transcription status (success, error, timeout)
        segments: List of transcribed segments
        language: Detected language
        duration: Duration of transcribed audio (seconds)
        processing_time: Time taken to process (seconds)
        error: Error message if status is error
        metadata: Additional provider-specific metadata
    """
    request_id: str = Field(description="Unique identifier for this transcription request")
    provider: TranscriptionProvider = Field(description="Provider used for transcription")
    status: str = Field(description="Transcription status (success, error, timeout)")
    segments: List[TranscriptionSegment] = Field(
        default_factory=list,
        description="List of transcribed segments"
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language"
    )
    duration: float = Field(
        ge=0,
        description="Duration of transcribed audio (seconds)"
    )
    processing_time: float = Field(
        ge=0,
        description="Time taken to process (seconds)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is error"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific metadata"
    )
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status is one of allowed values."""
        valid_statuses = ["success", "error", "timeout", "processing"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v
    
    @field_validator("segments")
    @classmethod
    def validate_segments_order(cls, v):
        """Validate segments are in chronological order."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].start < v[i-1].end:
                    raise ValueError("Segments must be in chronological order")
        return v


class OpenAITranscriptionResponse(BaseModel):
    """
    Specific response model for OpenAI Whisper API.
    
    Based on OpenAI's verbose_json response format.
    """
    task: str = Field(description="Task type (transcribe)")
    language: str = Field(description="Detected language")
    duration: float = Field(description="Audio duration")
    text: str = Field(description="Full transcribed text")
    segments: List[Dict[str, Any]] = Field(description="Segment data")
    words: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word-level timestamps"
    )


class ReplicateTranscriptionResponse(BaseModel):
    """
    Specific response model for Replicate WhisperX API.
    
    Based on victor-upmeet/whisperx response format.
    """
    segments: List[Dict[str, Any]] = Field(description="Segment data with word timestamps")
    detected_language: str = Field(description="Detected language")
    word_segments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word-level segment data"
    )


class LocalWhisperResponse(BaseModel):
    """
    Response model for local Whisper installation.
    
    Based on whisper library response format.
    """
    text: str = Field(description="Full transcribed text")
    segments: List[Dict[str, Any]] = Field(description="Segment data")
    language: str = Field(description="Detected language")


class TranscriptionStats(BaseModel):
    """
    Statistics about transcription performance.
    
    Attributes:
        total_requests: Total number of transcription requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        average_processing_time: Average processing time (seconds)
        total_audio_duration: Total duration of transcribed audio (seconds)
        provider_stats: Per-provider statistics
    """
    total_requests: int = Field(ge=0, description="Total number of transcription requests")
    successful_requests: int = Field(ge=0, description="Number of successful requests")
    failed_requests: int = Field(ge=0, description="Number of failed requests")
    average_processing_time: float = Field(
        ge=0,
        description="Average processing time (seconds)"
    )
    total_audio_duration: float = Field(
        ge=0,
        description="Total duration of transcribed audio (seconds)"
    )
    provider_stats: Dict[str, Dict[str, Union[int, float]]] = Field(
        default_factory=dict,
        description="Per-provider statistics"
    )
    
    @field_validator("successful_requests", "failed_requests")
    @classmethod
    def validate_request_counts(cls, v, info):
        """Validate request counts are consistent."""
        if info.data.get('total_requests'):
            total = info.data['total_requests']
            if info.data.get('successful_requests'):
                successful = info.data['successful_requests']
                if v + successful > total:
                    raise ValueError("Request counts exceed total")
        return v
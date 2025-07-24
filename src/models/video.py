"""
Video processing data models.

This module defines Pydantic models for video processing requests and responses,
following the video types and patterns from Context-Engineering/INITIAL.md.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import os


class VideoType(str, Enum):
    """
    Supported video types for editing templates.
    
    Based on the video types defined in INITIAL.md:
    - Vídeos de venda
    - Anúncios publicitários  
    - Reels para redes sociais
    - Aulas educacionais
    - Trechos de lives para YouTube
    - Cortes gerais para YouTube
    - Geral (o usuário definirá o objetivo)
    """
    GENERAL = "general"
    YOUTUBE_CUTS = "youtube_cuts"
    VSL = "vsl"  # Video Sales Letter
    SOCIAL_REELS = "social_reels"
    EDUCATIONAL = "educational"
    ADVERTISING = "advertising"
    LIVE_CUTS = "live_cuts"


class VideoProcessingRequest(BaseModel):
    """
    Request model for video processing.
    
    Attributes:
        video_type: Type of video editing template to apply
        custom_instructions: Optional custom editing instructions from user
        transcription_provider: Provider to use for transcription
        compress_audio: Whether to compress audio for API limits
        target_duration: Target duration for final video (in seconds)
        compression_ratio: Desired compression ratio (0.1-0.9)
    """
    video_type: VideoType = Field(
        default=VideoType.GENERAL,
        description="Type of video editing template to apply"
    )
    
    custom_instructions: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Custom editing instructions from user"
    )
    
    transcription_provider: str = Field(
        default="openai",
        description="Provider to use for transcription (openai, replicate, local)"
    )
    
    compress_audio: bool = Field(
        default=True,
        description="Whether to compress audio for API limits"
    )
    
    target_duration: Optional[int] = Field(
        default=None,
        ge=60,  # Minimum 1 minute
        le=1800,  # Maximum 30 minutes
        description="Target duration for final video in seconds"
    )
    
    compression_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=0.9,
        description="Desired compression ratio (0.1 = 90% reduction, 0.9 = 10% reduction)"
    )
    
    @field_validator("custom_instructions")
    @classmethod
    def validate_custom_instructions(cls, v):
        """Validate custom instructions are not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("Custom instructions cannot be empty")
        return v.strip() if v else None
    
    @field_validator("transcription_provider")
    @classmethod
    def validate_transcription_provider(cls, v):
        """Validate transcription provider is supported."""
        valid_providers = ["openai", "replicate", "local"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        return v


class VideoInfo(BaseModel):
    """
    Information about an uploaded video file.
    
    Attributes:
        filename: Original filename of the video
        file_size: Size of the video file in bytes
        duration: Duration of the video in seconds
        fps: Frames per second
        resolution: Video resolution (width x height)
        format: Video format (mp4, mov, avi, etc.)
        has_audio: Whether the video has audio track
        audio_duration: Duration of audio track in seconds
    """
    filename: str = Field(description="Original filename of the video")
    file_size: int = Field(ge=0, description="Size of the video file in bytes")
    duration: float = Field(ge=0, description="Duration of the video in seconds")
    fps: float = Field(ge=0, description="Frames per second")
    resolution: str = Field(description="Video resolution (width x height)")
    format: str = Field(description="Video format (mp4, mov, avi, etc.)")
    has_audio: bool = Field(description="Whether the video has audio track")
    audio_duration: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration of audio track in seconds"
    )
    
    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        """Validate filename is not empty and has valid extension."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        
        valid_extensions = ['.mp4', '.mov', '.avi', '.mp3', '.wav', '.m4a']
        file_ext = os.path.splitext(v.lower())[1]
        if file_ext not in valid_extensions:
            raise ValueError(f"File extension must be one of: {valid_extensions}")
        
        return v.strip()
    
    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        """Validate resolution format."""
        if not v or 'x' not in v:
            raise ValueError("Resolution must be in format 'width x height'")
        return v
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate video format."""
        valid_formats = ['mp4', 'mov', 'avi', 'mp3', 'wav', 'm4a']
        if v.lower() not in valid_formats:
            raise ValueError(f"Format must be one of: {valid_formats}")
        return v.lower()


class EditingSegment(BaseModel):
    """
    Represents a segment selected for the final edited video.
    
    Attributes:
        id: Unique identifier for the segment
        original_start: Start time in original video (seconds)
        original_end: End time in original video (seconds)
        final_start: Start time in final edited video (seconds)
        final_end: End time in final edited video (seconds)
        function: Function of the segment (gancho, desenvolvimento, etc.)
        priority_score: Priority score for this segment
        text: Transcribed text for this segment
        confidence: Confidence score for this segment selection
    """
    id: str = Field(description="Unique identifier for the segment")
    original_start: float = Field(ge=0, description="Start time in original video (seconds)")
    original_end: float = Field(ge=0, description="End time in original video (seconds)")
    final_start: float = Field(ge=0, description="Start time in final edited video (seconds)")
    final_end: float = Field(ge=0, description="End time in final edited video (seconds)")
    function: str = Field(description="Function of the segment (gancho, desenvolvimento, etc.)")
    priority_score: float = Field(
        ge=0,
        le=10,
        description="Priority score for this segment"
    )
    text: str = Field(description="Transcribed text for this segment")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this segment selection"
    )
    
    @field_validator("original_end")
    @classmethod
    def validate_original_times(cls, v, info):
        """Validate original end time is after start time."""
        if info.data.get('original_start') and v <= info.data['original_start']:
            raise ValueError("Original end time must be after start time")
        return v
    
    @field_validator("final_end")
    @classmethod
    def validate_final_times(cls, v, info):
        """Validate final end time is after start time."""
        if info.data.get('final_start') and v <= info.data['final_start']:
            raise ValueError("Final end time must be after start time")
        return v
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        """Validate text is not empty."""
        if not v or not v.strip():
            raise ValueError("Segment text cannot be empty")
        return v.strip()


class ProcessingResult(BaseModel):
    """
    Result of video processing operation.
    
    Attributes:
        processing_id: Unique identifier for this processing job
        video_filename: Original video filename
        video_info: Information about the processed video
        original_duration: Duration of original video (seconds)
        final_duration: Duration of final edited video (seconds)
        compression_ratio: Actual compression ratio achieved
        segments_found: Number of segments identified
        segments_selected: Number of segments selected for final video
        segments: List of selected editing segments
        xml_path: Path to generated Adobe Premiere XML file
        guide_path: Path to generated cutting guide file
        processing_time: Total processing time (seconds)
        status: Processing status
        created_at: Timestamp when processing was created
        completed_at: Timestamp when processing was completed
    """
    processing_id: str = Field(description="Unique identifier for this processing job")
    video_filename: str = Field(description="Original video filename")
    video_info: VideoInfo = Field(description="Information about the processed video")
    original_duration: float = Field(ge=0, description="Duration of original video (seconds)")
    final_duration: float = Field(ge=0, description="Duration of final edited video (seconds)")
    compression_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Actual compression ratio achieved"
    )
    segments_found: int = Field(ge=0, description="Number of segments identified")
    segments_selected: int = Field(ge=0, description="Number of segments selected for final video")
    segments: List[EditingSegment] = Field(description="List of selected editing segments")
    xml_path: str = Field(description="Path to generated Adobe Premiere XML file")
    guide_path: str = Field(description="Path to generated cutting guide file")
    processing_time: float = Field(ge=0, description="Total processing time (seconds)")
    status: str = Field(description="Processing status")
    created_at: str = Field(description="Timestamp when processing was created")
    completed_at: Optional[str] = Field(
        default=None,
        description="Timestamp when processing was completed"
    )
    
    @field_validator("final_duration")
    @classmethod
    def validate_final_duration(cls, v, info):
        """Validate final duration is reasonable compared to original."""
        if info.data.get('original_duration') and v > info.data['original_duration']:
            raise ValueError("Final duration cannot be longer than original")
        return v
    
    @field_validator("compression_ratio")
    @classmethod
    def validate_compression_ratio(cls, v, info):
        """Validate compression ratio makes sense."""
        if info.data.get('original_duration') and info.data.get('final_duration'):
            expected_ratio = info.data['final_duration'] / info.data['original_duration']
            if abs(v - expected_ratio) > 0.1:  # Allow 10% tolerance
                raise ValueError("Compression ratio doesn't match duration ratio")
        return v
    
    @field_validator("segments_selected")
    @classmethod
    def validate_segments_selected(cls, v, info):
        """Validate segments selected count matches segments list."""
        if info.data.get('segments') and v != len(info.data['segments']):
            raise ValueError("Segments selected count must match segments list length")
        return v
    
    @field_validator("xml_path", "guide_path")
    @classmethod
    def validate_file_paths(cls, v):
        """Validate file paths are not empty."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()


class ProcessingStatus(BaseModel):
    """
    Status of an ongoing processing operation.
    
    Attributes:
        processing_id: Unique identifier for this processing job
        status: Current status (pending, processing, completed, failed)
        progress: Progress percentage (0-100)
        current_step: Current processing step
        message: Status message
        error: Error message if status is failed
        estimated_completion: Estimated completion time
    """
    processing_id: str = Field(description="Unique identifier for this processing job")
    status: str = Field(description="Current status (pending, processing, completed, failed)")
    progress: int = Field(
        ge=0,
        le=100,
        description="Progress percentage (0-100)"
    )
    current_step: str = Field(description="Current processing step")
    message: str = Field(description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if status is failed")
    estimated_completion: Optional[str] = Field(
        default=None,
        description="Estimated completion time"
    )
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status is one of allowed values."""
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v
"""
Editing configuration data models.

This module defines Pydantic models for editing rules and configuration,
supporting different video types and editing strategies.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
import re


class EditingFunction(str, Enum):
    """
    Functions that segments can serve in the final video.
    
    Based on the editing rules from INITIAL.md:
    - GANCHO: Hook to grab attention at the beginning
    - DESENVOLVIMENTO: Development/main content
    - EXEMPLO_PRATICO: Practical example
    - DEFINICAO: Definition or explanation
    - ESTATISTICA: Statistical information
    - TRANSICAO: Transition between topics
    - CONCLUSAO: Conclusion or summary
    - CTA: Call to action
    - GARANTIA: Guarantee (for VSL)
    - PROVA: Social proof or testimonial
    - OFERTA: Offer presentation
    """
    GANCHO = "gancho"
    DESENVOLVIMENTO = "desenvolvimento"
    EXEMPLO_PRATICO = "exemplo_pratico"
    DEFINICAO = "definicao"
    ESTATISTICA = "estatistica"
    TRANSICAO = "transicao"
    CONCLUSAO = "conclusao"
    CTA = "cta"
    GARANTIA = "garantia"
    PROVA = "prova"
    OFERTA = "oferta"


class EditingStrategy(str, Enum):
    """
    Editing strategies for different video types.
    
    COMPRESSION: Focus on removing content (YouTube cuts)
    REORDER: Focus on reordering content (VSL)
    HIGHLIGHT: Focus on highlighting key moments (educational)
    STORYTELLING: Focus on narrative flow (general)
    """
    COMPRESSION = "compression"
    REORDER = "reorder"
    HIGHLIGHT = "highlight"
    STORYTELLING = "storytelling"


class HookPattern(BaseModel):
    """
    Pattern for detecting hooks in content.
    
    Attributes:
        regex: Regular expression pattern to match
        weight: Weight/priority of this pattern
        description: Description of what this pattern matches
        examples: Example phrases that match this pattern
    """
    regex: str = Field(description="Regular expression pattern to match")
    weight: float = Field(
        ge=0.0,
        le=10.0,
        description="Weight/priority of this pattern"
    )
    description: str = Field(description="Description of what this pattern matches")
    examples: List[str] = Field(
        default_factory=list,
        description="Example phrases that match this pattern"
    )
    
    @field_validator("regex")
    @classmethod
    def validate_regex(cls, v):
        """Validate regex pattern compiles correctly."""
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v


class EditingRule(BaseModel):
    """
    Base editing rule configuration.
    
    Attributes:
        name: Name of the rule
        description: Description of what this rule does
        priority: Priority level (1-10)
        enabled: Whether this rule is enabled
        conditions: Conditions that must be met for this rule to apply
        actions: Actions to take when this rule applies
    """
    name: str = Field(description="Name of the rule")
    description: str = Field(description="Description of what this rule does")
    priority: int = Field(
        ge=1,
        le=10,
        description="Priority level (1-10)"
    )
    enabled: bool = Field(default=True, description="Whether this rule is enabled")
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions that must be met for this rule to apply"
    )
    actions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Actions to take when this rule applies"
    )


class SegmentScoringConfig(BaseModel):
    """
    Configuration for segment scoring system.
    
    Attributes:
        hook_weight: Weight for hook detection
        info_weight: Weight for information content
        noise_penalty: Penalty for noise/filler words
        duration_weight: Weight for segment duration
        confidence_weight: Weight for transcription confidence
        min_score: Minimum score for segment selection
        max_score: Maximum possible score
    """
    hook_weight: float = Field(
        default=3.0,
        ge=0.0,
        description="Weight for hook detection"
    )
    info_weight: float = Field(
        default=2.0,
        ge=0.0,
        description="Weight for information content"
    )
    noise_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Penalty for noise/filler words"
    )
    duration_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for segment duration"
    )
    confidence_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for transcription confidence"
    )
    min_score: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum score for segment selection"
    )
    max_score: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum possible score"
    )


class CompressionConfig(BaseModel):
    """
    Configuration for content compression.
    
    Attributes:
        target_ratio: Target compression ratio (0.1-0.9)
        max_pause_ms: Maximum pause duration to keep (milliseconds)
        min_segment_length: Minimum segment length (seconds)
        max_segment_length: Maximum segment length (seconds)
        silence_threshold: Threshold for silence detection
        remove_filler_words: Whether to remove filler words
        filler_words: List of filler words to remove
    """
    target_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=0.9,
        description="Target compression ratio"
    )
    max_pause_ms: int = Field(
        default=800,
        ge=100,
        le=5000,
        description="Maximum pause duration to keep (milliseconds)"
    )
    min_segment_length: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Minimum segment length (seconds)"
    )
    max_segment_length: float = Field(
        default=180.0,
        ge=10.0,
        le=600.0,
        description="Maximum segment length (seconds)"
    )
    silence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for silence detection"
    )
    remove_filler_words: bool = Field(
        default=True,
        description="Whether to remove filler words"
    )
    filler_words: List[str] = Field(
        default_factory=lambda: ["né", "então", "cara", "tipo", "tá", "é", "humm", "ah", "eh"],
        description="List of filler words to remove"
    )


class VideoTypeConfig(BaseModel):
    """
    Configuration for specific video type editing.
    
    Attributes:
        video_type: Type of video this config applies to
        strategy: Primary editing strategy
        suggested_duration_range: Optional suggested duration range (min, max) in seconds
        content_focus: Content focus strategy for AI selection
        compression_config: Compression configuration
        scoring_config: Segment scoring configuration
        hook_patterns: Hook detection patterns
        editing_rules: List of editing rules
        required_functions: Functions that must be present in final video
        function_order: Preferred order of functions
        custom_prompts: Custom prompts for AI analysis
    """
    video_type: str = Field(description="Type of video this config applies to")
    strategy: EditingStrategy = Field(description="Primary editing strategy")
    suggested_duration_range: Optional[tuple] = Field(
        default=None,
        description="Suggested duration range (min, max) in seconds - IA decides based on content"
    )
    content_focus: Optional[str] = Field(
        default="balanced_quality",
        description="Content focus strategy for AI selection"
    )
    compression_config: CompressionConfig = Field(
        default_factory=CompressionConfig,
        description="Compression configuration"
    )
    scoring_config: SegmentScoringConfig = Field(
        default_factory=SegmentScoringConfig,
        description="Segment scoring configuration"
    )
    hook_patterns: List[HookPattern] = Field(
        default_factory=list,
        description="Hook detection patterns"
    )
    editing_rules: List[EditingRule] = Field(
        default_factory=list,
        description="List of editing rules"
    )
    required_functions: List[EditingFunction] = Field(
        default_factory=list,
        description="Functions that must be present in final video"
    )
    function_order: List[EditingFunction] = Field(
        default_factory=list,
        description="Preferred order of functions"
    )
    custom_prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom prompts for AI analysis"
    )
    
    @field_validator("suggested_duration_range")
    @classmethod
    def validate_suggested_duration_range(cls, v):
        """Validate suggested duration range is valid (optional)."""
        if v is None:
            return v
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("Suggested duration range must be a tuple/list of 2 values")
        min_dur, max_dur = v
        if min_dur >= max_dur:
            raise ValueError("Minimum duration must be less than maximum")
        if min_dur < 0:
            raise ValueError("Duration cannot be negative")
        return tuple(v)


class EditingContext(BaseModel):
    """
    Context information for editing decisions.
    
    Attributes:
        video_type: Type of video being edited
        custom_instructions: Custom instructions from user
        original_duration: Duration of original video
        target_duration: Target duration for final video
        transcription_confidence: Overall transcription confidence
        detected_language: Detected language
        speaker_count: Number of speakers detected
        topics: Detected topics/themes
        sentiment: Overall sentiment analysis
        keywords: Important keywords identified
    """
    video_type: str = Field(description="Type of video being edited")
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Custom instructions from user"
    )
    original_duration: float = Field(
        ge=0,
        description="Duration of original video"
    )
    target_duration: Optional[float] = Field(
        default=None,
        ge=0,
        description="Target duration for final video"
    )
    transcription_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall transcription confidence"
    )
    detected_language: Optional[str] = Field(
        default=None,
        description="Detected language"
    )
    speaker_count: int = Field(
        default=1,
        ge=1,
        description="Number of speakers detected"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Detected topics/themes"
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Overall sentiment analysis"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Important keywords identified"
    )


class EditingDecision(BaseModel):
    """
    A decision made during the editing process.
    
    Attributes:
        segment_id: ID of the segment this decision applies to
        decision_type: Type of decision (keep, remove, reorder, modify)
        function: Function assigned to this segment
        score: Score assigned to this segment
        reasoning: Reasoning for this decision
        confidence: Confidence in this decision
        applied_rules: Rules that were applied
        modifications: Any modifications made to the segment
    """
    segment_id: str = Field(description="ID of the segment this decision applies to")
    decision_type: str = Field(description="Type of decision (keep, remove, reorder, modify)")
    function: Optional[EditingFunction] = Field(
        default=None,
        description="Function assigned to this segment"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Score assigned to this segment"
    )
    reasoning: str = Field(description="Reasoning for this decision")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this decision"
    )
    applied_rules: List[str] = Field(
        default_factory=list,
        description="Rules that were applied"
    )
    modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any modifications made to the segment"
    )
    start_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Start time of segment in original video (seconds)"
    )
    end_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="End time of segment in original video (seconds)"
    )
    
    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v):
        """Validate decision type is allowed."""
        valid_types = ["keep", "remove", "reorder", "modify"]
        if v not in valid_types:
            raise ValueError(f"Decision type must be one of: {valid_types}")
        return v


class EditingResult(BaseModel):
    """
    Result of the editing process.
    
    Attributes:
        context: Editing context used
        decisions: List of editing decisions made
        selected_segments: List of selected segment IDs
        final_duration: Final duration of edited video
        compression_achieved: Actual compression ratio achieved
        quality_score: Overall quality score of the edit
        warnings: Any warnings generated during editing
        stats: Statistics about the editing process
    """
    context: EditingContext = Field(description="Editing context used")
    decisions: List[EditingDecision] = Field(
        default_factory=list,
        description="List of editing decisions made"
    )
    selected_segments: List[str] = Field(
        default_factory=list,
        description="List of selected segment IDs"
    )
    final_duration: float = Field(
        ge=0,
        description="Final duration of edited video"
    )
    compression_achieved: float = Field(
        ge=0.0,
        le=1.0,
        description="Actual compression ratio achieved"
    )
    quality_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall quality score of the edit"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during editing"
    )
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics about the editing process"
    )
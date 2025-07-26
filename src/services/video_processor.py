"""
Video Processing Service for AI Editor.

This service orchestrates the complete video processing pipeline:
1. Audio extraction from video files
2. Transcription using various providers
3. AI analysis for content segmentation
4. Guide generation with editing decisions
5. XML generation for Adobe Premiere Pro

Integrates with: TranscriptionService, AnalysisService, GuideGenerator, XMLGenerator
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict
from moviepy.editor import VideoFileClip
from fastapi import WebSocket

from .transcription import TranscriptionService, TranscriptionRequest
from .analysis_service import AnalysisService
from .guide_generator import GuideGenerator
from .xml_generator import XMLGenerator
from .ai_editor import AIEditor
from ..models.video import VideoInfo, EditingSegment, VideoType
from ..models.editing import EditingResult, EditingDecision, EditingContext
from ..models.transcription import TranscriptionSegment
from ..config.settings import get_settings

# Configuração do logging
logger = logging.getLogger(__name__)
settings = get_settings()


class VideoProcessorError(Exception):
    """Custom exception for video processing errors."""
    pass


class VideoProcessor:
    """
    Main video processing service that orchestrates the complete pipeline.
    
    Handles the full workflow from video upload to XML generation,
    providing real-time feedback via WebSocket connections.
    """
    
    def __init__(self, manager, client_id: str, transcription_service: TranscriptionService, temp_dir="temp"):
        """
        Initialize the video processor.
        
        Args:
            manager: WebSocket connection manager
            client_id: Unique client identifier
            transcription_service: Service for audio transcription
            temp_dir: Directory for temporary files
        """
        self.manager = manager
        self.client_id = client_id
        self.transcription_service = transcription_service
        self.analysis_service = AnalysisService()
        self.guide_generator = GuideGenerator()
        self.xml_generator = XMLGenerator()
        self.ai_editor = AIEditor()
        self.temp_dir = temp_dir
        self.settings = get_settings()
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"VideoProcessor initialized for client {client_id}")

    def _extract_audio(self, file_path: str) -> str:
        """
        Extract audio from video or convert audio files to WAV format.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Path to the extracted/converted audio file
            
        Raises:
            VideoProcessorError: If audio extraction fails
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            audio_path = os.path.splitext(file_path)[0] + '.wav'
            
            # Handle audio files directly
            if file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a']:
                logger.info(f"Audio file detected: {file_extension}. Converting to WAV...")
                
                # Use FFmpeg for audio conversion
                import subprocess
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', file_path, '-acodec', 'pcm_s16le', audio_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logger.info("Audio conversion completed successfully")
                
            else:
                # Handle video files - extract audio with multiple fallback strategies
                logger.info(f"Video file detected: {file_extension}. Extracting audio...")
                
                # Strategy 1: Try FFmpeg directly (most robust for problematic files)
                try:
                    logger.info("Trying FFmpeg direct extraction...")
                    import subprocess
                    result = subprocess.run([
                        'ffmpeg', '-y', '-i', file_path, 
                        '-vn',  # No video
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',  # Sample rate
                        '-ac', '1',      # Mono
                        '-ignore_unknown',  # Ignore unknown streams
                        '-err_detect', 'ignore_err',  # Ignore errors
                        audio_path
                    ], 
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300  # 5 minute timeout
                    )
                    logger.info("FFmpeg direct extraction successful")
                    
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning(f"FFmpeg direct failed: {e}. Trying MoviePy...")
                    
                    # Strategy 2: Try MoviePy with error handling
                    try:
                        with VideoFileClip(file_path) as video_clip:
                            if video_clip.audio is None:
                                raise VideoProcessorError("Video file has no audio track")
                            
                            video_clip.audio.write_audiofile(
                                audio_path, 
                                codec='pcm_s16le',
                                verbose=False,
                                logger=None
                            )
                        logger.info("MoviePy extraction successful")
                        
                    except Exception as moviepy_error:
                        logger.warning(f"MoviePy failed: {moviepy_error}. Trying FFmpeg with repair...")
                        
                        # Strategy 3: Try FFmpeg with file repair
                        try:
                            # First, try to repair the file
                            temp_repaired = file_path + '_repaired.mp4'
                            subprocess.run([
                                'ffmpeg', '-y', '-i', file_path,
                                '-c', 'copy',
                                '-avoid_negative_ts', 'make_zero',
                                '-fflags', '+genpts',
                                temp_repaired
                            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            # Then extract audio from repaired file
                            subprocess.run([
                                'ffmpeg', '-y', '-i', temp_repaired,
                                '-vn', '-acodec', 'pcm_s16le',
                                '-ar', '16000', '-ac', '1',
                                audio_path
                            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            # Clean up temp file
                            if os.path.exists(temp_repaired):
                                os.remove(temp_repaired)
                                
                            logger.info("FFmpeg with repair successful")
                            
                        except Exception as repair_error:
                            logger.error(f"All extraction strategies failed. Last error: {repair_error}")
                            raise VideoProcessorError(f"Could not extract audio from video file. File may be corrupted or use unsupported codec.")
                
                logger.info("Audio extraction completed successfully")
            
            if not os.path.exists(audio_path):
                raise VideoProcessorError(f"Audio file was not created: {audio_path}")
                
            return audio_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error during audio processing: {e.stderr}"
            logger.error(error_msg)
            raise VideoProcessorError(error_msg)
        except subprocess.TimeoutExpired as e:
            error_msg = f"FFmpeg timeout during audio processing (file too large or complex)"
            logger.error(error_msg)
            raise VideoProcessorError(error_msg)
        except Exception as e:
            error_msg = f"Error extracting audio from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessorError(error_msg)

    def _extract_video_metadata(self, file_path: str) -> VideoInfo:
        """
        Extract metadata from video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            VideoInfo object with metadata
            
        Raises:
            VideoProcessorError: If metadata extraction fails
        """
        try:
            logger.info(f"Extracting metadata from: {file_path}")
            
            with VideoFileClip(file_path) as video_clip:
                # Extract basic information
                duration = video_clip.duration
                fps = video_clip.fps
                width, height = video_clip.size
                has_audio = video_clip.audio is not None
                audio_duration = video_clip.audio.duration if has_audio else None
                
                # Get file information
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                file_format = os.path.splitext(filename)[1][1:].lower()  # Remove dot and lowercase
                resolution = f"{width}x{height}"
                
                video_info = VideoInfo(
                    filename=filename,
                    file_size=file_size,
                    duration=duration,
                    fps=fps,
                    resolution=resolution,
                    format=file_format,
                    has_audio=has_audio,
                    audio_duration=audio_duration
                )
                
                logger.info(f"Video metadata extracted: {duration:.2f}s, {fps}fps, {width}x{height}")
                return video_info
                
        except Exception as e:
            error_msg = f"Error extracting video metadata: {str(e)}"
            logger.error(error_msg)
            raise VideoProcessorError(error_msg)

    def _time_str_to_seconds(self, time_str: str) -> float:
        """
        Convert time string (MM:SS.mmm or HH:MM:SS.mmm) to seconds.
        
        Args:
            time_str: Time string in format MM:SS.mmm or HH:MM:SS.mmm
            
        Returns:
            Time in seconds as float
        """
        try:
            parts = time_str.split(':')
            
            if len(parts) == 2:
                # MM:SS.mmm format
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                # HH:MM:SS.mmm format
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            else:
                # Assume it's already in seconds
                return float(time_str)
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing time string '{time_str}': {e}")
            return 0.0

    async def _send_status_update(self, message: str, progress: int, **kwargs):
        """
        Send status update via WebSocket.
        
        Args:
            message: Status message
            progress: Progress percentage (0-100)
            **kwargs: Additional data to send
        """
        try:
            await self.manager.send_status_update(
                self.client_id, 
                message, 
                progress, 
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error sending status update: {e}")

    async def process_video(self, file_path: str, video_type: str, instructions: str = "") -> Tuple[str, str]:
        """
        Process video through the complete pipeline.
        
        Args:
            file_path: Path to the video file
            video_type: Type of video (VSL, YouTube, etc.)
            instructions: Additional instructions for processing
            
        Returns:
            Tuple of (guide_path, xml_path)
            
        Raises:
            VideoProcessorError: If processing fails
        """
        try:
            logger.info(f"Starting video processing: {file_path}, type: {video_type}")
            
            # Step 1: Extract video metadata
            await self._send_status_update("Analisando vídeo...", 5)
            video_info = self._extract_video_metadata(file_path)
            
            # Step 2: Extract audio
            await self._send_status_update("Extraindo áudio...", 10)
            audio_path = self._extract_audio(file_path)
            
            # Step 3: Transcribe audio
            await self._send_status_update("Transcrevendo áudio...", 30)
            transcription_result = await self.transcription_service.transcribe_audio(
                TranscriptionRequest(file_path=audio_path)
            )
            
            # Save transcription with video duration
            transcription_path = os.path.splitext(audio_path)[0] + '.json'
            transcription_dict = asdict(transcription_result)
            
            # Add video duration to transcription data for better timestamp estimation
            transcription_dict['duration'] = video_info.duration
            transcription_dict['video_info'] = {
                'filename': video_info.filename,
                'fps': video_info.fps,
                'resolution': video_info.resolution
            }
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(transcription_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Transcription saved to: {transcription_path}")
            
            # Step 4: Generate cutting guide
            await self._send_status_update("Analisando conteúdo com IA...", 60)
            guide_result = self.guide_generator.generate_cutting_guide(
                transcription_path, 
                video_type,
                instructions
            )
            
            # Save guide
            guide_path = os.path.splitext(file_path)[0] + '_guide.json'
            with open(guide_path, 'w', encoding='utf-8') as f:
                json.dump(guide_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cutting guide saved to: {guide_path}")
            
            # Step 5: Convert guide segments to TranscriptionSegments for AI analysis
            await self._send_status_update("Aplicando IA para seleção de segmentos...", 70)
            transcript_segments = self._convert_guide_to_transcript_segments(guide_result)
            
            # Step 6: Use AI Editor to make intelligent decisions
            logger.error(f" CRIANDO EDITING CONTEXT: video_type={video_type}, duration={video_info.duration}, instructions='{instructions}'")
            editing_context = EditingContext(
                video_type=video_type,
                original_duration=video_info.duration,
                custom_instructions=instructions
            )
            logger.error(f" EDITING CONTEXT CRIADO: {editing_context}")
            
            video_type_enum = VideoType.GENERAL
            if video_type.lower() == "vsl":
                video_type_enum = VideoType.VSL
            elif video_type.lower() == "youtube_live":
                video_type_enum = VideoType.YOUTUBE
            elif video_type.lower() == "educational":
                video_type_enum = VideoType.EDUCATIONAL
            
            logger.error(f" CHAMANDO AI EDITOR: {len(transcript_segments)} segmentos, tipo={video_type_enum}")
            editing_result = await self.ai_editor.analyze_and_cut(
                transcript_segments, 
                video_type_enum, 
                editing_context
            )
            logger.error(f" AI EDITOR RETORNOU: {len(editing_result.selected_segments)} segmentos selecionados")
            
            # Step 7: Generate XML
            await self._send_status_update("Gerando arquivo XML...", 90)
            xml_content = self.xml_generator.generate_premiere_xml(
                video_info, 
                editing_result, 
                file_path
            )
            
            # Save XML
            xml_path = os.path.splitext(file_path)[0] + '_AI_Cuts.xml'
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            logger.info(f"XML saved to: {xml_path}")
            
            # Step 8: Send completion status
            guide_url = f"/temp/{os.path.basename(guide_path)}"
            xml_url = f"/temp/{os.path.basename(xml_path)}"
            
            await self._send_status_update(
                "Processamento concluído!", 
                100, 
                guide_url=guide_url, 
                xml_url=xml_url
            )
            
            logger.info(f"Video processing completed successfully for {file_path}")
            return guide_path, xml_path
            
        except Exception as e:
            error_msg = f"Error processing video {file_path}: {str(e)}"
            logger.error(error_msg)
            await self._send_status_update(f"Erro: {str(e)}", -1)
            raise VideoProcessorError(error_msg)

    def _convert_guide_to_transcript_segments(self, guide_data: dict) -> List[TranscriptionSegment]:
        """
        Convert guide data to TranscriptionSegments for AI analysis.
        
        Args:
            guide_data: Guide data from AI analysis
            
        Returns:
            List of TranscriptionSegments
        """
        try:
            # Extract segments from different possible formats
            segments = []
            if 'cortes' in guide_data:
                segments = guide_data['cortes']
            elif 'cortes_identificados' in guide_data:
                segments = guide_data['cortes_identificados']
            elif 'segmentos' in guide_data:
                segments = guide_data['segmentos']
            elif 'vsl_final' in guide_data and 'segmentos' in guide_data['vsl_final']:
                segments = guide_data['vsl_final']['segmentos']
            
            # Create transcription segments
            transcript_segments = []
            
            for i, segment in enumerate(segments):
                # Get time values directly (not strings)
                start_time = segment.get('original_start', segment.get('start', 0.0))
                end_time = segment.get('original_end', segment.get('end', 0.0))
                
                # Convert to float if they are strings
                if isinstance(start_time, str):
                    start_time = self._time_str_to_seconds(start_time)
                if isinstance(end_time, str):
                    end_time = self._time_str_to_seconds(end_time)
                
                if start_time < end_time:  # Valid segment
                    # Create transcription segment
                    transcript_segment = TranscriptionSegment(
                        id=f"segment_{i+1}",
                        start=start_time,
                        end=end_time,
                        text=segment.get('texto', segment.get('text', '')),
                        confidence=0.8,  # Default confidence for guide segments
                        words=None,
                        speaker=None,
                        language="pt"
                    )
                    transcript_segments.append(transcript_segment)
            
            return transcript_segments
            
        except Exception as e:
            logger.error(f"Erro ao converter guia para segmentos de transcrição: {str(e)}")
            return []

    def cleanup_temp_files(self, file_path: str):
        """
        Clean up temporary files created during processing.
        
        Args:
            file_path: Original file path (used to derive temp file names)
        """
        try:
            base_path = os.path.splitext(file_path)[0]
            temp_files = [
                base_path + '.wav',  # Extracted audio
                base_path + '.json'  # Transcription
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temp file: {temp_file}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

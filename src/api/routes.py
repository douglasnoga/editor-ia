"""
FastAPI routes for the video editing automation system.

This module defines all API endpoints for video upload, processing,
status checking, and file download.
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..models.video import VideoType, ProcessingStatus
from ..models.transcription import TranscriptionProvider, TranscriptionRequest
from ..models.editing import EditingContext
from ..services.video_processor import VideoProcessor
from ..services.transcription import TranscriptionService
from ..services.ai_editor import AIEditor
from ..services.xml_generator import XMLGenerator
from ..utils.file_handler import get_file_handler
from ..utils.validators import FileValidator, InputValidator
from ..utils.logging import get_logger, performance_timer, processing_context
from ..config.settings import get_settings

logger = get_logger('api')
settings = get_settings()

# Create router
router = APIRouter()

# Initialize services
video_processor = VideoProcessor()
# transcription_service = TranscriptionService()  # Moved to function to get fresh instance
ai_editor = AIEditor()
xml_generator = XMLGenerator()
file_handler = get_file_handler()

# In-memory storage for processing status (in production, use Redis or database)
processing_jobs: Dict[str, Dict[str, Any]] = {}


class ProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_type: VideoType = VideoType.GENERAL
    custom_instructions: Optional[str] = None
    transcription_provider: TranscriptionProvider = TranscriptionProvider.REPLICATE


async def process_video_pipeline(processing_id: str, video_file: UploadFile, 
                               request: ProcessingRequest) -> None:
    """
    Background task for video processing pipeline.
    
    Args:
        processing_id: Unique processing identifier
        video_file: Uploaded video file
        request: Processing request parameters
    """
    try:
        with processing_context(processing_id):
            # Update status
            processing_jobs[processing_id]['status'] = 'processing'
            processing_jobs[processing_id]['current_step'] = 'Validating file'
            processing_jobs[processing_id]['progress'] = 10
            
            # Validate file
            with performance_timer("file_validation"):
                validation_result = FileValidator.validate_file_upload(video_file)
                if not validation_result['valid']:
                    raise HTTPException(status_code=400, detail=validation_result['errors'])
            
            # Process video and extract audio
            processing_jobs[processing_id]['current_step'] = 'Processing video'
            processing_jobs[processing_id]['progress'] = 20
            
            with performance_timer("video_processing"):
                logger.info(f"Starting upload processing for: {video_file.filename}")
                logger.info(f"File size: {video_file.size:,} bytes")
                logger.info(f"Content type: {video_file.content_type}")
                
                async with file_handler.handle_upload(video_file) as video_path:
                    logger.info(f"File uploaded to: {video_path}")
                    
                    # Verify uploaded file
                    import os
                    if os.path.exists(video_path):
                        actual_size = os.path.getsize(video_path)
                        logger.info(f"Uploaded file size on disk: {actual_size:,} bytes")
                        
                        # Check file header
                        with open(video_path, 'rb') as f:
                            header = f.read(32)
                            logger.info(f"File header: {header[:16].hex()}")
                    else:
                        logger.error(f"Uploaded file not found: {video_path}")
                    
                    audio_path, video_info = await video_processor.process_video_from_path(video_path, video_file.filename)
                    
                    # Update status
                    processing_jobs[processing_id]['video_info'] = video_info
                    processing_jobs[processing_id]['current_step'] = 'Transcribing audio'
                    processing_jobs[processing_id]['progress'] = 40
                    
                    # Transcribe audio
                    with performance_timer("transcription"):
                        # Create fresh TranscriptionService instance to get updated settings
                        transcription_service = TranscriptionService()
                        logger.info(f"🔥 PROVIDER DEBUG: Using transcription provider: {request.transcription_provider}")
                        logger.info(f"🔥 PROVIDER DEBUG: Transcription service replicate model: {transcription_service.settings.replicate_model}")
                        transcription_request = TranscriptionRequest(
                            audio_path=audio_path,
                            provider=request.transcription_provider,
                            enable_word_timestamps=True
                        )
                        
                        transcription_response = await transcription_service.transcribe(transcription_request)
                        
                        if transcription_response.status != "success":
                            raise HTTPException(status_code=500, detail="Transcription failed")
                    
                    # AI content analysis
                    processing_jobs[processing_id]['current_step'] = 'Analyzing content'
                    processing_jobs[processing_id]['progress'] = 60
                    
                    with performance_timer("ai_analysis"):
                        editing_context = EditingContext(
                            video_type=request.video_type.value,
                            custom_instructions=request.custom_instructions,
                            original_duration=video_info.duration,
                            transcription_confidence=0.8,
                            detected_language=transcription_response.language
                        )
                        
                        editing_result = await ai_editor.analyze_and_cut(
                            transcription_response.segments,
                            request.video_type,
                            editing_context
                        )
                    
                    # Generate XML
                    processing_jobs[processing_id]['current_step'] = 'Generating XML'
                    processing_jobs[processing_id]['progress'] = 80
                    
                    with performance_timer("xml_generation"):
                        # Generate Adobe Premiere XML
                        xml_content = xml_generator.generate_premiere_xml(
                            video_info,
                            editing_result,
                            video_path
                        )
                        
                        # Generate cutting guide
                        guide_content = xml_generator.generate_cutting_guide(
                            video_info,
                            editing_result
                        )
                        
                        # Save output files
                        xml_path = await file_handler.save_output_file(
                            xml_content,
                            f"{video_info.filename}_edited.xml",
                            processing_id
                        )
                        
                        guide_path = await file_handler.save_output_file(
                            guide_content,
                            f"{video_info.filename}_guide.txt",
                            processing_id
                        )
            
            # Update final status
            processing_jobs[processing_id].update({
                'status': 'completed',
                'current_step': 'Completed',
                'progress': 100,
                'xml_path': xml_path,
                'guide_path': guide_path,
                'editing_result': editing_result,
                'completed_at': datetime.now().isoformat()
            })
            
            logger.info(f"Processing completed for {processing_id}")
            
    except Exception as e:
        logger.error(f"Processing failed for {processing_id}: {str(e)}")
        processing_jobs[processing_id].update({
            'status': 'failed',
            'error': str(e),
            'progress': 0
        })


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    video_type: VideoType = Form(VideoType.GENERAL),
    custom_instructions: Optional[str] = Form(None),
    transcription_provider: TranscriptionProvider = Form(TranscriptionProvider.REPLICATE)
):
    """
    Upload video file and start processing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded video file
        video_type: Type of video editing template
        custom_instructions: Custom editing instructions
        transcription_provider: Transcription provider to use
        
    Returns:
        Processing ID and initial status
    """
    try:
        # Generate processing ID
        processing_id = str(uuid.uuid4())
        
        # Validate inputs
        video_type_validation = InputValidator.validate_video_type(video_type.value)
        if not video_type_validation['valid']:
            raise HTTPException(status_code=400, detail=video_type_validation['errors'])
        
        if custom_instructions:
            instructions_validation = InputValidator.validate_custom_instructions(custom_instructions)
            if not instructions_validation['valid']:
                raise HTTPException(status_code=400, detail=instructions_validation['errors'])
        
        # Create processing request
        request = ProcessingRequest(
            video_type=video_type,
            custom_instructions=custom_instructions,
            transcription_provider=transcription_provider
        )
        
        # Initialize processing status
        processing_jobs[processing_id] = {
            'processing_id': processing_id,
            'status': 'pending',
            'progress': 0,
            'current_step': 'Initializing',
            'filename': file.filename,
            'created_at': datetime.now().isoformat(),
            'request': request
        }
        
        # Start background processing
        background_tasks.add_task(process_video_pipeline, processing_id, file, request)
        
        return {
            "processing_id": processing_id,
            "status": "pending",
            "message": "Video upload started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status/{processing_id}")
async def get_processing_status(processing_id: str):
    """
    Get processing status for a job.
    
    Args:
        processing_id: Processing job identifier
        
    Returns:
        Processing status information
    """
    try:
        if processing_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        job = processing_jobs[processing_id]
        
        # Create status response
        status = ProcessingStatus(
            processing_id=processing_id,
            status=job['status'],
            progress=job['progress'],
            current_step=job['current_step'],
            message=job.get('message', ''),
            error=job.get('error'),
            estimated_completion=job.get('estimated_completion')
        )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/download/{processing_id}/xml")
async def download_xml(processing_id: str):
    """
    Download generated XML file.
    
    Args:
        processing_id: Processing job identifier
        
    Returns:
        XML file download
    """
    try:
        if processing_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        job = processing_jobs[processing_id]
        
        if job['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Processing not completed")
        
        xml_path = job.get('xml_path')
        if not xml_path or not Path(xml_path).exists():
            raise HTTPException(status_code=404, detail="XML file not found")
        
        return FileResponse(
            path=xml_path,
            media_type='application/xml',
            filename=f"{job['filename']}_edited.xml"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"XML download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"XML download failed: {str(e)}")


@router.get("/download/{processing_id}/guide")
async def download_guide(processing_id: str):
    """
    Download cutting guide file.
    
    Args:
        processing_id: Processing job identifier
        
    Returns:
        Guide file download
    """
    try:
        if processing_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        job = processing_jobs[processing_id]
        
        if job['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Processing not completed")
        
        guide_path = job.get('guide_path')
        if not guide_path or not Path(guide_path).exists():
            raise HTTPException(status_code=404, detail="Guide file not found")
        
        return FileResponse(
            path=guide_path,
            media_type='text/plain',
            filename=f"{job['filename']}_guide.txt"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guide download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Guide download failed: {str(e)}")


@router.get("/jobs")
async def list_processing_jobs():
    """
    List all processing jobs.
    
    Returns:
        List of processing jobs
    """
    try:
        jobs = []
        for job_id, job_data in processing_jobs.items():
            jobs.append({
                'processing_id': job_id,
                'status': job_data['status'],
                'progress': job_data['progress'],
                'filename': job_data['filename'],
                'created_at': job_data['created_at']
            })
        
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Job listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")


@router.delete("/jobs/{processing_id}")
async def delete_processing_job(processing_id: str):
    """
    Delete a processing job and its files.
    
    Args:
        processing_id: Processing job identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        if processing_id not in processing_jobs:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        job = processing_jobs[processing_id]
        
        # Clean up files
        for file_key in ['xml_path', 'guide_path']:
            if file_key in job:
                file_path = Path(job[file_key])
                if file_path.exists():
                    file_path.unlink()
        
        # Remove from memory
        del processing_jobs[processing_id]
        
        return {"message": "Processing job deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job deletion failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    try:
        # Check services
        transcription_status = await transcription_service.get_provider_status()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "transcription": transcription_status,
                "video_processor": {"available": True},
                "ai_editor": {"available": bool(settings.openai_api_key)},
                "xml_generator": {"available": True}
            },
            "active_jobs": len(processing_jobs),
            "disk_usage": file_handler.get_disk_usage()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/formats")
async def get_supported_formats():
    """
    Get supported video formats.
    
    Returns:
        List of supported formats
    """
    try:
        formats = video_processor.get_supported_formats()
        return formats
        
    except Exception as e:
        logger.error(f"Format listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Format listing failed: {str(e)}")


@router.post("/preview")
async def get_editing_preview(
    file: UploadFile = File(...),
    video_type: VideoType = Form(VideoType.GENERAL)
):
    """
    Get editing preview without full processing.
    
    Args:
        file: Uploaded video file
        video_type: Type of video editing template
        
    Returns:
        Editing preview information
    """
    try:
        # Quick validation
        validation_result = FileValidator.validate_file_upload(file)
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['errors'])
        
        # Get basic video info
        async with file_handler.handle_upload(file) as video_path:
            video_info = await video_processor._extract_video_info(video_path, file.filename)
            
            # Get editing preview
            preview = await ai_editor.get_editing_preview([], video_type)
            preview['video_info'] = video_info
            
            return preview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
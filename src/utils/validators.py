"""
Input validation utilities.

This module provides comprehensive validation for video types, custom instructions,
file formats, API keys, and other input parameters.
"""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from fastapi import UploadFile

from ..models.video import VideoType
from ..models.transcription import TranscriptionProvider
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class FileValidator:
    """
    File validation utilities.
    
    Provides comprehensive validation for uploaded files including
    format checking, size limits, and security validation.
    """
    
    # Supported file extensions and MIME types
    SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi']
    SUPPORTED_AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a']
    SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS + SUPPORTED_AUDIO_EXTENSIONS
    
    SUPPORTED_MIME_TYPES = [
        'video/mp4',
        'video/quicktime',
        'video/x-msvideo',
        'audio/mpeg',
        'audio/wav',
        'audio/mp4',
        'audio/x-m4a'
    ]
    
    # Dangerous file extensions (security)
    DANGEROUS_EXTENSIONS = [
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
        '.vbs', '.js', '.jar', '.app', '.deb', '.rpm',
        '.dmg', '.pkg', '.msi', '.ps1', '.sh'
    ]
    
    @classmethod
    def validate_file_upload(cls, file: UploadFile) -> Dict[str, Any]:
        """
        Comprehensive validation for uploaded files.
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            result = {
                'valid': True,
                'filename': file.filename,
                'size': file.size,
                'content_type': file.content_type,
                'warnings': [],
                'errors': []
            }
            
            # Check if filename exists
            if not file.filename:
                result['errors'].append("No filename provided")
                result['valid'] = False
                return result
            
            # Validate filename
            filename_result = cls.validate_filename(file.filename)
            if not filename_result['valid']:
                result['errors'].extend(filename_result['errors'])
                result['valid'] = False
            
            # Validate file size
            size_result = cls.validate_file_size(file.size)
            if not size_result['valid']:
                result['errors'].extend(size_result['errors'])
                result['valid'] = False
            
            # Validate MIME type
            mime_result = cls.validate_mime_type(file.content_type)
            if not mime_result['valid']:
                result['errors'].extend(mime_result['errors'])
                result['valid'] = False
            
            # Add warnings
            result['warnings'].extend(filename_result.get('warnings', []))
            result['warnings'].extend(size_result.get('warnings', []))
            result['warnings'].extend(mime_result.get('warnings', []))
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating file upload: {str(e)}")
            raise ValidationError(f"File validation failed: {str(e)}")
    
    @classmethod
    def validate_filename(cls, filename: str) -> Dict[str, Any]:
        """
        Validate filename for security and format compliance.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check for empty filename
            if not filename or not filename.strip():
                result['errors'].append("Filename cannot be empty")
                result['valid'] = False
                return result
            
            # Check filename length
            if len(filename) > 255:
                result['errors'].append("Filename too long (max 255 characters)")
                result['valid'] = False
            
            # Check for dangerous characters
            dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
            for char in dangerous_chars:
                if char in filename:
                    result['errors'].append(f"Filename contains dangerous character: {char}")
                    result['valid'] = False
            
            # Check for dangerous extensions
            file_ext = Path(filename).suffix.lower()
            if file_ext in cls.DANGEROUS_EXTENSIONS:
                result['errors'].append(f"Dangerous file extension: {file_ext}")
                result['valid'] = False
            
            # Check for supported extensions
            if file_ext not in cls.SUPPORTED_EXTENSIONS:
                result['errors'].append(f"Unsupported file extension: {file_ext}")
                result['valid'] = False
            
            # Check for hidden files (starting with .)
            if filename.startswith('.'):
                result['warnings'].append("Hidden file detected")
            
            # Check for common problematic patterns
            if '..' in filename:
                result['errors'].append("Filename contains directory traversal pattern")
                result['valid'] = False
            
            # Check for spaces at beginning/end
            if filename != filename.strip():
                result['warnings'].append("Filename has leading/trailing spaces")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating filename: {str(e)}")
            result['errors'].append(f"Filename validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_file_size(cls, size: Optional[int]) -> Dict[str, Any]:
        """
        Validate file size against limits.
        
        Args:
            size: File size in bytes
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if size is provided
            if size is None:
                result['errors'].append("File size not provided")
                result['valid'] = False
                return result
            
            # Check for zero size
            if size == 0:
                result['errors'].append("File is empty")
                result['valid'] = False
                return result
            
            # Check maximum size
            if size > settings.max_file_size:
                max_size_mb = settings.max_file_size / (1024 * 1024)
                current_size_mb = size / (1024 * 1024)
                result['errors'].append(
                    f"File too large: {current_size_mb:.1f}MB (max: {max_size_mb:.1f}MB)"
                )
                result['valid'] = False
            
            # Warn about large files
            if size > 100 * 1024 * 1024:  # 100MB
                size_mb = size / (1024 * 1024)
                result['warnings'].append(f"Large file: {size_mb:.1f}MB")
            
            # Warn about OpenAI API limits
            if size > settings.openai_file_limit:
                limit_mb = settings.openai_file_limit / (1024 * 1024)
                result['warnings'].append(
                    f"File exceeds OpenAI API limit ({limit_mb:.1f}MB), will use alternative provider"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating file size: {str(e)}")
            result['errors'].append(f"File size validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_mime_type(cls, mime_type: Optional[str]) -> Dict[str, Any]:
        """
        Validate MIME type for security and format compliance.
        
        Args:
            mime_type: MIME type to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if MIME type is provided
            if not mime_type:
                result['warnings'].append("No MIME type provided")
                return result
            
            # Check for supported MIME types
            if mime_type not in cls.SUPPORTED_MIME_TYPES:
                result['errors'].append(f"Unsupported MIME type: {mime_type}")
                result['valid'] = False
            
            # Check for dangerous MIME types
            dangerous_mimes = [
                'application/x-executable',
                'application/x-msdownload',
                'application/x-msdos-program',
                'application/x-bat',
                'text/x-script'
            ]
            
            if mime_type in dangerous_mimes:
                result['errors'].append(f"Dangerous MIME type: {mime_type}")
                result['valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating MIME type: {str(e)}")
            result['errors'].append(f"MIME type validation error: {str(e)}")
            result['valid'] = False
            return result


class InputValidator:
    """
    Input validation for various application parameters.
    
    Provides validation for video types, custom instructions,
    processing parameters, and other user inputs.
    """
    
    @classmethod
    def validate_video_type(cls, video_type: str) -> Dict[str, Any]:
        """
        Validate video type parameter.
        
        Args:
            video_type: Video type string
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if video type is provided
            if not video_type:
                result['errors'].append("Video type is required")
                result['valid'] = False
                return result
            
            # Check if video type is valid
            valid_types = [vt.value for vt in VideoType]
            if video_type not in valid_types:
                result['errors'].append(f"Invalid video type: {video_type}")
                result['errors'].append(f"Valid types: {', '.join(valid_types)}")
                result['valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating video type: {str(e)}")
            result['errors'].append(f"Video type validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_custom_instructions(cls, instructions: Optional[str]) -> Dict[str, Any]:
        """
        Validate custom instructions parameter.
        
        Args:
            instructions: Custom instructions string
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Instructions are optional
            if not instructions:
                return result
            
            # Check length
            if len(instructions) > 2000:
                result['errors'].append("Custom instructions too long (max 2000 characters)")
                result['valid'] = False
            
            # Check for empty after stripping
            if not instructions.strip():
                result['errors'].append("Custom instructions cannot be empty")
                result['valid'] = False
            
            # Check for potentially dangerous content
            dangerous_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'data:.*base64',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, instructions, re.IGNORECASE):
                    result['errors'].append("Custom instructions contain potentially dangerous content")
                    result['valid'] = False
                    break
            
            # Warn about very long instructions
            if len(instructions) > 1000:
                result['warnings'].append("Long custom instructions may affect AI processing")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating custom instructions: {str(e)}")
            result['errors'].append(f"Custom instructions validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_transcription_provider(cls, provider: str) -> Dict[str, Any]:
        """
        Validate transcription provider parameter.
        
        Args:
            provider: Transcription provider string
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if provider is provided
            if not provider:
                result['errors'].append("Transcription provider is required")
                result['valid'] = False
                return result
            
            # Check if provider is valid
            valid_providers = [tp.value for tp in TranscriptionProvider]
            if provider not in valid_providers:
                result['errors'].append(f"Invalid transcription provider: {provider}")
                result['errors'].append(f"Valid providers: {', '.join(valid_providers)}")
                result['valid'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating transcription provider: {str(e)}")
            result['errors'].append(f"Transcription provider validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_compression_ratio(cls, ratio: float) -> Dict[str, Any]:
        """
        Validate compression ratio parameter.
        
        Args:
            ratio: Compression ratio (0.1 to 0.9)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check range
            if ratio < 0.1 or ratio > 0.9:
                result['errors'].append("Compression ratio must be between 0.1 and 0.9")
                result['valid'] = False
            
            # Warn about extreme values
            if ratio < 0.2:
                result['warnings'].append("Very aggressive compression may affect quality")
            elif ratio > 0.8:
                result['warnings'].append("Low compression may result in long videos")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating compression ratio: {str(e)}")
            result['errors'].append(f"Compression ratio validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_target_duration(cls, duration: Optional[int]) -> Dict[str, Any]:
        """
        Validate target duration parameter.
        
        Args:
            duration: Target duration in seconds
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Duration is optional
            if duration is None:
                return result
            
            # Validação flexível - IA decide a duração baseada no conteúdo
            # Apenas limite máximo para evitar processamento excessivo
            if duration > 3600:  # 1 hora
                result['errors'].append("Target duration cannot exceed 1 hour")
                result['valid'] = False
            
            # Informativo apenas - sem erro
            if duration < 30:
                result['warnings'].append("Very short duration - ensure content is meaningful")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating target duration: {str(e)}")
            result['errors'].append(f"Target duration validation error: {str(e)}")
            result['valid'] = False
            return result


class APIKeyValidator:
    """
    API key validation utilities.
    
    Provides validation for various API keys used by the application.
    """
    
    @classmethod
    def validate_openai_api_key(cls, api_key: Optional[str]) -> Dict[str, Any]:
        """
        Validate OpenAI API key format.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # API key is optional
            if not api_key:
                return result
            
            # Check format (starts with sk-)
            if not api_key.startswith('sk-'):
                result['errors'].append("OpenAI API key must start with 'sk-'")
                result['valid'] = False
            
            # Check length (OpenAI keys are typically 51 characters)
            if len(api_key) != 51:
                result['warnings'].append("OpenAI API key length is unusual")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating OpenAI API key: {str(e)}")
            result['errors'].append(f"OpenAI API key validation error: {str(e)}")
            result['valid'] = False
            return result
    
    @classmethod
    def validate_replicate_api_token(cls, api_token: Optional[str]) -> Dict[str, Any]:
        """
        Validate Replicate API token format.
        
        Args:
            api_token: Replicate API token
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # API token is optional
            if not api_token:
                return result
            
            # Check format (starts with r8_)
            if not api_token.startswith('r8_'):
                result['errors'].append("Replicate API token must start with 'r8_'")
                result['valid'] = False
            
            # Check length (Replicate tokens are typically 40 characters)
            if len(api_token) != 40:
                result['warnings'].append("Replicate API token length is unusual")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating Replicate API token: {str(e)}")
            result['errors'].append(f"Replicate API token validation error: {str(e)}")
            result['valid'] = False
            return result


class ValidationSummary:
    """
    Validation summary utilities.
    
    Provides methods to combine and summarize validation results.
    """
    
    @classmethod
    def combine_results(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple validation results into a single summary.
        
        Args:
            results: List of validation result dictionaries
            
        Returns:
            Combined validation summary
        """
        combined = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'details': results
        }
        
        try:
            for result in results:
                if not result.get('valid', True):
                    combined['valid'] = False
                
                combined['errors'].extend(result.get('errors', []))
                combined['warnings'].extend(result.get('warnings', []))
            
            # Remove duplicates
            combined['errors'] = list(set(combined['errors']))
            combined['warnings'] = list(set(combined['warnings']))
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining validation results: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation combination error: {str(e)}"],
                'warnings': [],
                'details': []
            }
    
    @classmethod
    def format_error_message(cls, result: Dict[str, Any]) -> str:
        """
        Format validation result into a user-friendly error message.
        
        Args:
            result: Validation result dictionary
            
        Returns:
            Formatted error message
        """
        if result.get('valid', True):
            return "Validation successful"
        
        errors = result.get('errors', [])
        warnings = result.get('warnings', [])
        
        message_parts = []
        
        if errors:
            message_parts.append("Errors:")
            for error in errors:
                message_parts.append(f"  - {error}")
        
        if warnings:
            message_parts.append("Warnings:")
            for warning in warnings:
                message_parts.append(f"  - {warning}")
        
        return "\n".join(message_parts)
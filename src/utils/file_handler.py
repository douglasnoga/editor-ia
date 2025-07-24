"""
File handling utilities for secure file operations.

This module provides utilities for handling file uploads, temporary files,
and secure file operations with proper cleanup and validation.
"""

import shutil
import hashlib
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from pathlib import Path
from contextlib import asynccontextmanager
import aiofiles
import aiofiles.os
from fastapi import UploadFile

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FileHandlerError(Exception):
    """Custom exception for file handling errors."""
    pass


class SecureFileHandler:
    """
    Secure file handler with automatic cleanup and validation.
    
    Provides secure file upload handling, temporary file management,
    and automatic cleanup using context managers.
    """
    
    def __init__(self):
        """Initialize the file handler."""
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)
        self.temp_dir = Path(self.settings.temp_dir)
        self.output_dir = Path(self.settings.output_dir)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track temporary files for cleanup
        self._temp_files: List[str] = []
    
    @asynccontextmanager
    async def handle_upload(self, upload_file: UploadFile) -> AsyncGenerator[str, None]:
        """
        Context manager for handling file uploads with automatic cleanup.
        
        Args:
            upload_file: Uploaded file from FastAPI
            
        Yields:
            Path to the saved file
            
        Raises:
            FileHandlerError: If file handling fails
        """
        temp_path = None
        try:
            # Generate secure filename
            filename = await self._generate_secure_filename(upload_file.filename)
            temp_path = self.temp_dir / filename
            
            # Save uploaded file
            await self._save_uploaded_file(upload_file, temp_path)
            
            # Add to cleanup list
            self._temp_files.append(str(temp_path))
            
            yield str(temp_path)
            
        except Exception as e:
            logger.error(f"Error handling upload: {str(e)}")
            raise FileHandlerError(f"Failed to handle upload: {str(e)}")
        finally:
            # Cleanup on exit
            if temp_path and temp_path.exists():
                await self._cleanup_file(temp_path)
    
    @asynccontextmanager
    async def temporary_file(self, suffix: str = "", prefix: str = "temp_") -> AsyncGenerator[str, None]:
        """
        Context manager for creating temporary files with automatic cleanup.
        
        Args:
            suffix: File extension (e.g., ".wav", ".mp4")
            prefix: Filename prefix
            
        Yields:
            Path to the temporary file
        """
        temp_path = None
        try:
            # Generate temporary filename
            temp_filename = f"{prefix}{uuid.uuid4()}{suffix}"
            temp_path = self.temp_dir / temp_filename
            
            # Add to cleanup list
            self._temp_files.append(str(temp_path))
            
            yield str(temp_path)
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {str(e)}")
            raise FileHandlerError(f"Failed to create temporary file: {str(e)}")
        finally:
            # Cleanup on exit
            if temp_path and temp_path.exists():
                await self._cleanup_file(temp_path)
    
    async def save_output_file(self, content: str, filename: str, processing_id: str) -> str:
        """
        Save output file (XML, guide, etc.) to output directory.
        
        Args:
            content: File content to save
            filename: Output filename
            processing_id: Processing ID for organization
            
        Returns:
            Path to saved file
        """
        try:
            # Create processing-specific directory
            processing_dir = self.output_dir / processing_id
            processing_dir.mkdir(exist_ok=True)
            
            # Generate output path
            output_path = processing_dir / filename
            
            # Save content
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.info(f"Saved output file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving output file: {str(e)}")
            raise FileHandlerError(f"Failed to save output file: {str(e)}")
    
    async def save_binary_file(self, content: bytes, filename: str, processing_id: str) -> str:
        """
        Save binary file to output directory.
        
        Args:
            content: Binary content to save
            filename: Output filename
            processing_id: Processing ID for organization
            
        Returns:
            Path to saved file
        """
        try:
            # Create processing-specific directory
            processing_dir = self.output_dir / processing_id
            processing_dir.mkdir(exist_ok=True)
            
            # Generate output path
            output_path = processing_dir / filename
            
            # Save content
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"Saved binary file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving binary file: {str(e)}")
            raise FileHandlerError(f"Failed to save binary file: {str(e)}")
    
    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """
        Copy file from source to destination.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
        """
        try:
            # Create destination directory if needed
            dest_dir = Path(dest_path).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            await asyncio.to_thread(shutil.copy2, source_path, dest_path)
            
            logger.debug(f"Copied file: {source_path} -> {dest_path}")
            
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            raise FileHandlerError(f"Failed to copy file: {str(e)}")
    
    async def move_file(self, source_path: str, dest_path: str) -> None:
        """
        Move file from source to destination.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
        """
        try:
            # Create destination directory if needed
            dest_dir = Path(dest_path).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            await asyncio.to_thread(shutil.move, source_path, dest_path)
            
            logger.debug(f"Moved file: {source_path} -> {dest_path}")
            
        except Exception as e:
            logger.error(f"Error moving file: {str(e)}")
            raise FileHandlerError(f"Failed to move file: {str(e)}")
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileHandlerError(f"File not found: {file_path}")
            
            stat = await aiofiles.os.stat(file_path)
            
            return {
                'path': str(path),
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'is_file': path.is_file(),
                'is_dir': path.is_dir()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise FileHandlerError(f"Failed to get file info: {str(e)}")
    
    async def calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc.)
            
        Returns:
            File hash as hex string
        """
        try:
            hash_func = hashlib.new(algorithm)
            
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            raise FileHandlerError(f"Failed to calculate file hash: {str(e)}")
    
    async def list_files(self, directory: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files in directory with optional pattern matching.
        
        Args:
            directory: Directory to list
            pattern: Glob pattern for filtering
            
        Returns:
            List of file information dictionaries
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                raise FileHandlerError(f"Directory not found: {directory}")
            
            files = []
            for file_path in dir_path.glob(pattern):
                if file_path.is_file():
                    file_info = await self.get_file_info(str(file_path))
                    files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise FileHandlerError(f"Failed to list files: {str(e)}")
    
    async def cleanup_old_files(self, directory: str, max_age_hours: int = 24) -> int:
        """
        Cleanup old files in directory.
        
        Args:
            directory: Directory to cleanup
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        try:
            import time
            
            dir_path = Path(directory)
            if not dir_path.exists():
                return 0
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleanup_count = 0
            
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        await self._cleanup_file(file_path)
                        cleanup_count += 1
            
            logger.info(f"Cleaned up {cleanup_count} old files from {directory}")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}")
            return 0
    
    async def _save_uploaded_file(self, upload_file: UploadFile, dest_path: Path) -> None:
        """
        Save uploaded file to destination path.
        
        Args:
            upload_file: Uploaded file
            dest_path: Destination path
        """
        try:
            # Reset file pointer
            await upload_file.seek(0)
            
            # Save file with detailed logging
            total_written = 0
            async with aiofiles.open(dest_path, 'wb') as f:
                while chunk := await upload_file.read(8192):
                    await f.write(chunk)
                    total_written += len(chunk)
            
            # Verify file was saved correctly
            final_size = dest_path.stat().st_size
            logger.info(f"Uploaded file saved: {upload_file.filename} -> {dest_path}")
            logger.info(f"Upload details: {total_written:,} bytes written, {final_size:,} bytes on disk")
            
            if final_size == 0:
                raise FileHandlerError("Uploaded file is empty")
            
            if final_size != total_written:
                logger.warning(f"Size mismatch: written={total_written:,}, saved={final_size:,}")
            
            # Verify file integrity by checking header
            with open(dest_path, 'rb') as f:
                header = f.read(32)
                if len(header) < 4:
                    raise FileHandlerError("Uploaded file is too small or corrupted")
                
                logger.info(f"File header: {header[:16].hex()}")
                
                # Check for common video file signatures
                if not (header.startswith(b'\x00\x00\x00') or  # MP4
                        header.startswith(b'ftypmp4') or     # MP4
                        header.startswith(b'RIFF') or        # AVI
                        header.startswith(b'ID3') or         # MP3
                        header.startswith(b'fLaC')):         # FLAC
                    logger.warning(f"Unexpected file header - file may be corrupted")
                else:
                    logger.info("File header validation passed")
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise FileHandlerError(f"Failed to save uploaded file: {str(e)}")
    
    async def _generate_secure_filename(self, original_filename: Optional[str]) -> str:
        """
        Generate secure filename with timestamp and UUID.
        
        Args:
            original_filename: Original filename from upload
            
        Returns:
            Secure filename
        """
        if not original_filename:
            original_filename = "upload"
        
        # Extract extension
        path = Path(original_filename)
        extension = path.suffix
        
        # Generate secure filename
        timestamp = int(asyncio.get_event_loop().time())
        unique_id = str(uuid.uuid4())[:8]
        
        return f"{timestamp}_{unique_id}{extension}"
    
    async def _cleanup_file(self, file_path: Path) -> None:
        """
        Clean up a single file.
        
        Args:
            file_path: Path to file to cleanup
        """
        try:
            if file_path.exists():
                await aiofiles.os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
                
                # Remove from tracking list
                file_str = str(file_path)
                if file_str in self._temp_files:
                    self._temp_files.remove(file_str)
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")
    
    async def cleanup_all_temp_files(self) -> None:
        """
        Clean up all tracked temporary files.
        """
        cleanup_count = 0
        for file_path in self._temp_files.copy():
            try:
                if Path(file_path).exists():
                    await aiofiles.os.remove(file_path)
                    cleanup_count += 1
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")
        
        # Clear tracking list
        self._temp_files.clear()
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} temporary files")
    
    def get_disk_usage(self) -> Dict[str, Dict[str, Any]]:
        """
        Get disk usage statistics for all directories.
        
        Returns:
            Dictionary with disk usage information
        """
        try:
            usage = {}
            
            for name, path in [
                ('upload', self.upload_dir),
                ('temp', self.temp_dir),
                ('output', self.output_dir)
            ]:
                if path.exists():
                    disk_usage = shutil.disk_usage(path)
                    
                    # Calculate directory size
                    dir_size = sum(
                        f.stat().st_size for f in path.rglob('*') if f.is_file()
                    )
                    
                    usage[name] = {
                        'path': str(path),
                        'total_space': disk_usage.total,
                        'used_space': disk_usage.used,
                        'free_space': disk_usage.free,
                        'directory_size': dir_size,
                        'usage_percent': (disk_usage.used / disk_usage.total) * 100
                    }
            
            return usage
            
        except Exception as e:
            logger.error(f"Error getting disk usage: {str(e)}")
            return {}


# Global file handler instance
file_handler = SecureFileHandler()


def get_file_handler() -> SecureFileHandler:
    """
    Get the global file handler instance.
    
    Returns:
        SecureFileHandler instance
    """
    return file_handler
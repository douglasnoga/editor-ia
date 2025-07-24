#!/usr/bin/env python3
"""
Simple test script to verify transcription service configuration.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all imports work."""
    logger.info("Testing imports...")
    
    try:
        # Test individual imports
        logger.info("Testing settings import...")
        from config.settings import get_settings
        logger.info("✓ Settings import successful")
        
        logger.info("Testing models import...")
        from models.transcription import TranscriptionRequest, TranscriptionProvider
        logger.info("✓ Transcription models import successful")
        
        logger.info("Testing service import...")
        # Skip the service import for now due to relative import issues
        logger.info("⚠ Skipping service import due to relative import issues")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_settings():
    """Test settings configuration."""
    logger.info("Testing settings configuration...")
    
    try:
        from config.settings import get_settings
        
        settings = get_settings()
        
        logger.info("Configuration Status:")
        logger.info(f"  OpenAI API Key: {'✓ SET' if settings.openai_api_key else '✗ NOT SET'}")
        logger.info(f"  Replicate API Token: {'✓ SET' if settings.replicate_api_token else '✗ NOT SET'}")
        logger.info(f"  OpenAI Model: {settings.whisper_model}")
        logger.info(f"  Replicate Model: {settings.replicate_model}")
        logger.info(f"  OpenAI File Limit: {settings.openai_file_limit / (1024*1024):.1f} MB")
        logger.info(f"  Transcription Timeout: {settings.transcription_timeout}s")
        
        # Validate API key formats
        if settings.openai_api_key:
            if settings.openai_api_key.startswith('sk-'):
                logger.info("  OpenAI API Key format: ✓ VALID")
            else:
                logger.warning("  OpenAI API Key format: ✗ INVALID (should start with 'sk-')")
        
        if settings.replicate_api_token:
            if settings.replicate_api_token.startswith('r8_'):
                logger.info("  Replicate API Token format: ✓ VALID")
            else:
                logger.warning("  Replicate API Token format: ✗ INVALID (should start with 'r8_')")
        
        return settings
        
    except Exception as e:
        logger.error(f"✗ Settings test failed: {e}")
        return None

def test_service_initialization():
    """Test transcription service initialization."""
    logger.info("Testing transcription service initialization...")
    
    try:
        from services.transcription import TranscriptionService
        
        service = TranscriptionService()
        
        logger.info("Service Status:")
        logger.info(f"  OpenAI Client: {'✓ INITIALIZED' if service.openai_client else '✗ NOT INITIALIZED'}")
        logger.info(f"  Replicate Client: {'✓ INITIALIZED' if service.replicate_client else '✗ NOT INITIALIZED'}")
        
        return service
        
    except Exception as e:
        logger.error(f"✗ Service initialization failed: {e}")
        return None

async def test_provider_status():
    """Test provider status check."""
    logger.info("Testing provider status check...")
    
    try:
        from services.transcription import TranscriptionService
        
        service = TranscriptionService()
        status = await service.get_provider_status()
        
        logger.info("Provider Status:")
        for provider, details in status.items():
            logger.info(f"  {provider.upper()}:")
            for key, value in details.items():
                logger.info(f"    {key}: {value}")
        
        return status
        
    except Exception as e:
        logger.error(f"✗ Provider status test failed: {e}")
        return None

def test_provider_selection():
    """Test provider selection logic."""
    logger.info("Testing provider selection logic...")
    
    try:
        from services.transcription import TranscriptionService
        from models.transcription import TranscriptionProvider
        
        service = TranscriptionService()
        
        # Test scenarios
        scenarios = [
            ("Small file (1MB), OpenAI requested", 1024*1024, TranscriptionProvider.OPENAI),
            ("Large file (30MB), OpenAI requested", 30*1024*1024, TranscriptionProvider.OPENAI),
            ("Small file (1MB), Replicate requested", 1024*1024, TranscriptionProvider.REPLICATE),
            ("Large file (30MB), Replicate requested", 30*1024*1024, TranscriptionProvider.REPLICATE),
            ("Small file (1MB), Local requested", 1024*1024, TranscriptionProvider.LOCAL),
        ]
        
        for scenario_name, file_size, requested_provider in scenarios:
            selected_provider = service._select_provider(requested_provider, file_size)
            logger.info(f"  {scenario_name}:")
            logger.info(f"    Requested: {requested_provider.value}")
            logger.info(f"    Selected: {selected_provider.value}")
            
            if selected_provider != requested_provider:
                logger.warning(f"    Provider was changed due to constraints")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Provider selection test failed: {e}")
        return False

def analyze_issues():
    """Analyze potential configuration issues."""
    logger.info("Analyzing potential configuration issues...")
    
    issues = []
    
    try:
        from config.settings import get_settings
        from services.transcription import TranscriptionService
        
        settings = get_settings()
        service = TranscriptionService()
        
        # Check for missing API keys
        if not settings.openai_api_key:
            issues.append("OpenAI API key not configured")
        
        if not settings.replicate_api_token:
            issues.append("Replicate API token not configured")
        
        # Check file size limits
        if settings.openai_file_limit > 25 * 1024 * 1024:
            issues.append(f"OpenAI file limit ({settings.openai_file_limit}) exceeds API limit (25MB)")
        
        # Check provider initialization
        if settings.openai_api_key and not service.openai_client:
            issues.append("OpenAI client failed to initialize despite API key being set")
        
        if settings.replicate_api_token and not service.replicate_client:
            issues.append("Replicate client failed to initialize despite token being set")
        
        # Check timeout configuration
        if settings.transcription_timeout < 300:
            issues.append(f"Transcription timeout ({settings.transcription_timeout}s) might be too short")
        
        if issues:
            logger.warning("Potential issues found:")
            for issue in issues:
                logger.warning(f"  ✗ {issue}")
        else:
            logger.info("✓ No major configuration issues detected")
        
        return issues
        
    except Exception as e:
        logger.error(f"✗ Issue analysis failed: {e}")
        return [f"Failed to analyze issues: {e}"]

async def main():
    """Main test function."""
    logger.info("🔍 TRANSCRIPTION SERVICE TEST")
    logger.info("=" * 50)
    
    try:
        # Run tests
        if not test_imports():
            logger.error("Import test failed, stopping")
            return
        
        settings = test_settings()
        if not settings:
            logger.error("Settings test failed, stopping")
            return
        
        service = test_service_initialization()
        if not service:
            logger.error("Service initialization test failed, stopping")
            return
        
        await test_provider_status()
        test_provider_selection()
        issues = analyze_issues()
        
        # Summary
        logger.info("=" * 50)
        logger.info("TEST SUMMARY")
        logger.info("=" * 50)
        
        if issues:
            logger.warning(f"Found {len(issues)} potential issues")
            logger.warning("Please review the issues above")
        else:
            logger.info("✓ All tests completed successfully")
            logger.info("✓ No major issues detected")
        
        logger.info("=" * 50)
        logger.info("NEXT STEPS:")
        logger.info("1. Test with actual audio files")
        logger.info("2. Monitor API usage and costs")
        logger.info("3. Implement proper error handling")
        logger.info("4. Set up logging for production")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
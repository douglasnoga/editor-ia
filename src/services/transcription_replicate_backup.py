import os
import asyncio
import logging
import time
import requests
from typing import List, Dict, Any
import aiofiles
from openai import AsyncOpenAI

from ..models.transcription import (
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    WordTimestamp
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass

class TranscriptionService:
    """
    Multi-provider transcription service.
    
    Supports OpenAI Whisper API, Replicate WhisperX, and local Whisper
    with automatic provider selection based on file size and requirements.
    """
    
    def __init__(self):
        """Initialize the transcription service."""
        self.settings = get_settings()
        self.openai_client = None
        
        # Não usamos mais o replicate_client da biblioteca replicate
        # pois estamos usando chamadas HTTP diretas à API
        
        if self.settings.openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.transcription_timeout
            )
        
        # Registrar a disponibilidade do token Replicate
        if self.settings.replicate_api_token:
            logger.info("Replicate API token is configured and available")
        else:
            logger.warning("No Replicate API token found - Replicate provider will not work")
            
        # Verificar o valor do modelo Replicate
        if hasattr(self.settings, 'replicate_model') and self.settings.replicate_model:
            # Remover aspas, se existirem
            model = self.settings.replicate_model.strip().strip('"\'')
            logger.info(f"Replicate model configured as: {model}")
            
            # Verificar se o formato está correto (owner/name:version)
            if ':' not in model:
                logger.warning(f"Replicate model format may be incorrect: {model}")
                logger.warning("Expected format: owner/name:version_hash")
        else:
            logger.warning("No Replicate model configured - will use default")

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio using the specified provider.
        """
        try:
            if not os.path.exists(request.audio_path):
                raise TranscriptionError(f"Audio file not found: {request.audio_path}")
            
            file_size = os.path.getsize(request.audio_path)
            provider = self._select_provider(request.provider, file_size, request.enable_word_timestamps)
            
            if provider == TranscriptionProvider.OPENAI:
                return await self._transcribe_openai(request)
            elif provider == TranscriptionProvider.REPLICATE:
                return await self._transcribe_replicate(request)
            elif provider == TranscriptionProvider.LOCAL:
                return await self._transcribe_local(request)
            else:
                raise TranscriptionError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    def _select_provider(self, requested_provider: TranscriptionProvider, file_size: int, needs_word_timestamps: bool) -> TranscriptionProvider:
        """
        Select the best provider based on file size, availability, and features.
        """
        # Rule 1: If word timestamps are needed, Replicate is the priority
        if needs_word_timestamps and self.replicate_client:
            logger.info("PROVIDER DEBUG: Word timestamps needed, selecting Replicate.")
            return TranscriptionProvider.REPLICATE

        # Rule 2: If file is too large for OpenAI, use Replicate or Local
        if file_size > self.settings.openai_file_limit:
            logger.warning(f"File too large for OpenAI ({file_size} bytes).")
            if self.replicate_client:
                logger.info("PROVIDER DEBUG: Switching to Replicate for large file.")
                return TranscriptionProvider.REPLICATE
            else:
                logger.info("PROVIDER DEBUG: Switching to Local for large file.")
                return TranscriptionProvider.LOCAL

        # Rule 3: Use requested provider if available
        if requested_provider == TranscriptionProvider.OPENAI and self.openai_client:
            return TranscriptionProvider.OPENAI
        if requested_provider == TranscriptionProvider.REPLICATE and self.replicate_client:
            return TranscriptionProvider.REPLICATE
        if requested_provider == TranscriptionProvider.LOCAL:
             return TranscriptionProvider.LOCAL # Assuming local is always available if requested

        # Rule 4: Fallback logic if requested provider is unavailable
        logger.warning(f"Requested provider {requested_provider.value} not available.")
        if self.openai_client:
            logger.info("PROVIDER DEBUG: Falling back to OpenAI.")
            return TranscriptionProvider.OPENAI
        if self.replicate_client:
            logger.info("PROVIDER DEBUG: Falling back to Replicate.")
            return TranscriptionProvider.REPLICATE
        
        logger.info("PROVIDER DEBUG: Falling back to Local.")
        return TranscriptionProvider.LOCAL

    async def _transcribe_openai(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio using OpenAI Whisper API.
        """
        logger.info(f"Starting OpenAI transcription for {request.audio_path}")
        try:
            if not self.openai_client:
                raise TranscriptionError("OpenAI client not initialized")

            params = {
                "model": request.model or self.settings.whisper_model,
                "response_format": "verbose_json",
                "temperature": request.temperature
            }
            if request.language:
                params["language"] = request.language
            if request.enable_word_timestamps:
                params["timestamp_granularities"] = ["word"]
                logger.info("Word-level timestamps enabled for OpenAI")
            else:
                params["timestamp_granularities"] = ["segment"]
                logger.info("Segment-level timestamps enabled for OpenAI")

            async with aiofiles.open(request.audio_path, 'rb') as audio_file:
                audio_content = await audio_file.read()
                from io import BytesIO
                audio_buffer = BytesIO(audio_content)
                audio_buffer.name = os.path.basename(request.audio_path)

                response = await self.openai_client.audio.transcriptions.create(
                    file=audio_buffer,
                    **params
                )

            segments = self._convert_openai_response(response)
            
            return TranscriptionResponse(
                request_id=f"openai_{hash(request.audio_path)}",
                provider=TranscriptionProvider.OPENAI,
                status="success",
                segments=segments,
                language=response.language,
                duration=response.duration,
                metadata={"model": params["model"]}
            )
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {str(e)}")
            raise TranscriptionError(f"OpenAI transcription failed: {str(e)}")

    def _convert_openai_response(self, response: Any) -> List[TranscriptionSegment]:
        """
        Convert OpenAI API response to a list of TranscriptionSegment objects.
        """
        segments = []
        if not hasattr(response, 'segments') or not response.segments:
            logger.warning("No segments found in OpenAI response.")
            return segments

        for i, seg_data in enumerate(response.segments):
            words = []
            if hasattr(seg_data, 'words') and seg_data.words:
                for word_data in seg_data.words:
                    words.append(WordTimestamp(
                        word=word_data.word,
                        start=word_data.start,
                        end=word_data.end,
                        confidence=getattr(word_data, 'confidence', 0.8)
                    ))
            
            segment_confidence = getattr(seg_data, 'avg_logprob', -0.2)
            if segment_confidence < 0:
                segment_confidence = max(0.0, 1.0 + segment_confidence)

            segments.append(TranscriptionSegment(
                start=seg_data.start,
                end=seg_data.end,
                text=seg_data.text,
                confidence=segment_confidence,
                words=words if words else None,
                language=response.language
            ))
        return segments

    async def _transcribe_replicate(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio using Replicate WhisperX API.
        Implementação completa baseada na documentação oficial:
        https://replicate.com/docs/reference/http
        """
        try:
            # 1. Verificar configuração
            if not self.settings.replicate_api_token:
                raise TranscriptionError("Replicate API token not configured")

            # 2. Preparar e fazer upload do arquivo de áudio
            # Esta função agora devolve uma URL HTTP de servidor do Replicate em vez de data URL
            audio_file_url = await self._prepare_audio_for_replicate(request.audio_path)
            logger.info(f"Audio prepared and uploaded, using URL: {audio_file_url}")
            
            # 3. Preparar parâmetros para a transcrição
            input_params = {
                "audio_file": audio_file_url,  # URL HTTP do arquivo no storage do Replicate
                "align_output": True,
                "diarization": request.enable_diarization or False,
                "temperature": request.temperature or 0,
            }
            
            # Adicionar parâmetros opcionais
            if request.language:
                input_params["language"] = request.language
            
            # 4. Obter o modelo com versão completa
            # A versão deve estar no formato: owner/name:version_hash
            original_model = self.settings.replicate_model
            logger.warning(f"CRITICAL DEBUG: Original model from settings: '{original_model}', type: {type(original_model)}")
            
            # Forçar o modelo completo correto, sem depender da configuração
            model_version = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
            
            # Remover aspas e espaços extras por segurança
            model_version = model_version.strip().strip('"\'')
            
            # Log detalhado para debug
            logger.warning(f"CRITICAL DEBUG: Final model version being used: '{model_version}'")
            logger.warning(f"CRITICAL DEBUG: Does model contain ':' character? {'Yes' if ':' in model_version else 'No'}")
            logger.warning(f"CRITICAL DEBUG: Model string length: {len(model_version)}")
            
            # Garantir que temos a versão completa (verificação redundante)
            if ':' not in model_version:
                # Caso não tenha a hash, usar a versão completa fixa novamente
                logger.error(f"CRITICAL DEBUG: Something is wrong, model should have ':' but doesn't: {model_version}")
                model_version = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
            
            # 5. Configurar requisição para API Replicate
            headers = {
                "Authorization": f"Token {self.settings.replicate_api_token}",
                "Content-Type": "application/json"
            }
            
            # Payload da requisição, seguindo exatamente a documentação
            payload = {
                "version": model_version,  # Formato: owner/name:version_hash
                "input": input_params      # Parâmetros de entrada para o modelo
            }
            
            # URL do endpoint de predições
            predictions_url = "https://api.replicate.com/v1/predictions"
            
            # 6. Fazer a requisição para criar a predição
            logger.info(f"Creating prediction with payload: {payload}")
            
            response = requests.post(
                predictions_url,
                headers=headers,
                json=payload
            )
            
            # Verificar resposta
            if response.status_code != 201:
                error_details = response.json() if response.content else "No details"
                logger.error(f"Failed to create prediction: {response.status_code}, {error_details}")
                raise TranscriptionError(f"Replicate API error: {response.status_code}, {error_details}")
            
            # 7. Obter e monitorar o status da predição
            prediction = response.json()
            prediction_id = prediction["id"]
            logger.info(f"Prediction created with ID: {prediction_id}")
            
            # 8. Polling para aguardar o resultado
            max_wait_time = self.settings.transcription_timeout or 600  # Default 10 min
            start_time = time.time()
            poll_interval = 5  # segundos
            
            while time.time() - start_time < max_wait_time:
                # Verificar status atual
                status_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
                status_response = requests.get(status_url, headers=headers)
                
                if status_response.status_code != 200:
                    logger.error(f"Failed to check prediction status: {status_response.status_code}")
                    raise TranscriptionError(f"Error checking prediction status: {status_response.status_code}")
                
                prediction = status_response.json()
                status = prediction["status"]
                logger.info(f"Prediction status: {status}")
                
                # Verificar estado da predição
                if status == "succeeded":
                    logger.info("Prediction completed successfully!")
                    output = prediction["output"]
                    break
                elif status == "failed":
                    error = prediction.get("error", "Unknown error")
                    logger.error(f"Prediction failed: {error}")
                    raise TranscriptionError(f"Replicate prediction failed: {error}")
                elif status == "canceled":
                    logger.error("Prediction was canceled")
                    raise TranscriptionError("Prediction was canceled")
                
                # Aguardar antes do próximo poll
                await asyncio.sleep(poll_interval)
            else:
                # Timeout atingido
                raise TranscriptionError(f"Transcription timed out after {max_wait_time} seconds")
            
            # 9. Processar o resultado
            logger.info(f"Processing prediction output: {output}")
            segments = self._convert_replicate_response(output)
            
            # 10. Construir e retornar a resposta
            return TranscriptionResponse(
                request_id=f"replicate_{prediction_id}",
                provider=TranscriptionProvider.REPLICATE,
                status="success",
                segments=segments,
                language=output.get("detected_language", "unknown"),
                duration=segments[-1].end if segments else 0,
                metadata={
                    "model": model_version,
                    "prediction_id": prediction_id,
                    "processing_time": time.time() - start_time
                }
            )
        
        except Exception as e:
            logger.error(f"Replicate transcription failed: {str(e)}")
            raise TranscriptionError(f"Replicate transcription failed: {str(e)}")
        except Exception as e:
            logger.error(f"Replicate transcription failed: {str(e)}")
            raise TranscriptionError(f"Replicate transcription failed: {str(e)}")

    def _convert_replicate_response(self, response: Dict[str, Any]) -> List[TranscriptionSegment]:
        """
        Convert Replicate WhisperX API response to a list of TranscriptionSegment objects.
        """
        segments = []
        if not response or 'segments' not in response:
            logger.warning("No segments found in Replicate response.")
            return segments

        for seg_data in response['segments']:
            words = []
            if 'words' in seg_data and seg_data['words']:
                for word_data in seg_data['words']:
                    words.append(WordTimestamp(
                        word=word_data['word'],
                        start=word_data['start'],
                        end=word_data['end'],
                        confidence=word_data.get('score', 0.8)
                    ))
            
            segments.append(TranscriptionSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                text=seg_data['text'],
                confidence=seg_data.get('avg_logprob', 0.8),
                words=words if words else None,
                speaker=seg_data.get('speaker'),
                language=response.get('detected_language')
            ))
        return segments

    async def _transcribe_local(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Transcribe audio using local Whisper installation.
        """
        try:
            import whisper
            model_name = request.model or "base"
            model = whisper.load_model(model_name)
            
            options = {
                "language": request.language,
                "temperature": request.temperature,
                "word_timestamps": request.enable_word_timestamps
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: model.transcribe(request.audio_path, **options))
            
            segments = self._convert_local_response(result)
            
            return TranscriptionResponse(
                request_id=f"local_{hash(request.audio_path)}",
                provider=TranscriptionProvider.LOCAL,
                status="success",
                segments=segments,
                language=result.get("language", "unknown"),
                duration=sum(seg.end - seg.start for seg in segments),
                metadata={"model": model_name}
            )
        except ImportError:
            raise TranscriptionError("Local Whisper not installed. Install with: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Local transcription failed: {str(e)}")
            raise TranscriptionError(f"Local transcription failed: {str(e)}")

    def _convert_local_response(self, response: Dict[str, Any]) -> List[TranscriptionSegment]:
        """
        Convert local Whisper response to a list of TranscriptionSegment objects.
        """
        segments = []
        if 'segments' not in response:
            return segments

        for seg_data in response['segments']:
            words = []
            if 'words' in seg_data:
                for word_data in seg_data['words']:
                    words.append(WordTimestamp(
                        word=word_data['word'],
                        start=word_data['start'],
                        end=word_data['end'],
                        confidence=word_data.get('confidence', 0.8)
                    ))
            
            segments.append(TranscriptionSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                text=seg_data['text'],
                confidence=seg_data.get('avg_logprob', 0.8),
                words=words if words else None,
                language=response.get('language')
            ))
        return segments

    async def _prepare_audio_for_replicate(self, audio_path: str) -> str:
        """
        Prepare audio file for Replicate API by uploading to Replicate's storage and getting a serving URL.
        
        Para arquivos grandes, é recomendado usar URLs HTTP em vez de data URLs conforme a documentação oficial.
        https://replicate.com/docs/reference/http#predictions.create
        """
        try:
            logger.info(f"Preparing audio for Replicate using file upload API: {audio_path}")
            
            # 1. Obter um URL pré-assinado para upload do arquivo
            # Endpoint de upload do Replicate
            upload_url = "https://api.replicate.com/v1/uploads"
            
            headers = {
                "Authorization": f"Token {self.settings.replicate_api_token}",
                "Content-Type": "application/json"
            }
            
            # Determinar o tipo de conteúdo apropriado
            mime_type = 'audio/mpeg'  # Default
            if audio_path.lower().endswith('.wav'): mime_type = 'audio/wav'
            elif audio_path.lower().endswith('.m4a'): mime_type = 'audio/mp4'
            
            # Requisitar URLs para upload
            logger.info("Requesting presigned upload URL from Replicate")
            async with aiofiles.open(audio_path, 'rb') as audio_file:
                file_size = os.path.getsize(audio_path)
                logger.info(f"Audio file size: {file_size} bytes")
                
                response = requests.post(
                    upload_url,
                    headers=headers,
                    json={"type": "audio"}
                )
                
                if response.status_code != 201:
                    logger.error(f"Error requesting upload URL: {response.status_code} - {response.text}")
                    raise TranscriptionError(f"Failed to get upload URL: {response.status_code}")
                
                upload_data = response.json()
                upload_url = upload_data.get("upload_url")
                serving_url = upload_data.get("serving_url")
                
                if not upload_url or not serving_url:
                    logger.error(f"Missing URLs in response: {upload_data}")
                    raise TranscriptionError("Invalid upload response from Replicate")
                
                logger.info(f"Got presigned URL. Upload URL length: {len(upload_url)}")
                logger.info(f"Serving URL that will be used in prediction: {serving_url}")
                
                # 2. Fazer upload do arquivo para o URL pré-assinado
                logger.info("Uploading audio file to Replicate storage...")
                audio_content = await audio_file.read()
                
                upload_response = requests.put(
                    upload_url,
                    data=audio_content,
                    headers={"Content-Type": mime_type}
                )
                
                if upload_response.status_code != 200:
                    logger.error(f"Failed to upload file: {upload_response.status_code} - {upload_response.text}")
                    raise TranscriptionError(f"Failed to upload audio file: {upload_response.status_code}")
                
                logger.info(f"File uploaded successfully. Serving URL: {serving_url}")
                return serving_url
                
        except Exception as e:
            logger.error(f"Error preparing audio for Replicate: {str(e)}")
            raise TranscriptionError(f"Failed to prepare audio for Replicate: {e}")

    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all transcription providers.
        """
        status = {}
        
        # OpenAI status
        status['openai'] = {
            'available': self.openai_client is not None,
            'api_key_set': bool(self.settings.openai_api_key),
            'file_limit_mb': self.settings.openai_file_limit / (1024 * 1024),
            'model': self.settings.whisper_model
        }
        
        # Replicate status
        status['replicate'] = {
            'available': bool(self.settings.replicate_api_token),
            'api_token_set': bool(self.settings.replicate_api_token),
            'model': self.settings.replicate_model
        }
        
        # Local Whisper status
        try:
            import importlib.util
            local_available = importlib.util.find_spec("whisper") is not None
        except ImportError:
            local_available = False
        
        status['local'] = {
            'available': local_available,
            'models': ['tiny', 'base', 'small', 'medium', 'large'] if local_available else []
        }
        
        return status
    
    async def estimate_transcription_cost(self, audio_path: str, provider: TranscriptionProvider) -> Dict[str, Any]:
        """
        Estimate transcription cost for given audio file.
        """
        try:
            # Get audio duration
            from moviepy.editor import AudioFileClip
            
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            
            # Estimate costs (rough estimates)
            if provider == TranscriptionProvider.OPENAI:
                # OpenAI charges $0.006 per minute
                estimated_cost = duration / 60 * 0.006
            elif provider == TranscriptionProvider.REPLICATE:
                # Replicate charges ~$0.018 per run
                estimated_cost = 0.018
            else:
                # Local is free
                estimated_cost = 0.0
            
            return {
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'estimated_cost_usd': estimated_cost,
                'provider': provider.value
            }
            
        except Exception as e:
            logger.error(f"Error estimating transcription cost: {str(e)}")
            return {
                'error': str(e)
            }
from __future__ import annotations
import os
import subprocess
from openai import OpenAI, APIStatusError
from dotenv import load_dotenv
import json
from dataclasses import dataclass, asdict
import logging
import math
import tempfile
import shutil

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Limite da API do Whisper é 25MB. Usamos um valor um pouco menor por segurança.
# Limite da API do Whisper é 25MB. Usamos 15MB por segurança - ajustado para menor ainda.
WHISPER_API_LIMIT_BYTES = 15 * 1024 * 1024 

@dataclass
class TranscriptionWord:
    word: str
    start: float
    end: float

@dataclass
class TranscriptionResult:
    text: str
    words: list[TranscriptionWord]

@dataclass
class TranscriptionRequest:
    file_path: str

class TranscriptionService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no .env")
        self.client = OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")

    async def _transcribe_chunk(self, file_path: str) -> dict:
        """Transcreve um chunk de áudio usando a API OpenAI Whisper"""
        logger.info(f"Transcrevendo chunk de áudio: {file_path}")
        with open(file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json"
                # timestamp_granularities não suportado nesta versão do OpenAI
            )
            
            # Log da estrutura da resposta para debug
            logger.debug(f"Resposta da API: {type(response)} - Atributos: {dir(response)}")
            
            # Verificar se words está disponível na resposta
            if hasattr(response, 'words') and response.words:
                word_count = len(response.words)
                logger.info(f"Transcrição bem-sucedida: {word_count} palavras com timestamps")
            else:
                logger.info(f"Transcrição bem-sucedida: texto completo (sem timestamps de palavras)")
                # Criar estrutura compatível quando words não está disponível
                if not hasattr(response, 'words'):
                    response.words = []
                    
            return response

    async def _compress_audio(self, input_path: str) -> str:
        """Comprime o áudio para reduzir seu tamanho mantendo qualidade suficiente para transcrição, usando FFmpeg diretamente"""
        try:
            logger.info(f"Comprimindo áudio com FFmpeg: {input_path}")
            
            # Criar nome para o arquivo comprimido
            compressed_path = f"{os.path.splitext(input_path)[0]}_compressed.mp3"
            
            # Verifica se o arquivo de entrada existe
            if not os.path.exists(input_path):
                logger.error(f"Arquivo de entrada não encontrado: {input_path}")
                return input_path
                
            # Comandos FFmpeg para compressão máxima:
            # -i: arquivo de entrada
            # -ac 1: converter para mono (1 canal)
            # -ar 16000: reduzir sample rate para 16kHz (suficiente para fala)
            # -q:a 9: menor qualidade de áudio (0-9, onde 9 é o máximo de compressão)
            # -b:a 16k: bitrate ultra-baixo
            # -y: sobrescrever arquivo de saída se existir
            # -loglevel warning: mostra apenas avisos e erros para reduzir ruído de log
            command = [
                "ffmpeg",
                "-i", input_path,
                "-ac", "1",
                "-ar", "16000",
                "-q:a", "9",
                "-b:a", "16k",
                "-y",
                "-loglevel", "warning",
                compressed_path
            ]
            
            logger.info(f"Executando comando: {' '.join(command)}")
            
            # Executar o comando FFmpeg - no Windows, shell=True pode ser necessário
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Verificar se o comando foi executado com sucesso
            if process.returncode != 0:
                logger.error(f"Erro ao comprimir áudio com FFmpeg: {process.stderr}")
                return input_path
                
            # Verificar se o arquivo comprimido foi criado e seu tamanho
            if not os.path.exists(compressed_path):
                logger.error(f"FFmpeg falhou: arquivo comprimido não foi criado: {compressed_path}")
                return input_path
                
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = compressed_size / original_size
            
            logger.info(f"Áudio comprimido de {original_size/1024/1024:.2f}MB para {compressed_size/1024/1024:.2f}MB (compressão: {compression_ratio:.2%})")
            return compressed_path
        except Exception as e:
            logger.error(f"Erro ao comprimir áudio: {e}")
            # Se falhar a compressão, retorna o caminho original
            return input_path
    
    async def transcribe_audio(self, request: TranscriptionRequest) -> TranscriptionResult:
        logger.info(f"Iniciando transcrição para o arquivo: {request.file_path}")
        
        # Comprimir o áudio para reduzir seu tamanho
        compressed_audio_path = await self._compress_audio(request.file_path)
        file_size = os.path.getsize(compressed_audio_path)

        full_text = ""
        all_words = []

        if file_size <= WHISPER_API_LIMIT_BYTES:
            logger.info("Arquivo dentro do limite de tamanho, transcrevendo diretamente.")
            try:
                response = await self._transcribe_chunk(compressed_audio_path)
                full_text = response.text
                
                # Corrigindo o processamento da resposta Whisper
                if hasattr(response, 'words') and response.words:
                    logger.info(f"Processando {len(response.words)} palavras da resposta da API Whisper")
                    all_words = []
                    for word_data in response.words:
                        word = word_data.word if hasattr(word_data, 'word') else word_data.get('word', '')
                        start = word_data.start if hasattr(word_data, 'start') else word_data.get('start', 0.0)
                        end = word_data.end if hasattr(word_data, 'end') else word_data.get('end', 0.0)
                        all_words.append(TranscriptionWord(word=word, start=start, end=end))
                else:
                    logger.info("Resposta da API Whisper sem timestamps de palavras, continuando sem eles")
                    all_words = []
            except APIStatusError as e:
                logger.error(f"Erro durante a chamada da API de transcrição direta: {e}")
                raise e
        else:
            logger.info(f"Arquivo excede o limite de {WHISPER_API_LIMIT_BYTES} bytes. Iniciando fatiamento.")
            
            # Obter a duração do arquivo de áudio usando FFmpeg
            try:
                duration_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    compressed_audio_path
                ]
                duration_process = subprocess.run(duration_cmd, capture_output=True, text=True)
                if duration_process.returncode != 0:
                    logger.error(f"Erro ao obter duração do áudio: {duration_process.stderr}")
                    raise Exception(f"Erro ao obter duração do áudio: {duration_process.stderr}")
                
                total_duration_sec = float(duration_process.stdout.strip())
                total_duration_ms = total_duration_sec * 1000
            except Exception as e:
                logger.error(f"Erro ao obter informações do áudio: {e}")
                raise

            max_chunk_duration_ms = 20 * 60 * 1000  # 20 minutos por chunk (em ms)
            
            num_chunks_by_size = math.ceil(file_size / WHISPER_API_LIMIT_BYTES)
            estimated_chunk_duration_ms = math.ceil(total_duration_ms / num_chunks_by_size)
            
            chunk_duration_ms = min(estimated_chunk_duration_ms, max_chunk_duration_ms)
            num_chunks = math.ceil(total_duration_ms / chunk_duration_ms)

            logger.info(f"Dividindo o áudio em {num_chunks} chunks de aproximadamente {chunk_duration_ms / 1000 / 60:.2f} minutos.")

            temp_dir = tempfile.mkdtemp()
            try:
                for i in range(num_chunks):
                    start_sec = (i * chunk_duration_ms) / 1000
                    duration_sec = min(chunk_duration_ms / 1000, total_duration_sec - start_sec)
                    
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
                    
                    try:
                        # Extrair chunk com FFmpeg
                        logger.info(f"Extraindo chunk {i+1}/{num_chunks} (de {start_sec:.2f}s por {duration_sec:.2f}s)")
                        chunk_cmd = [
                            "ffmpeg",
                            "-i", compressed_audio_path,
                            "-ss", str(start_sec),
                            "-t", str(duration_sec),
                            "-c:a", "libmp3lame",
                            "-q:a", "8",  # Qualidade média-baixa para manter arquivo pequeno
                            "-y",
                            chunk_path
                        ]
                        
                        chunk_process = subprocess.run(chunk_cmd, capture_output=True, text=True)
                        if chunk_process.returncode != 0:
                            logger.error(f"Erro ao extrair chunk {i+1}: {chunk_process.stderr}")
                            continue
                        
                        # Transcrever o chunk
                        logger.info(f"Transcrevendo chunk {i+1}/{num_chunks}")
                        transcribed_chunk = await self._transcribe_chunk(chunk_path)
                        full_text += transcribed_chunk.text + " "
                        
                        # Ajustar os timestamps das palavras para o offset correto
                        chunk_words = []
                        if hasattr(transcribed_chunk, 'words') and transcribed_chunk.words:
                            for word_data in transcribed_chunk.words:
                                word = word_data.word if hasattr(word_data, 'word') else word_data.get('word', '')
                                start = word_data.start if hasattr(word_data, 'start') else word_data.get('start', 0.0)
                                end = word_data.end if hasattr(word_data, 'end') else word_data.get('end', 0.0)
                                
                                # Adiciona o offset ao timestamp
                                chunk_words.append(TranscriptionWord(
                                    word=word,
                                    start=start + offset, 
                                    end=end + offset
                                ))
                            all_words.extend(chunk_words)
                        else:
                            logger.warning(f"Chunk {i+1} não contém informações de palavras com timestamps")
                            # Criar timestamps estimados baseados no texto
                            if hasattr(transcribed_chunk, 'text') and transcribed_chunk.text:
                                words = transcribed_chunk.text.split()
                                if words:
                                    # Distribuir palavras uniformemente ao longo da duração do chunk
                                    word_duration = duration_sec / len(words)
                                    for j, word in enumerate(words):
                                        word_start = offset + (j * word_duration)
                                        word_end = offset + ((j + 1) * word_duration)
                                        chunk_words.append(TranscriptionWord(
                                            word=word,
                                            start=word_start,
                                            end=word_end
                                        ))
                                    all_words.extend(chunk_words)
                                    logger.info(f"Criados timestamps estimados para {len(words)} palavras no chunk {i+1}")
                        offset += duration_sec
                    except Exception as e:
                        logger.error(f"Erro ao processar o chunk {i+1}: {e}")
                    finally:
                        # Remover arquivo de chunk após o uso
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
            finally:
                # Limpar diretório temporário
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Diretório temporário removido: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Erro ao remover diretório temporário: {e}")
        
        result = TranscriptionResult(text=full_text.strip(), words=all_words)

        # Salvar os arquivos de transcrição usando o caminho do arquivo original
        base_path = os.path.splitext(request.file_path)[0]
        json_path = f"{base_path}_transcription.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=4)
        logger.info(f"Transcrição completa salva em {json_path}")

        txt_path = f"{base_path}_transcription.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.text)
        logger.info(f"Texto da transcrição salvo em {txt_path}")
        
        # Limpar o arquivo de áudio comprimido temporário
        if compressed_audio_path != request.file_path and os.path.exists(compressed_audio_path):
            try:
                os.remove(compressed_audio_path)
                logger.info(f"Arquivo de áudio comprimido temporário removido: {compressed_audio_path}")
            except Exception as e:
                logger.warning(f"Não foi possível remover o arquivo temporário {compressed_audio_path}: {e}")

        return result

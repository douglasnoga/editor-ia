import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class Settings(BaseSettings):
    """Configurações da aplicação, carregadas a partir de variáveis de ambiente."""
    
    # --- Configurações da OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    whisper_model: str = os.getenv("WHISPER_MODEL", "whisper-1")
    openai_file_limit: int = 25 * 1024 * 1024  # Limite de 25MB da API da OpenAI
    transcription_timeout: int = 600  # 10 minutos
    ai_analysis_timeout: int = 300    # 5 minutos
    
    # --- Configurações do Replicate ---
    replicate_api_token: str = os.getenv("REPLICATE_API_TOKEN", "")
    replicate_model: str = os.getenv("REPLICATE_MODEL", "")

    # --- Configurações do Servidor ---
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    
    # --- Configurações de Processamento ---
    temp_dir: str = "temp_files"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000  # Taxa de amostragem padrão para vídeos

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'  # Ignora variáveis extras do .env

@lru_cache()
def get_settings() -> Settings:
    """
    Retorna uma instância cacheada das configurações.
    O uso de lru_cache garante que o arquivo .env seja lido apenas uma vez.
    """
    return Settings()

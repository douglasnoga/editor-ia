import uvicorn
import os
import logging
import uuid
import json
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocketDisconnect

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List

from .services.video_processor import VideoProcessor
from .services.transcription import TranscriptionService

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Gerenciador de Conexões WebSocket ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Cliente conectado: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Cliente desconectado: {client_id}")

    async def send_status_update(self, client_id: str, status: str, progress: int, guide_url: str = None, xml_url: str = None):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            message = {
                "status": status,
                "progress": progress,
                "guide_url": guide_url,
                "xml_url": xml_url
            }
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()
transcription_service = TranscriptionService()

# --- Montagem de diretórios Estáticos ---
TEMP_DIR = "temp"
STATIC_DIR = "static"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# --- Endpoints da API ---
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text() # Apenas para manter a conexão viva
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/upload/{client_id}")
async def upload_video(
    client_id: str,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    video_type: str = Form("Geral (Padrão)"),
    instructions: str = Form("")
):
    if not client_id in manager.active_connections:
        raise HTTPException(status_code=404, detail="Cliente WebSocket não encontrado.")

    try:
        # Salva o arquivo temporariamente
        temp_filename = f"{uuid.uuid4()}_{video_file.filename}"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await video_file.read())
        logger.info(f"Arquivo salvo temporariamente em: {temp_file_path}")

        # Instancia o processador e adiciona a tarefa em background
        video_processor = VideoProcessor(manager, client_id, transcription_service)
        background_tasks.add_task(
            video_processor.process_video, temp_file_path, video_type, instructions
        )

        return {"message": "Processamento iniciado.", "file": video_file.filename}
    except Exception as e:
        logger.error(f"Falha no upload do arquivo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Falha no upload: {e}")

# --- Ponto de Entrada para Execução ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request

# Carrega as variáveis de ambiente do arquivo .env na raiz do projeto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.connection_manager import ConnectionManager
from src.services.transcription import TranscriptionService
from src.services.analysis_service import AnalysisService
from src.services.video_processor import VideoProcessor

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Monta o diretório estático para servir o frontend
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Monta o diretório temporário para servir os arquivos gerados
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Inicialização dos serviços e do gerenciador de conexões
manager = ConnectionManager()
transcription_service = TranscriptionService()
analysis_service = AnalysisService()

@app.get("/", response_class=HTMLResponse)
async def get():
    html_path = os.path.join(static_dir, 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/upload/{client_id}")
async def transcribe_video(client_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...), instructions: str = Form(""), videoType: str = Form("geral")):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join("temp", unique_filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(f"Arquivo salvo temporariamente em: {temp_file_path}")

        # Inicia o processamento em segundo plano
        video_processor = VideoProcessor(manager=manager, client_id=client_id, transcription_service=transcription_service)
        background_tasks.add_task(video_processor.process_video, temp_file_path, videoType, instructions)

        return JSONResponse(
            status_code=202, # Accepted
            content={"message": "O processamento do vídeo foi iniciado."}
        )
    except Exception as e:
        logger.error(f"Erro durante o upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro no servidor: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

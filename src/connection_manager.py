import logging
import json
from typing import Dict
from fastapi import WebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                "progress": progress
            }
            if guide_url is not None:
                message["guide_url"] = guide_url
            if xml_url is not None:
                message["xml_url"] = xml_url
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {client_id}: {e}")
                self.disconnect(client_id)
                
    async def send_progress(self, client_id: str, status: str, progress: int):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json({"status": status, "progress": progress})
            except Exception as e:
                logger.error(f"Falha ao enviar mensagem para {client_id}: {e}")
                self.disconnect(client_id)

    async def send_download_links(self, client_id: str, guide_url: str, xml_url: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json({
                    "status": "Processo concluído!",
                    "progress": 100,
                    "guide_url": guide_url,
                    "xml_url": xml_url
                })
            except Exception as e:
                logger.error(f"Falha ao enviar links de download para {client_id}: {e}")
                self.disconnect(client_id)

    async def send_error(self, client_id: str, error_message: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json({"status": "Falhou", "error": error_message, "progress": 100})
            except Exception as e:
                logger.error(f"Falha ao enviar mensagem de erro para {client_id}: {e}")
                self.disconnect(client_id)

"""
Guide Generator - Gera guias de corte diretamente das transcrições

Este módulo fornece funções para gerar guias de corte precisos a partir de
transcrições originais, garantindo alinhamento perfeito entre timestamps 
e o vídeo/áudio original.
"""

import json
import os
import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from openai import OpenAI

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuideGenerator:
    """Classe responsável por gerar guias de corte a partir de transcrições."""
    
    def __init__(self):
        # Inicializar cliente OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        
        # Configurações para tipos específicos de vídeos
        self.config = {
            "vsl": {
                "target_seg_len_sec": 75,  # 60-90s por segmento
                "max_pause_ms": 600,       # Silêncios acima disso são cortados
                "compress_range": (0.25, 0.40),  # Manter 60-75% do bruto
                "gap_at_join_ms": 150,      # Pausa entre segmentos
                "muletas": ["né", "então", "cara", "tipo", "é", "ééé", "humm", "olha só"],
                "anchor_regex": [
                    r"5 ?mil", r"8 ?mil", r"10 ?mil",
                    r"vale[ra]?\s+a\s+pena", r"seria\s+caro"
                ]
            },
            "youtube_live": {
                "target_seg_len_sec": 90,   # Tamanho ideal por segmento
                "max_pause_ms": 800,        # Silêncios a cortar
                "compress_range": (0.70, 0.90),  # % de remoção desejada
                "gap_at_join_ms": 100,      # Pausa entre segmentos
                "muletas": ["né", "então", "cara", "tipo", "tá", "é", "humm"],
                "hook_regex": [
                    r"\bo que é\b",
                    r"\bpresta[m]?\s+atenção\b",
                    r"\beste[áa]\s+estat[íi]stic",
                    r"\bexiste um artigo\b",
                    r"\bquando você\b.*\bcopy\b"
                ]
            },
            "geral": {
                "target_seg_len_sec": 60,   # Tamanho padrão
                "max_pause_ms": 700,        # Silêncio padrão
                "compress_range": (0.50, 0.80),  # Compressão padrão
                "gap_at_join_ms": 100       # Pausa padrão
            }
        }
        
    def _format_time(self, seconds: float) -> str:
        """Converte segundos para o formato HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        if h > 0:
            return f"{h:02}:{m:02}:{s:02}.{ms:03}"
        else:
            return f"{m:02}:{s:02}.{ms:03}"
            
    def _time_str_to_seconds(self, time_str: str) -> float:
        """Converte uma string de tempo (HH:MM:SS ou MM:SS) para segundos."""
        try:
            # Limpando a string para remover espaços extras
            time_str = time_str.strip()
            
            # Verificando formato
            parts = time_str.split(':')        
            if len(parts) == 3:  # HH:MM:SS
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
            elif len(parts) == 2:  # MM:SS
                hours = 0
                minutes = float(parts[0])
                seconds = float(parts[1])
            else:
                # Tente interpretar como número puro de segundos
                try:
                    return float(time_str)
                except ValueError:
                    logger.error(f"Formato de tempo inválido: {time_str}")
                    return 0.0
                
            # Cálculo com precisão decimal
            total_seconds = hours * 3600.0 + minutes * 60.0 + seconds
            
            # Arredondamento para 3 casas decimais para evitar erros de ponto flutuante
            total_seconds = round(total_seconds, 3)
            
            return total_seconds
        except Exception as e:
            logger.error(f"Erro ao converter timestamp {time_str}: {str(e)}")
            return 0.0

    def _extract_words_from_transcription(self, transcription_data: Dict) -> List[Dict]:
        """Extrai a lista de palavras de uma transcrição em diversos formatos."""
        words = []
        
        # Tenta encontrar palavras no formato Whisper
        if "segments" in transcription_data and isinstance(transcription_data["segments"], list):
            for segment in transcription_data["segments"]:
                if "words" in segment and isinstance(segment["words"], list):
                    words.extend(segment["words"])
                elif "text" in segment and "start" in segment and "end" in segment:
                    # Segmento sem palavras individuais, criar uma palavra para o segmento inteiro
                    words.append({
                        "word": segment["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"]
                    })
        
        # Formato alternativo com palavras diretamente na raiz
        elif "words" in transcription_data and isinstance(transcription_data["words"], list):
            words = transcription_data["words"]
            
        # Pesquisa genérica para qualquer lista contendo palavras
        else:
            for key, value in transcription_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    # Verifica se tem word, start e end nos primeiros itens
                    if all("word" in item and "start" in item and "end" in item for item in value[:5]):
                        words = value
                        break
        
        # FALLBACK: Se não há palavras com timestamps, criar a partir do texto completo
        if not words and "text" in transcription_data and transcription_data["text"].strip():
            logger.warning("Nenhuma palavra com timestamp encontrada. Criando segmentos estimados a partir do texto.")
            text = transcription_data["text"].strip()
            
            # Usar duração real do vídeo se disponível
            video_duration = transcription_data.get("duration", None)
            if not video_duration:
                # Estimar duração baseada no texto (assumindo ~150 palavras por minuto)
                total_words = len(text.split())
                video_duration = (total_words / 150) * 60  # em segundos
                logger.info(f"Duração estimada baseada no texto: {video_duration:.1f}s")
            else:
                logger.info(f"Usando duração real do vídeo: {video_duration:.1f}s")
            
            # Dividir o texto em palavras individuais para criar timestamps mais precisos
            import re
            text_words = text.split()
            total_words = len(text_words)
            
            # Criar timestamps para cada palavra individual
            current_time = 0
            word_duration = video_duration / total_words if total_words > 0 else 1.0
            
            for word in text_words:
                # Limpar pontuação para análise
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word.strip():  # Só adicionar se não for vazio
                    words.append({
                        "word": word,
                        "start": current_time,
                        "end": current_time + word_duration
                    })
                current_time += word_duration
            
            logger.info(f"Criados {len(words)} segmentos estimados usando duração real de {video_duration:.1f}s")
        
        # Filtra palavras vazias e ordena por tempo
        words = [w for w in words if w.get("word", "").strip()]
        words.sort(key=lambda x: x.get("start", 0))
        
        return words

    def _group_words_into_sentences(self, words: List[Dict]) -> List[Dict]:
        """Agrupa palavras em frases."""
        sentences = []
        current_sentence = {"text": "", "words": [], "start": None, "end": None}
        
        # Caracteres que normalmente indicam o fim de uma frase
        punctuation = [".", "!", "?", ";", ","]
        
        for word in words:
            word_text = word.get("word", "").strip()
            if not word_text:
                continue
                
            # Inicializa o tempo de início se for a primeira palavra da frase
            if current_sentence["start"] is None:
                current_sentence["start"] = word["start"]
                
            # Adiciona a palavra à frase atual
            current_sentence["text"] += " " + word_text if current_sentence["text"] else word_text
            current_sentence["words"].append(word)
            current_sentence["end"] = word["end"]
            
            # Termina a frase apenas com pontuação forte (deixa IA decidir tamanho por conteúdo)
            strong_punctuation = [".", "!", "?"]
            if any(p in word_text for p in strong_punctuation):
                sentences.append(current_sentence)
                current_sentence = {"text": "", "words": [], "start": None, "end": None}
        
        # Adiciona a última frase se não estiver vazia
        if current_sentence["text"] and current_sentence["start"] is not None:
            sentences.append(current_sentence)
            
        return sentences
    
    def _create_segments_for_vsl(self, sentences: List[Dict]) -> List[Dict]:
        """Cria segmentos específicos para um VSL, seguindo as regras do prompt VSL."""
        config = self.config["vsl"]
        target_seg_len = config["target_seg_len_sec"]
        anchor_patterns = [re.compile(pattern) for pattern in config["anchor_regex"]]
        
        segments = []
        current_segment = None
        
        # Primeiro, identificamos o gancho (hook) se existir
        hook_sentence = None
        for i, sentence in enumerate(sentences):
            # Verifica se a frase contém uma âncora de preço ou gancho
            if any(pattern.search(sentence["text"].lower()) for pattern in anchor_patterns):
                hook_sentence = sentence
                # Se aparecer depois de 30s, será movido para o início depois
                if hook_sentence["start"] > 30:
                    break
                
        # Se encontrou um gancho depois de 30s, move para o início
        processed_sentences = []
        if hook_sentence and hook_sentence["start"] > 30:
            hook_index = sentences.index(hook_sentence)
            processed_sentences.append(hook_sentence)
            processed_sentences.extend(sentences[:hook_index])
            processed_sentences.extend(sentences[hook_index+1:])
        else:
            processed_sentences = sentences
        
        # Agrupa as frases em segmentos
        for sentence in processed_sentences:
            if current_segment is None:
                current_segment = {
                    "texto": sentence["text"],
                    "inicio": self._format_time(sentence["start"]),
                    "fim": self._format_time(sentence["end"])
                }
            else:
                # Verifica se adicionar esta frase mantém o segmento dentro do tamanho alvo
                segment_duration = sentence["end"] - self._time_str_to_seconds(current_segment["inicio"])
                if segment_duration <= target_seg_len:
                    current_segment["texto"] += " " + sentence["text"]
                    current_segment["fim"] = self._format_time(sentence["end"])
                else:
                    segments.append(current_segment)
                    current_segment = {
                        "texto": sentence["text"],
                        "inicio": self._format_time(sentence["start"]),
                        "fim": self._format_time(sentence["end"])
                    }
        
        # Adiciona o último segmento se não estiver vazio
        if current_segment:
            segments.append(current_segment)
            
        # Define funções para os segmentos
        for i, segment in enumerate(segments):
            if i == 0 and hook_sentence:
                segment["funcao"] = "gancho"
            else:
                # Analisa o texto para tentar identificar a função
                text = segment["texto"].lower()
                if any(re.search(pattern, text) for pattern in config["anchor_regex"]):
                    if "garantia" in text or "devolvemos" in text:
                        segment["funcao"] = "garantia"
                    else:
                        segment["funcao"] = "oferta"
                elif "garantia" in text or "devolvemos" in text or "risco" in text:
                    segment["funcao"] = "garantia"
                elif "resultados" in text or "depoimento" in text or "case" in text:
                    segment["funcao"] = "prova"
                elif "clicar" in text or "botão" in text or "agora" in text:
                    segment["funcao"] = "cta"
                else:
                    segment["funcao"] = "desenvolvimento"
        
        return segments
    
    def _create_segments_for_youtube_live(self, sentences: List[Dict]) -> List[Dict]:
        """Cria segmentos específicos para um corte de Live do YouTube."""
        config = self.config["youtube_live"]
        target_seg_len = config["target_seg_len_sec"]
        hook_patterns = [re.compile(pattern) for pattern in config["hook_regex"]]
        
        segments = []
        current_segment = None
        
        # Primeiro, identificamos o gancho (hook) se existir
        hook_sentence = None
        for i, sentence in enumerate(sentences):
            if sentence["start"] > 60:  # Procura após 60s conforme regra
                if any(pattern.search(sentence["text"].lower()) for pattern in hook_patterns):
                    hook_sentence = sentence
                    break
        
        # Se encontrou um gancho depois de 60s, move para o início
        processed_sentences = []
        if hook_sentence:
            hook_index = sentences.index(hook_sentence)
            processed_sentences.append(hook_sentence)
            processed_sentences.extend(sentences[:hook_index])
            processed_sentences.extend(sentences[hook_index+1:])
        else:
            processed_sentences = sentences
        
        # Agrupa as frases em segmentos
        for sentence in processed_sentences:
            if current_segment is None:
                current_segment = {
                    "texto": sentence["text"],
                    "inicio": self._format_time(sentence["start"]),
                    "fim": self._format_time(sentence["end"])
                }
            else:
                segment_duration = sentence["end"] - self._time_str_to_seconds(current_segment["inicio"])
                if segment_duration <= target_seg_len:
                    current_segment["texto"] += " " + sentence["text"]
                    current_segment["fim"] = self._format_time(sentence["end"])
                else:
                    segments.append(current_segment)
                    current_segment = {
                        "texto": sentence["text"],
                        "inicio": self._format_time(sentence["start"]),
                        "fim": self._format_time(sentence["end"])
                    }
        
        # Adiciona o último segmento
        if current_segment:
            segments.append(current_segment)
            
        # Define funções para os segmentos
        for i, segment in enumerate(segments):
            if i == 0 and hook_sentence:
                segment["funcao"] = "gancho"
            elif "exemplo" in segment["texto"].lower() or "caso" in segment["texto"].lower():
                segment["funcao"] = "exemplo_pratico"
            else:
                segment["funcao"] = "desenvolvimento"
        
        return segments
    
    def _create_segments_general(self, sentences: List[Dict]) -> List[Dict]:
        """Cria segmentos gerais sem regras específicas de prompt."""
        config = self.config["geral"]
        target_seg_len = config["target_seg_len_sec"]
        
        segments = []
        current_segment = None
        
        # Agrupa as frases em segmentos de duração aproximada
        for sentence in sentences:
            if current_segment is None:
                current_segment = {
                    "texto": sentence["text"],
                    "inicio": self._format_time(sentence["start"]),
                    "fim": self._format_time(sentence["end"])
                }
            else:
                segment_duration = sentence["end"] - self._time_str_to_seconds(current_segment["inicio"])
                if segment_duration <= target_seg_len:
                    current_segment["texto"] += " " + sentence["text"]
                    current_segment["fim"] = self._format_time(sentence["end"])
                else:
                    segments.append(current_segment)
                    current_segment = {
                        "texto": sentence["text"],
                        "inicio": self._format_time(sentence["start"]),
                        "fim": self._format_time(sentence["end"])
                    }
        
        # Adiciona o último segmento
        if current_segment:
            segments.append(current_segment)
            
        return segments
    
    def generate_cutting_guide(self, transcription_path: str, prompt_type: str = "geral", custom_instructions: str = None) -> Dict:
        """Gera um guia de corte a partir da transcrição original.
        
        Args:
            transcription_path: Caminho para o arquivo JSON da transcrição
            prompt_type: Tipo de prompt a ser usado ('vsl', 'youtube_live', ou 'geral')
            custom_instructions: Instruções personalizadas do usuário para guiar os cortes
            
        Returns:
            Dict contendo o guia de corte gerado
        """
        try:
            # Carregar a transcrição
            logger.info(f"Carregando transcrição de {transcription_path}")
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
                
            # Extrair palavras da transcrição
            words = self._extract_words_from_transcription(transcription_data)
            if not words:
                logger.warning("Nenhuma palavra encontrada na transcrição")
                return {"segmentos": []}
                
            # Agrupar palavras em frases
            sentences = self._group_words_into_sentences(words)
            
            # Criar segmentos baseados no tipo de prompt
            if prompt_type.lower() == "vsl":
                segments = self._create_segments_for_vsl(sentences, custom_instructions)
            elif prompt_type.lower() == "youtube_live":
                segments = self._create_segments_for_youtube_live(sentences, custom_instructions)
            else:
                segments = self._create_segments_general(sentences, custom_instructions)
            
            # Criar o guia de corte final
            cutting_guide = {
                "segmentos": segments
            }
            
            logger.info(f"Guia de corte gerado com {len(segments)} segmentos")
            return cutting_guide
            
        except Exception as e:
            logger.error(f"Erro ao gerar guia de corte da transcrição: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"segmentos": []}
    
    def verify_guide_transcription_compatibility(self, guide_path: str, transcription_path: str) -> Tuple[bool, float]:
        """Verifica se há compatibilidade entre os timestamps do guia de corte e da transcrição.
        
        Args:
            guide_path: Caminho para o arquivo JSON do guia de corte
            transcription_path: Caminho para o arquivo JSON da transcrição
            
        Returns:
            Tuple(is_compatible, offset_seconds)
        """
        try:
            # Carregar guia de corte
            with open(guide_path, 'r', encoding='utf-8') as f:
                cutting_guide = json.load(f)
                
            # Carregar transcrição
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
                
            # Obter palavras da transcrição
            words = self._extract_words_from_transcription(transcription_data)
                
            if not words:
                logger.warning("Não foi possível obter palavras da transcrição para verificação")
                return False, 0
                
            # Obter o primeiro e último segmento do guia
            segments = cutting_guide.get("segmentos", [])
            if not segments:
                logger.warning("Não há segmentos no guia de corte para verificar")
                return False, 0
                
            first_segment = segments[0]
            last_segment = segments[-1]
            
            # Converter timestamps para segundos
            first_guide_start = self._time_str_to_seconds(first_segment.get("inicio", "0:00"))
            last_guide_end = self._time_str_to_seconds(last_segment.get("fim", "0:00"))
            
            # Obter primeiro e último timestamp da transcrição
            first_word_time = words[0].get("start", 0) if words else 0
            last_word_time = words[-1].get("end", 0) if words else 0
            
            # Calcular diferença média (possível offset)
            start_diff = first_guide_start - first_word_time
            end_diff = last_guide_end - last_word_time
            avg_diff = (start_diff + end_diff) / 2
            
            # Verificar se há uma diferença significativa
            threshold = 10.0  # segundos
            if abs(start_diff) > threshold or abs(end_diff) > threshold:
                logger.warning(f"Possível incompatibilidade entre guia e transcrição!")
                logger.warning(f"Primeiro timestamp do guia: {first_guide_start}s vs. transcrição: {first_word_time}s (diff: {start_diff:.2f}s)")
                logger.warning(f"Último timestamp do guia: {last_guide_end}s vs. transcrição: {last_word_time}s (diff: {end_diff:.2f}s)")
                
                # Se houver uma grande diferença, o guia pode estar usando outro ponto de referência
                if abs(start_diff) > 30.0 or abs(end_diff) > 30.0:
                    logger.warning("Diferença muito grande detectada! Recomenda-se regenerar o guia de corte.")
                    return False, avg_diff
                    
            logger.info("Guia de corte e transcrição parecem compatíveis.")
            return True, avg_diff
            
        except Exception as e:
            logger.error(f"Erro ao verificar compatibilidade: {e}")
            return False, 0

    def _create_segments_general(self, sentences: List[Dict], custom_instructions: str = None) -> List[Dict]:
        """Cria segmentos para o tipo geral, fazendo CORTES baseados no conteúdo."""
        if not sentences:
            return []
        
        # Texto completo para análise da IA
        full_text = " ".join([s["text"] for s in sentences])
        
        # Construir critérios baseados nas instruções personalizadas
        if custom_instructions and custom_instructions.strip():
            criteria = f"Instruções específicas: {custom_instructions.strip()}"
        else:
            criteria = """Critérios gerais:
        - Trechos com maior valor e relevância
        - Conteúdo mais interessante e envolvente
        - Exemplos práticos e histórias marcantes
        - Evitar repetições desnecessárias
        - Manter fluxo narrativo coerente"""
        
        # Prompt para IA fazer CORTES reais com segmentos de qualidade
        prompt = f"""Analise este texto de vídeo e identifique os melhores trechos que devem ser MANTIDOS no vídeo final.

OBJETIVO: Criar um vídeo de QUALIDADE com trechos substanciais.

Duração: depende do tipo de vídeo. Os templates já trazem essa informação. O usuário também pode colocá-la nas instruções.

{criteria}

IMPORTANTE:
- Selecione segmentos baseado APENAS na QUALIDADE e RELEVÂNCIA do conteúdo
- Ignore duração ou tamanho - foque no SIGNIFICADO e na COMPLETUDE da informação
- Busque por conceitos relacionados mesmo que não sejam mencionados diretamente
- Priorize trechos que desenvolvam o tema solicitado de forma completa
- Um segmento pode ser curto se for muito relevante, ou longo se precisar de contexto

Texto dividido em {len(sentences)} segmentos:
{full_text}

Responda APENAS com os números dos segmentos que devem ser MANTIDOS (ex: 1,3,5,7):"""
        
        try:
            # Guide Generator agora cria TODOS os segmentos - AI Editor fará a seleção
            logger.info(f"Guide Generator: Criando {len(sentences)} segmentos para AI Editor analisar")
            
            # Retornar TODOS os segmentos (sem pré-seleção) - AI Editor fará a seleção
            all_segments = []
            total_duration = 0
            
            for idx, sentence in enumerate(sentences):
                segment_duration = sentence["end"] - sentence["start"]
                total_duration += segment_duration
                
                all_segments.append({
                    "texto": sentence["text"],
                    "inicio": self._format_time(sentence["start"]),  # Tempo original no vídeo
                    "fim": self._format_time(sentence["end"]),      # Tempo original no vídeo
                    "duracao": f"{segment_duration:.1f}s",
                    "priority_score": 8.0,
                    "original_start": sentence["start"],  # Para uso no XML
                    "original_end": sentence["end"]       # Para uso no XML
                })
            
            logger.info(f"Todos os segmentos criados: {len(all_segments)} com duração total de {total_duration:.1f}s")
            
            # Retornar TODOS os segmentos - AI Editor fará a seleção final
            logger.info(f"Guide Generator: Retornando {len(all_segments)} segmentos para AI Editor")
            
            return all_segments
            
        except Exception as e:
            logger.error(f"Erro na seleção de segmentos pela IA: {str(e)}")
            # Fallback: retornar primeiros 3 segmentos
            return sentences[:3] if len(sentences) >= 3 else sentences

    def _create_segments_for_vsl(self, sentences: List[Dict], custom_instructions: str = None) -> List[Dict]:
        """Cria segmentos otimizados para VSL."""
        # Por enquanto, usar a mesma lógica geral
        return self._create_segments_general(sentences, custom_instructions)
    
    def _create_segments_for_youtube_live(self, sentences: List[Dict], custom_instructions: str = None) -> List[Dict]:
        """Cria segmentos otimizados para YouTube Live."""
        # Por enquanto, usar a mesma lógica geral
        return self._create_segments_general(sentences, custom_instructions)
    
    def generate_cutting_guide(self, transcription_path: str, video_type: str = "geral", custom_instructions: str = None) -> Dict[str, Any]:
        """Método público para gerar guia de cortes."""
        try:
            logger.info(f"Gerando guia de cortes para: {transcription_path}")
            
            # Carregar transcrição
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
            
            # Extrair palavras da transcrição
            words = self._extract_words_from_transcription(transcription_data)
            
            if not words:
                logger.warning("Nenhuma palavra com timestamp encontrada. Criando segmentos estimados.")
                # Usar fallback para criar segmentos estimados
                video_duration = transcription_data.get('duration', 600)  # Default 10 min
                text = transcription_data.get('text', '')
                words = self._create_fallback_words(text, video_duration)
            
            # Agrupar palavras em sentenças
            sentences = self._group_words_into_sentences(words)
            
            # Criar segmentos baseados no tipo de vídeo
            if video_type.lower() == "vsl":
                segments = self._create_segments_for_vsl(sentences, custom_instructions)
            elif video_type.lower() == "youtube_live":
                segments = self._create_segments_for_youtube_live(sentences, custom_instructions)
            else:
                segments = self._create_segments_general(sentences, custom_instructions)
            
            # Retornar resultado no formato esperado
            result = {
                "segmentos": segments
            }
            
            logger.info(f"Guia de cortes gerado com {len(segments)} segmentos")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao gerar guia de cortes: {str(e)}")
            raise

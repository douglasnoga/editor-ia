"""
AI Content Analysis Service.

This service handles GPT-4.1 integration for intelligent content analysis and editing
decisions based on video type templates and editing rules.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from ..models.transcription import TranscriptionSegment
from ..models.video import VideoType
from ..models.editing import EditingContext, EditingDecision, EditingResult
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class AIEditorError(Exception):
    """Custom exception for AI editor errors."""
    pass


class AIEditor:
    """
    AI-powered content analysis and editing service.
    
    Uses GPT-4.1 to analyze transcribed content and make intelligent editing
    decisions based on video type templates and editing rules.
    """
    
    def __init__(self):
        """Initialize the AI editor."""
        self.settings = get_settings()
        self.openai_client = None
        
        # Initialize OpenAI client if API key is available
        if self.settings.openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.ai_analysis_timeout
            )
    
    async def analyze_and_cut(self, transcript: List[TranscriptionSegment], 
                             video_type: VideoType, 
                             context: EditingContext) -> EditingResult:
        """
        Analyze transcript and generate editing decisions.
        
        Args:
            transcript: List of transcription segments
            video_type: Type of video for editing template
            context: Editing context with additional information
            
        Returns:
            EditingResult with decisions and selected segments
            
        Raises:
            AIEditorError: If analysis fails
        """
        try:
            if not self.openai_client:
                raise AIEditorError("OpenAI client not initialized")
            
            # Get editing rules for video type
            editing_rules = self._get_editing_rules(video_type)
            
            # Chunk transcript for GPT-4.1 context limits
            chunks = self._chunk_transcript(transcript, max_tokens=8000)
            
            all_decisions = []
            selected_segments = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Generate prompt for this chunk
                prompt = self._generate_prompt(chunk, editing_rules, context)
                
                # Call OpenAI API  
                logger.info(f"🔥 CALLING {self.settings.openai_model} for chunk {i+1}")
                response = await self._call_openai_api(prompt)
                logger.info(f"🔥 {self.settings.openai_model} RESPONSE RECEIVED for chunk {i+1}")
                
                # Parse response
                chunk_decisions = self._parse_ai_response(response, chunk, context)
                all_decisions.extend(chunk_decisions)
                
                # Log detailed decision analysis
                keep_count = len([d for d in chunk_decisions if d.decision_type == "keep"])
                remove_count = len([d for d in chunk_decisions if d.decision_type == "remove"])
                logger.info(f"🔍 CHUNK {i+1} DECISIONS: {keep_count} keep, {remove_count} remove")
                
                # Extract selected segments
                for decision in chunk_decisions:
                    logger.info(f"🔍 DECISION: {decision.segment_id} -> {decision.decision_type} (score: {decision.score})")
                    if decision.decision_type == "keep":
                        selected_segments.append(decision.segment_id)
            
            # Calculate final metrics
            final_duration = self._calculate_final_duration(transcript, selected_segments)
            compression_achieved = final_duration / context.original_duration
            
            # Generate editing result
            result = EditingResult(
                context=context,
                decisions=all_decisions,
                selected_segments=selected_segments,
                final_duration=final_duration,
                compression_achieved=compression_achieved,
                quality_score=self._calculate_quality_score(all_decisions),
                warnings=self._generate_warnings(all_decisions, context),
                stats=self._generate_stats(all_decisions, transcript)
            )
            
            logger.info(f"AI analysis completed: {len(selected_segments)} segments selected")
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            raise AIEditorError(f"AI analysis failed: {str(e)}")
    
    def _get_editing_rules(self, video_type: VideoType) -> Dict[str, Any]:
        """
        Get editing rules for the specified video type.
        
        Args:
            video_type: Type of video
            
        Returns:
            Dictionary with editing rules
        """
        # Base rules for all video types
        base_rules = {
            "remove_filler_words": ["né", "então", "cara", "tipo", "tá", "é", "humm"],
            "max_pause_ms": 800,
            "min_segment_length": 2.0,
            "hook_patterns": [
                r"\bo que é\b",
                r"\bpresta[m]?\s+atenção\b",
                r"\bexiste um artigo\b",
                r"\bquando você\b"
            ]
        }
        
        # Video type specific rules (IA-driven, no fixed targets)
        if video_type == VideoType.YOUTUBE_CUTS:
            base_rules.update({
                "suggested_duration_range": (360, 720),  # 6-12 minutes (sugestão)
                "content_focus": "quality_over_duration",
                "hook_weight": 3.0,
                "info_weight": 2.0,
                "required_functions": ["gancho", "desenvolvimento"]
            })
        elif video_type == VideoType.VSL:
            base_rules.update({
                "suggested_duration_range": (600, 1200),  # 10-20 minutes (sugestão)
                "content_focus": "conversion_quality",
                "hook_weight": 3.0,
                "proof_weight": 2.0,
                "required_functions": ["gancho", "dor", "solucao", "oferta", "garantia", "cta"]
            })
        elif video_type == VideoType.EDUCATIONAL:
            base_rules.update({
                "suggested_duration_range": (300, 900),   # 5-15 minutes (sugestão)
                "content_focus": "educational_clarity",
                "info_weight": 3.0,
                "required_functions": ["definicao", "exemplo_pratico", "conclusao"]
            })
        else:  # GENERAL
            base_rules.update({
                "suggested_duration_range": (180, 1800),  # 3-30 minutes (sugestão)
                "content_focus": "balanced_quality",
                "hook_weight": 2.0,
                "info_weight": 2.0
            })
        
        return base_rules
    
    def _chunk_transcript(self, transcript: List[TranscriptionSegment], 
                         max_tokens: int = 8000) -> List[List[TranscriptionSegment]]:
        """
        Chunk transcript into smaller pieces for GPT-4.1 context limits.
        
        Args:
            transcript: List of transcription segments
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of transcript chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for segment in transcript:
            # Rough estimation: 1 token per 4 characters
            segment_tokens = len(segment.text) // 4
            
            if current_tokens + segment_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_tokens = segment_tokens
            else:
                current_chunk.append(segment)
                current_tokens += segment_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _generate_prompt(self, chunk: List[TranscriptionSegment], 
                        editing_rules: Dict[str, Any], 
                        context: EditingContext) -> str:
        """
        Generate prompt for GPT-4.1 analysis.
        
        Args:
            chunk: Transcript chunk to analyze
            editing_rules: Editing rules for this video type
            context: Editing context
            
        Returns:
            Formatted prompt string
        """
        # Build transcript text
        transcript_text = "\n".join([
            f"ID: {seg.id} | [{seg.start:.1f}-{seg.end:.1f}s] {seg.text}"
            for seg in chunk
        ])
        
        # Build editing rules text
        rules_text = json.dumps(editing_rules, indent=2)
        
        prompt = f"""
Você é um editor de vídeo especializado em cortes inteligentes usando IA. 
Analise a transcrição abaixo e determine quais segmentos manter, remover ou modificar.

CONTEXTO DO VÍDEO:
- Tipo: {context.video_type}
- Duração original: {context.original_duration:.1f} segundos
- Idioma: {context.detected_language or 'pt-br'}
- Instruções customizadas: {context.custom_instructions or 'Nenhuma'}

REGRAS DE EDIÇÃO:
{rules_text}

TRANSCRIÇÃO:
{transcript_text}

INSTRUÇÕES:
1. Analise cada segmento da transcrição
2. Identifique ganchos, informações importantes e ruído
3. Atribua uma pontuação de 0-10 para cada segmento
4. Determine se deve manter, remover ou modificar cada segmento
5. Explique o raciocínio para cada decisão
6. IMPORTANTE: Use exatamente o ID mostrado (ex: segment_1, segment_2) no campo segment_id

FORMATO DE RESPOSTA (JSON):
{{
    "decisions": [
        {{
            "segment_id": "segment_1",
            "decision_type": "keep/remove/modify",
            "function": "gancho/desenvolvimento/exemplo_pratico/etc",
            "score": 8.5,
            "reasoning": "explicação da decisão",
            "confidence": 0.9
        }}
    ],
    "summary": {{
        "total_segments": 10,
        "kept_segments": 6,
        "estimated_final_duration": 420.0,
        "compression_ratio": 0.7
    }}
}}

Responda APENAS com o JSON, sem texto adicional.
"""
        
        return prompt
    
    async def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API with the generated prompt.
        
        Args:
            prompt: Prompt to send to GPT-4.1
            
        Returns:
            API response text
        """
        try:
            logger.info(f"🔍 OPENAI DEBUG: Using model: {self.settings.openai_model}")
            logger.info(f"🔍 OPENAI DEBUG: Model type: {type(self.settings.openai_model)}")
            logger.info(f"🔍 OPENAI DEBUG: Model repr: {repr(self.settings.openai_model)}")
            logger.info(f"🔍 OPENAI DEBUG: Prompt length: {len(prompt)} characters")
            # logger.error(f"🔥 PROMPT COMPLETO ENVIADO:\n{prompt}")  # Temporariamente comentado
            
            # Adjust parameters for o1 models (special handling)
            if self.settings.openai_model.startswith("o1"):
                # o1 models don't support system messages, temperature, or max_tokens
                response = await self.openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "user", "content": f"Você é um editor de vídeo especializado.\n\n{prompt}"}
                    ]
                )
            else:
                # Standard GPT models (including GPT-4.1)
                response = await self.openai_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": "Você é um editor de vídeo especializado."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
            
            result = response.choices[0].message.content
            logger.info(f"🔍 OPENAI DEBUG: Response received, length: {len(result) if result else 'None'}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            logger.error(f"🔍 OPENAI DEBUG: Exception type: {type(e).__name__}")
            raise AIEditorError(f"OpenAI API call failed: {str(e)}")
    
    def _parse_ai_response(self, response: str, 
                          chunk: List[TranscriptionSegment],
                          context: Optional[EditingContext] = None) -> List[EditingDecision]:
        """
        Parse AI response into editing decisions.
        
        Args:
            response: AI response text
            chunk: Original transcript chunk
            
        Returns:
            List of editing decisions
        """
        try:
            # Log the AI response for debugging
            logger.error(f"🔥 AI RESPONSE COMPLETA: {response}")
            logger.error(f"🔥 AI RESPONSE TIPO: {type(response)}")
            logger.error(f"🔥 AI RESPONSE TAMANHO: {len(response) if response else 'None'}")
            logger.info(f"🔍 AI RESPONSE DEBUG: Raw response (first 500 chars): {response[:500]}")
            logger.info(f"🔍 AI RESPONSE DEBUG: Response type: {type(response)}")
            logger.info(f"🔍 AI RESPONSE DEBUG: Response length: {len(response) if response else 'None'}")
            
            # Check if response is empty or None
            if not response or response.strip() == "":
                logger.error("AI response is empty or None")
                return self._create_fallback_decisions(chunk, context)
            
            # Parse JSON response
            data = json.loads(response)
            decisions = []
            
            for decision_data in data.get("decisions", []):
                segment_id = decision_data.get("segment_id")
                
                # Se a IA retornou um ID no formato [start-end], mapear para o ID correto
                if segment_id and segment_id.startswith('[') and segment_id.endswith(']'):
                    # Extrair timestamps do ID incorreto
                    timestamp_part = segment_id[1:-1]  # Remove [ ]
                    try:
                        start_time = float(timestamp_part.split('-')[0])
                        # Encontrar o segmento com timestamp mais próximo
                        matching_segment = None
                        min_diff = float('inf')
                        for seg in chunk:
                            diff = abs(seg.start - start_time)
                            if diff < min_diff:
                                min_diff = diff
                                matching_segment = seg
                        logger.info(f"🔄 MAPEANDO ID INCORRETO: {segment_id} → {matching_segment.id if matching_segment else 'None'}")
                    except:
                        matching_segment = None
                else:
                    # Busca normal por ID exato
                    matching_segment = None
                    for seg in chunk:
                        if seg.id == segment_id:
                            matching_segment = seg
                            break
                
                if not matching_segment:
                    logger.warning(f"No matching segment found for ID: {segment_id}")
                    continue
                
                # Create editing decision
                # Mapear valores de function da IA para valores válidos do enum
                raw_function = decision_data.get("function", "")
                function_mapping = {
                    'gancho': 'gancho',
                    'introdução': 'gancho',
                    'definição': 'definicao',
                    'desenvolvimento': 'desenvolvimento', 
                    'contextualização': 'desenvolvimento',
                    'contexto': 'desenvolvimento',
                    'exemplo_pratico': 'exemplo_pratico',
                    'exemplo': 'exemplo_pratico',
                    'explicação': 'exemplo_pratico',
                    'estatística': 'estatistica',
                    'estatistica': 'estatistica',
                    'transição': 'transicao',
                    'transicao': 'transicao',
                    'conclusão': 'conclusao',
                    'conclusao': 'conclusao',
                    'reforço': 'conclusao',
                    'finalização': 'conclusao',
                    'cta': 'cta',
                    'garantia': 'garantia',
                    'prova': 'prova',
                    'oferta': 'oferta',
                    'ruído': 'transicao',  # Mapear ruído para transição
                    'lateral': 'transicao'
                }
                
                # Extrair primeira parte e mapear
                first_part = raw_function.split('/')[0].lower().strip() if raw_function else ''
                function_value = function_mapping.get(first_part, 'desenvolvimento')  # Default para desenvolvimento
                
                decision = EditingDecision(
                    segment_id=segment_id,
                    decision_type=decision_data.get("decision_type", "remove"),
                    function=function_value,
                    score=decision_data.get("score", 0.0),
                    reasoning=decision_data.get("reasoning", ""),
                    confidence=decision_data.get("confidence", 0.5),
                    applied_rules=["ai_analysis"],
                    modifications={},
                    start_time=matching_segment.start,  # Timestamp real do segmento
                    end_time=matching_segment.end       # Timestamp real do segmento
                )
                
                decisions.append(decision)
            
            return decisions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {str(e)}")
            # Fallback: create default decisions
            return self._create_fallback_decisions(chunk, context)
        except Exception as e:
            logger.error(f"Error processing AI response: {str(e)}")
            return self._create_fallback_decisions(chunk, context)
    
    def _create_fallback_decisions(self, chunk: List[TranscriptionSegment], context: Optional[EditingContext] = None) -> List[EditingDecision]:
        """
        Create intelligent fallback decisions when AI response parsing fails.
        
        Args:
            chunk: Transcript chunk
            
        Returns:
            List of fallback decisions
        """
        decisions = []
        logger.warning(f"😨 AI EDITOR FALLBACK ATIVADO! Criando decisões fallback para {len(chunk)} segmentos")
        logger.info(f"🔄 Creating fallback decisions for {len(chunk)} segments")
        
        # Keywords that indicate important content
        important_keywords = [
            'preferência temporal', 'vender', 'marketing', 'oferta', 'cliente', 
            'compra', 'agora', 'urgência', 'desconto', 'promoção', 'estratégia',
            'vendas', 'negócio', 'dinheiro', 'lucro', 'resultado'
        ]
        
        # Palavras-chave específicas da instrução do usuário
        instruction_keywords = []
        if context and hasattr(context, 'custom_instructions') and context.custom_instructions:
            instruction_text = context.custom_instructions.lower()
            logger.info(f"🔍 FALLBACK: Analisando instruções customizadas: '{instruction_text}'")
            # Extrair palavras-chave das instruções
            instruction_words = instruction_text.split()
            instruction_keywords = [word for word in instruction_words if len(word) > 3]
            logger.info(f"🔍 FALLBACK: Palavras-chave extraídas das instruções: {instruction_keywords}")
            
            # Adicionar às palavras-chave importantes
            important_keywords.extend(instruction_keywords)
        
        # Calculate segment importance scores
        for segment in chunk:
            segment_duration = segment.end - segment.start
            text_lower = segment.text.lower()
            
            # Base score from content quality, not duration
            # IA decide relevância por conteúdo, não por tempo
            duration_score = 5.0  # Score neutro - foco no conteúdo
            
            # Content score based on keywords
            keyword_score = 0
            for keyword in important_keywords:
                if keyword in text_lower:
                    keyword_score += 2
            
            # Length score (very short segments are often filler words)
            length_score = min(3.0, len(segment.text) / 20)  # Max 3 points for 60+ chars
            
            # Confidence score (higher confidence = better transcription)
            confidence_score = segment.confidence * 3  # Max 3 points
            
            # Total score (capped at 10 to meet EditingDecision validation)
            raw_score = duration_score + keyword_score + length_score + confidence_score
            total_score = min(10.0, raw_score)  # Ensure score doesn't exceed 10
            
            # Decision logic - Threshold MUITO mais permissivo para garantir seleção
            decision_type = "keep" if total_score >= 2.0 else "remove"
            
            logger.info(f"🔄 FALLBACK DECISION: Segment {segment.id} - Score: {total_score:.1f} -> {decision_type}")
            
            # Use fixed confidence value of 0.3 as expected by tests
            confidence = 0.3  # Fixed confidence value for fallback decisions
            
            reasoning = f"Fallback analysis - Score: {total_score:.1f} (raw: {raw_score:.1f}, duration: {duration_score:.1f}, keywords: {keyword_score}, length: {length_score:.1f}, confidence: {confidence_score:.1f})"
            
            decision = EditingDecision(
                segment_id=segment.id,
                decision_type=decision_type,
                score=total_score,
                reasoning=reasoning,
                confidence=confidence,
                applied_rules=["fallback"],
                start_time=segment.start,  # Timestamp real do segmento
                end_time=segment.end       # Timestamp real do segmento
            )
            
            decisions.append(decision)
            logger.info(f"🔄 Segment {segment.id}: {decision_type} (score: {total_score:.1f})")
        
        # Ensure we keep at least some content for narrative flow
        kept_decisions = [d for d in decisions if d.decision_type == "keep"]
        if len(kept_decisions) < 3:
            # Keep top 3 scoring segments if we don't have enough
            decisions_by_score = sorted(decisions, key=lambda x: x.score, reverse=True)
            for i in range(min(3, len(decisions_by_score))):
                if decisions_by_score[i].decision_type == "remove":
                    decisions_by_score[i].decision_type = "keep"
                    decisions_by_score[i].reasoning += " (Kept for narrative flow)"
        
        kept_count = len([d for d in decisions if d.decision_type == "keep"])
        logger.info(f"🔄 Fallback decisions complete: {kept_count}/{len(decisions)} segments kept")
        
        return decisions
    
    def _calculate_final_duration(self, transcript: List[TranscriptionSegment], 
                                 selected_segments: List[str]) -> float:
        """
        Calculate final duration based on selected segments.
        
        Args:
            transcript: Original transcript
            selected_segments: List of selected segment IDs
            
        Returns:
            Final duration in seconds
        """
        total_duration = 0.0
        
        for segment in transcript:
            if segment.id in selected_segments:
                total_duration += segment.end - segment.start
        
        return total_duration
    
    def _calculate_quality_score(self, decisions: List[EditingDecision]) -> float:
        """
        Calculate overall quality score for the editing decisions.
        
        Args:
            decisions: List of editing decisions
            
        Returns:
            Quality score (0-10)
        """
        if not decisions:
            return 0.0
        
        # Average confidence weighted by decision scores
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for decision in decisions:
            if decision.decision_type == "keep":
                weight = decision.confidence
                total_weighted_score += decision.score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 5.0
    
    def _generate_warnings(self, decisions: List[EditingDecision], 
                          context: EditingContext) -> List[str]:
        """
        Generate warnings based on editing decisions.
        
        Args:
            decisions: List of editing decisions
            context: Editing context
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for low confidence decisions
        low_confidence_count = sum(1 for d in decisions if d.confidence < 0.5)
        if low_confidence_count > len(decisions) * 0.3:
            warnings.append(f"Muitas decisões com baixa confiança ({low_confidence_count})")
        
        # Check for very aggressive compression
        kept_decisions = [d for d in decisions if d.decision_type == "keep"]
        if len(kept_decisions) < len(decisions) * 0.1:
            warnings.append("Compressão muito agressiva pode afetar a qualidade")
        
        return warnings
    
    def _generate_stats(self, decisions: List[EditingDecision], 
                       transcript: List[TranscriptionSegment]) -> Dict[str, Any]:
        """
        Generate statistics about the editing process.
        
        Args:
            decisions: List of editing decisions
            transcript: Original transcript
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_segments": len(transcript),
            "total_decisions": len(decisions),
            "kept_segments": len([d for d in decisions if d.decision_type == "keep"]),
            "removed_segments": len([d for d in decisions if d.decision_type == "remove"]),
            "modified_segments": len([d for d in decisions if d.decision_type == "modify"]),
            "average_confidence": sum(d.confidence for d in decisions) / len(decisions) if decisions else 0.0,
            "average_score": sum(d.score for d in decisions) / len(decisions) if decisions else 0.0
        }
        
        return stats
    
    async def get_editing_preview(self, transcript: List[TranscriptionSegment], 
                                 video_type: VideoType) -> Dict[str, Any]:
        """
        Get a preview of editing decisions without full processing.
        
        Args:
            transcript: List of transcription segments
            video_type: Type of video
            
        Returns:
            Dictionary with preview information
        """
        try:
            # Get editing rules
            editing_rules = self._get_editing_rules(video_type)
            
            # Simple preview based on rules (IA-driven, no fixed targets)
            total_duration = sum(seg.end - seg.start for seg in transcript)
            suggested_range = editing_rules.get("suggested_duration_range", (300, 900))
            content_focus = editing_rules.get("content_focus", "balanced_quality")
            
            preview = {
                "original_duration": total_duration,
                "suggested_duration_range": suggested_range,
                "content_focus": content_focus,
                "total_segments": len(transcript),
                "ai_driven_selection": True,
                "editing_rules": editing_rules
            }
            
            return preview
            
        except Exception as e:
            logger.error(f"Error generating editing preview: {str(e)}")
            return {"error": str(e)}
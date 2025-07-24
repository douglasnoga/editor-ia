"""
Editing templates for different video types.

This module contains predefined editing rules and templates for various video types,
providing specialized configurations for optimal content editing.
"""

from typing import Dict, Any, List
from enum import Enum

class VideoTypeTemplate(Enum):
    """Video type templates with predefined editing rules."""
    
    YOUTUBE_CUTS = "youtube_cuts"
    VSL = "vsl"
    EDUCATIONAL = "educational"
    SOCIAL_REELS = "social_reels"
    ADVERTISING = "advertising"
    GENERAL = "general"


class EditingTemplates:
    """
    Collection of editing templates for different video types.
    
    Each template contains specific rules, patterns, and configurations
    optimized for the target video type and audience.
    """
    
    @staticmethod
    def get_template(template_type: VideoTypeTemplate) -> Dict[str, Any]:
        """
        Get editing template configuration for a specific video type.
        
        Args:
            template_type: Type of video template
            
        Returns:
            Dictionary with template configuration
        """
        templates = {
            VideoTypeTemplate.YOUTUBE_CUTS: EditingTemplates._get_youtube_cuts_template(),
            VideoTypeTemplate.VSL: EditingTemplates._get_vsl_template(),
            VideoTypeTemplate.EDUCATIONAL: EditingTemplates._get_educational_template(),
            VideoTypeTemplate.SOCIAL_REELS: EditingTemplates._get_social_reels_template(),
            VideoTypeTemplate.ADVERTISING: EditingTemplates._get_advertising_template(),
            VideoTypeTemplate.GENERAL: EditingTemplates._get_general_template()
        }
        
        return templates.get(template_type, EditingTemplates._get_general_template())
    
    @staticmethod
    def _get_youtube_cuts_template() -> Dict[str, Any]:
        """Template for YouTube live stream cuts."""
        return {
            "name": "Cortes de Live do YouTube",
            "description": "Otimizado para extrair os melhores momentos de lives",
            
            # Duration suggestions (IA decides based on content)
            "suggested_duration_range": (360, 720),  # 6-12 minutes (just a suggestion)
            "content_focus": "quality_over_duration",  # Focus on quality, not duration
            
            # Content weights
            "content_weights": {
                "gancho": 3.0,        # High priority for hooks
                "piada": 2.5,         # Jokes and funny moments
                "informacao": 2.0,    # Important information
                "interacao": 2.0,     # Chat interactions
                "reacao": 1.8,        # Reactions
                "exemplo": 1.5,       # Examples
                "transicao": 0.5      # Transitions
            },
            
            # Removal patterns
            "remove_patterns": [
                r"\b(né|então|cara|tipo|tá|é|humm|ahh)\b",  # Filler words
                r"aguarda aí|espera um pouquinho",          # Wait phrases
                r"vou beber água|deixa eu",                 # Personal actions
                r"pessoal do chat|galera"                   # Chat references (keep some)
            ],
            
            # Keep patterns (high priority)
            "keep_patterns": [
                r"presta[m]?\s+atenção",
                r"isso é importante",
                r"olha só isso",
                r"vocês sabiam que",
                r"deixa eu contar",
                r"aconteceu comigo"
            ],
            
            # Function requirements
            "required_functions": ["gancho", "desenvolvimento", "climax"],
            "optional_functions": ["introducao", "exemplo", "conclusao"],
            
            # Technical settings
            "max_pause_duration": 1.0,    # Max silence in seconds
            "min_segment_length": 3.0,    # Min segment duration
            "transition_buffer": 0.5,     # Buffer between cuts
            
            # Quality thresholds
            "min_segment_score": 6.0,     # Minimum score to keep
            "confidence_threshold": 0.7,   # Minimum confidence
            
            # Special rules
            "preserve_punchlines": True,
            "group_related_content": True,
            "maintain_narrative_flow": True
        }
    
    @staticmethod
    def _get_vsl_template() -> Dict[str, Any]:
        """Template for Video Sales Letters (VSL)."""
        return {
            "name": "Vídeo de Vendas (VSL)",
            "description": "Estrutura otimizada para conversão em vendas",
            
            # Duration suggestions (IA decide baseado no conteúdo)
            "suggested_duration_range": (600, 1200),  # 10-20 minutes (apenas sugestão)
            "content_focus": "conversion_quality",  # Foco em qualidade de conversão
            
            # Content weights (VSL funnel structure)
            "content_weights": {
                "gancho": 3.0,           # Hook - critical
                "dor": 2.8,              # Pain points
                "solucao": 2.8,          # Solution presentation
                "prova_social": 2.5,     # Social proof
                "oferta": 3.0,           # Offer presentation
                "urgencia": 2.5,         # Urgency/scarcity
                "garantia": 2.3,         # Guarantee
                "cta": 3.0,              # Call to action
                "objecoes": 2.0,         # Objection handling
                "historia": 2.0,         # Story telling
                "beneficios": 2.3,       # Benefits
                "caracteristicas": 1.5   # Features
            },
            
            # VSL-specific patterns
            "keep_patterns": [
                r"imagine se você",
                r"o que eu vou te mostrar",
                r"isso mudou minha vida",
                r"por apenas|somente",
                r"garantia de.*dias",
                r"últimas vagas",
                r"oferta especial",
                r"clique agora",
                r"não perca essa oportunidade"
            ],
            
            "remove_patterns": [
                r"como eu estava dizendo",
                r"enfim|resumindo",
                r"isso é bem simples"
            ],
            
            # Required VSL structure
            "required_functions": [
                "gancho", "dor", "solucao", "oferta", "garantia", "cta"
            ],
            "optional_functions": [
                "prova_social", "urgencia", "objecoes", "historia"
            ],
            
            # VSL-specific settings
            "max_pause_duration": 0.8,
            "min_segment_length": 5.0,
            "transition_buffer": 0.3,
            "min_segment_score": 7.0,
            "confidence_threshold": 0.8,
            
            # Special VSL rules
            "preserve_emotional_peaks": True,
            "maintain_sales_flow": True,
            "prioritize_conversion_elements": True,
            "ensure_cta_presence": True
        }
    
    @staticmethod
    def _get_educational_template() -> Dict[str, Any]:
        """Template for educational content."""
        return {
            "name": "Conteúdo Educacional",
            "description": "Focado em clareza e retenção de informação",
            
            # Duration suggestions (IA decide baseado no conteúdo)
            "suggested_duration_range": (300, 900),   # 5-15 minutes (apenas sugestão)
            "content_focus": "educational_clarity",  # Foco em clareza educacional
            
            # Content weights (educational priorities)
            "content_weights": {
                "definicao": 3.0,        # Definitions
                "exemplo_pratico": 2.8,   # Practical examples
                "explicacao": 2.5,        # Explanations
                "demonstracao": 2.5,      # Demonstrations
                "resumo": 2.3,            # Summaries
                "exercicio": 2.0,         # Exercises
                "dica": 2.0,              # Tips
                "contexto": 1.8,          # Context
                "introducao": 1.5,        # Introduction
                "transicao": 1.0          # Transitions
            },
            
            # Educational patterns
            "keep_patterns": [
                r"vamos entender",
                r"isso significa que",
                r"por exemplo",
                r"na prática",
                r"o conceito de",
                r"é importante saber",
                r"resumindo",
                r"para fixar"
            ],
            
            "remove_patterns": [
                r"como eu disse antes",
                r"obviamente|claramente",
                r"todo mundo sabe"
            ],
            
            # Required educational structure
            "required_functions": [
                "definicao", "exemplo_pratico", "explicacao", "resumo"
            ],
            "optional_functions": [
                "introducao", "exercicio", "dica", "demonstracao"
            ],
            
            # Educational settings
            "max_pause_duration": 1.2,    # Allow for thinking pauses
            "min_segment_length": 4.0,
            "transition_buffer": 0.8,
            "min_segment_score": 5.5,
            "confidence_threshold": 0.6,
            
            # Educational-specific rules
            "preserve_step_by_step": True,
            "maintain_logical_flow": True,
            "prioritize_clarity": True,
            "include_examples": True
        }
    
    @staticmethod
    def _get_social_reels_template() -> Dict[str, Any]:
        """Template for social media reels/shorts."""
        return {
            "name": "Reels para Redes Sociais",
            "description": "Conteúdo viral e engajante para redes sociais",
            
            # Duration suggestions (IA decide baseado no conteúdo)
            "suggested_duration_range": (15, 90),     # 15 seconds to 1.5 minutes (apenas sugestão)
            "content_focus": "viral_engagement",  # Foco em engajamento viral
            
            # Content weights (viral content priorities)
            "content_weights": {
                "gancho": 3.0,           # Hook is everything
                "climax": 3.0,           # Peak moment
                "surpresa": 2.8,         # Surprise elements
                "humor": 2.8,            # Humor
                "transformacao": 2.5,     # Before/after
                "trending": 2.5,         # Trending topics
                "call_to_action": 2.3,   # Engagement CTA
                "tip": 2.0,              # Quick tips
                "behind_scenes": 1.8     # Behind the scenes
            },
            
            # Viral patterns
            "keep_patterns": [
                r"espera até o final",
                r"você não vai acreditar",
                r"plot twist",
                r"dica rápida",
                r"segredo que ninguém conta",
                r"trending|viral",
                r"salva esse vídeo",
                r"compartilha com"
            ],
            
            "remove_patterns": [
                r"explicação longa",
                r"detalhadamente",
                r"vou explicar tudo"
            ],
            
            # Reel structure
            "required_functions": ["gancho", "climax"],
            "optional_functions": ["surpresa", "call_to_action"],
            
            # Social media settings
            "max_pause_duration": 0.3,    # Very fast-paced
            "min_segment_length": 1.0,    # Short segments
            "transition_buffer": 0.1,
            "min_segment_score": 8.0,     # Only the best content
            "confidence_threshold": 0.9,
            
            # Social media rules
            "ultra_aggressive_cutting": True,
            "preserve_viral_moments": True,
            "fast_paced_editing": True,
            "hook_within_3_seconds": True
        }
    
    @staticmethod
    def _get_advertising_template() -> Dict[str, Any]:
        """Template for advertising content."""
        return {
            "name": "Anúncios Publicitários",
            "description": "Otimizado para campanhas publicitárias",
            
            # Duration suggestions (IA decide baseado no conteúdo)
            "suggested_duration_range": (30, 120),    # 30 seconds to 2 minutes (apenas sugestão)
            "content_focus": "advertising_impact",  # Foco em impacto publicitário
            
            # Content weights (advertising priorities)
            "content_weights": {
                "gancho": 3.0,           # Hook
                "problema": 2.8,         # Problem identification
                "solucao": 2.8,          # Solution
                "beneficio": 2.5,        # Benefits
                "prova": 2.3,            # Proof/testimonial
                "oferta": 2.8,           # Offer
                "cta": 3.0,              # Call to action
                "urgencia": 2.5,         # Urgency
                "marca": 2.0             # Brand mention
            },
            
            # Advertising patterns
            "keep_patterns": [
                r"nova solução",
                r"revolucionário",
                r"comprovado cientificamente",
                r"resultado garantido",
                r"aproveite agora",
                r"oferta limitada",
                r"ligue já",
                r"acesse o site"
            ],
            
            "remove_patterns": [
                r"talvez|pode ser",
                r"não tenho certeza",
                r"acho que"
            ],
            
            # Advertising structure
            "required_functions": ["gancho", "problema", "solucao", "cta"],
            "optional_functions": ["beneficio", "prova", "oferta", "urgencia"],
            
            # Advertising settings
            "max_pause_duration": 0.5,
            "min_segment_length": 2.0,
            "transition_buffer": 0.2,
            "min_segment_score": 7.5,
            "confidence_threshold": 0.85,
            
            # Advertising rules
            "punchy_delivery": True,
            "clear_value_proposition": True,
            "strong_cta": True,
            "credibility_focus": True
        }
    
    @staticmethod
    def _get_general_template() -> Dict[str, Any]:
        """General template for unspecified content."""
        return {
            "name": "Geral",
            "description": "Template genérico adaptável",
            
            # Duration suggestions (IA decide baseado no conteúdo)
            "suggested_duration_range": (180, 1800),  # 3-30 minutes (apenas sugestão)
            "content_focus": "balanced_quality",  # Foco em qualidade equilibrada
            
            # Content weights (balanced)
            "content_weights": {
                "gancho": 2.5,
                "informacao": 2.0,
                "exemplo": 1.8,
                "explicacao": 1.8,
                "conclusao": 2.0,
                "transicao": 1.0
            },
            
            # General patterns
            "keep_patterns": [
                r"importante",
                r"interessante",
                r"por exemplo",
                r"resumindo"
            ],
            
            "remove_patterns": [
                r"\b(né|então|cara|tipo|tá|é|humm)\b"
            ],
            
            # Flexible structure
            "required_functions": [],
            "optional_functions": ["gancho", "desenvolvimento", "conclusao"],
            
            # General settings
            "max_pause_duration": 1.0,
            "min_segment_length": 3.0,
            "transition_buffer": 0.5,
            "min_segment_score": 5.0,
            "confidence_threshold": 0.6,
            
            # General rules
            "balanced_approach": True,
            "preserve_content_flow": True,
            "adaptive_compression": True
        }
    
    @staticmethod
    def get_template_list() -> List[Dict[str, str]]:
        """
        Get list of available templates with descriptions.
        
        Returns:
            List of template information dictionaries
        """
        templates = []
        for template_type in VideoTypeTemplate:
            template_config = EditingTemplates.get_template(template_type)
            templates.append({
                "id": template_type.value,
                "name": template_config["name"],
                "description": template_config["description"],
                "suggested_duration": f"{template_config.get('suggested_duration_range', (180, 1800))[0]}-{template_config.get('suggested_duration_range', (180, 1800))[1]}s",
                "content_focus": template_config.get('content_focus', 'balanced_quality')
            })
        
        return templates
    
    @staticmethod
    def validate_template(template_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate template configuration.
        
        Args:
            template_config: Template configuration to validate
            
        Returns:
            Validation result with errors and warnings
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = [
            "name", "description", "content_weights"
        ]
        
        for field in required_fields:
            if field not in template_config:
                result["errors"].append(f"Missing required field: {field}")
                result["valid"] = False
        
        # Validate optional ranges (now suggestions only)
        if "suggested_duration_range" in template_config:
            duration_range = template_config["suggested_duration_range"]
            if len(duration_range) != 2 or duration_range[0] >= duration_range[1]:
                result["warnings"].append("Invalid suggested_duration_range")
        
        # Warnings for missing optional fields
        optional_fields = ["keep_patterns", "remove_patterns", "required_functions", "suggested_duration_range", "content_focus"]
        for field in optional_fields:
            if field not in template_config:
                result["warnings"].append(f"Missing optional field: {field}")
        
        return result
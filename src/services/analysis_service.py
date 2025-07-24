import os
import json
import logging
from openai import OpenAI
from typing import Dict, Any, List

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("A chave da API da OpenAI não foi encontrada. Defina a variável de ambiente OPENAI_API_KEY.")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")  # Usa o modelo do .env ou 'gpt-4o' como padrão

    def _get_system_prompt(self, video_type: str, user_instructions: str) -> str:
        prompts = {
            "vsl": """### **Vídeo de Vendas (VSL) **

🏆 Objetivo

Gerar **um VSL persuasivo**, sem pausas mortas, seguindo a ordem clássica:
 `Gancho → Dor → Story → Solução → Oferta → Garantia → Prova → CTA`.

------

#### ⚙️ Config (ajuste se quiser)

```
MAX_PAUSE_MS       = 600           # silêncios acima disso são cortados
TARGET_SEG_LEN_SEC = 75            # 60‑90 s por sub‑segmento
COMPRESS_RANGE     = (0.25, 0.40)  # manter 60‑75 % do bruto
GAP_AT_JOIN_MS     = 150           # pausa inserida entre segmentos
MULETAS = [\"né\",\"então\",\"cara\",\"tipo\",\"é\",\"ééé\",\"humm\",\"olha só\"]
ANCHOR_REGEX = [
  r\"5 ?mil\", r\"8 ?mil\", r\"10 ?mil\",
  r\"vale[ra]?\\s+a\\s+pena\", r\"seria\\s+caro\"
]
```

------

#### 🔍 Regras de Corte

1. **Gancho visceral (obrigatório)**
   - Detecte primeira frase que contenha **dor extrema** ou **âncora de preço** (`ANCHOR_REGEX`).
   - Se aparecer depois de 30 s ➜ mover para 0 – 10 s.
2. **Eliminar sempre**
   - Saudação, clapboard (“gravando”), bastidor sem humor.
   - Pausas > `MAX_PAUSE_MS` (reduzir para 120‑200 ms).
   - Muletas da lista **MULETAS** + repetições imediatas (“esse é, esse é…”).
   - Vulgaridades que não agregam ênfase.
3. **Manter sempre**
   - Âncora de preço + contraste com oferta.
   - Promessa principal / grande ideia.
   - Story de autoridade (transformação pessoal) se for ≤ 2 min.
   - Garantia 100 % e prova social (depoimento, print de resultado).
   - CTA verbal sempre que preço é dito.
4. **Reordenar quando necessário**
   - `Gancho` p/ início.
   - `Garantia` deve vir **imediatamente após** revelar o preço real.
   - `Prova social externa` (frases marcadas `[[insert_testimonial]]` ou similares) deve seguir a garantia.
   - Objeções dispersas ➜ agrupar numa sequência única ≤ 2 min.
5. **Pontuação de Segmento**

```
hook  = 3  se contém ANCHOR_REGEX ou pergunta/dor forte
proof = 2  se contém garantia, depoimento, números concretos
noise = nº de tokens em MULETAS
score = hook + proof – noise
manter se score ≥ 3
```

1. **Agrupamento & Compressão**
   - Junte segmentos contíguos até `TARGET_SEG_LEN_SEC` (±30 s), sem quebrar ordem lógica.
   - Pare quando `kept_ratio` ∉ `COMPRESS_RANGE`.
2. **Transições**
   - Jump‑cut seco (gap=`GAP_AT_JOIN_MS`).
   - Para inserts externos marque `\"insert\": \"testimonial\"` ou `\"insert\": \"broll\"` em `comment`.

------

#### 📤 Formato de Resposta (único bloco JSON)

```json
{
  \"vsl_final\": {
    \"duracao\": \"15:12\",
    \"compressao\": 0.34,
    \"segmentos\": [
      {\"orig\": \"1279.0-1283.5\", \"funcao\": \"gancho\"},
      {\"orig\": \"1285.0-1305.8\", \"funcao\": \"oferta\"},
      {\"orig\": \"1447.2-1453.0\", \"funcao\": \"garantia\"},
      {\"insert\": \"testimonial\", \"at\": \"630.0\"},
      {\"orig\": \"1453.0-1458.0\", \"funcao\": \"cta\"}
    ]
  }
}
```

*Cada item `segmentos` pode ter `orig` **ou** `insert`.*
 *O sistema de pós‑processo converterá este JSON em markers XML.*

------

#### 🧩 Mini‑Exemplos (few‑shot)

*Reordenar Gancho*

```
Orig: \"… 50 s … Seria caro 5 mil reais? … 65 s … É por isso que…\"
Resp: primeiro segmento começa em 50 s (gancho), emenda 65 s (oferta).
```

*Inserção Externa*

```
Orig: [[insert_testimonial]]
Resp: gera marker {\"insert\":\"testimonial\",\"at\":CURRENT_TIME}
```

------

**Fim do prompt VSL.**""",
            "youtube_live_cut": """### PROMPT – **Corte de Live para YouTube**

🏆 Objetivo

Criar cortes autossuficientes para YouTube, com duração entre 6-12 minutos, focados em um único tema central, para quem nunca assistiu a live original. Como você não conhece a promessa do título/thumbnail, deve identificar temas importantes e criar ganchos impactantes.

------

#### 📋 Regras Essenciais

1. **Estrutura obrigatória:**
   - **Primeiros 30 segundos**: Conteúdo mais relevante e impactante sobre o tema (mesmo que esteja em outro ponto da live, mova para o início)
   - **Corpo**: Desenvolvimento claro e direto do tema (exemplos, explicações, pontos principais)
   - **Fechamento**: Conclusão que amarra o conteúdo

2. **Critérios de seleção:**
   - **Remover**: saudações, comentários momentâneos, brincadeiras contextuais da live, interações não relacionadas
   - **Manter**: conteúdo substancial sobre o tema, explicações, exemplos práticos, insights
   - **Priorizar**: segmentos que entregam valor concreto para quem nunca assistiu a live

3. **Duração e qualidade:**
   - **Total**: Entre 6-12 minutos (360-720 segundos)
   - **Cortes naturais**: Preservar frases completas, não cortar no meio do raciocínio
   - **Coesão**: Os segmentos devem formar um vídeo fluido como se tivesse sido gravado especificamente sobre o tema

4. **Mapeamento de Temas:**
   - Identifique temas que são **desenvolvidos o suficiente** na live para gerar um vídeo completo
   - Cada tema deve ter conteúdo substancial que justifique um vídeo independente (início, meio e fim)
   - Prefira temas com exemplos práticos, explicações detalhadas e conclusões claras
   - Se necessário, agrupe partes distantes da live que tratam do mesmo tema

5. **Lembre-se:**
   - Uma live pode gerar múltiplos cortes independentes, cada um com seu próprio tema
   - O vídeo é para quem nunca assistiu a live original e precisa de contexto completo
   - Editar como um profissional que está criando conteúdo independente a partir da live

------

#### 📝 Formato de Resposta

Retorne **apenas** um objeto JSON contendo a lista `"cortes_identificados"`. Cada corte deve ter segmentos que formem um vídeo coeso:

```json
{
  "cortes_identificados": [
    {
      "corte": 1,
      "tema": "Título descritivo do corte",
      "duracao": "08:30",
      "segmentos": [
        {"original": "335.0-370.0", "funcao": "gancho_impactante", "texto_completo": "Transcrição completa deste trecho do vídeo."},
        {"original": "375.0-485.0", "funcao": "desenvolvimento", "texto_completo": "Transcrição completa desta seção do vídeo."},
        {"original": "500.0-665.0", "funcao": "exemplo_pratico", "texto_completo": "Transcrição completa deste exemplo prático."},
        {"original": "670.0-795.0", "funcao": "conclusao", "texto_completo": "Transcrição completa da conclusão."}
      ]
    }
  ]
}
```

**IMPORTANTE:** 
- Os timestamps dos segmentos devem ser precisos para garantir cortes limpos no vídeo final
- O campo `texto_completo` deve conter a transcrição completa do segmento, não apenas as primeiras palavras
- NÃO inclua descrições de função ou rótulos no campo `texto_completo`

```
Orig: "… 85 s … vocês sabem o que é funil? … 110 s … Funil de vendas é …"
Resp: primeiro segmento começa em 85 s (gancho), depois 110 s (definição)
```

*Corte linear curto*

```
Orig: "olhem essa estatística…" (10 s bloco)
Resp: bloco mantido integral – vira gancho do corte
```

------
""",
        }
        # Se o tipo for 'geral' ou um tipo não encontrado, o comportamento é diferente.
        if video_type not in prompts:
            if user_instructions:
                # Adicionamos um contexto para a IA saber o que fazer.
                return f"Você é um assistente de IA para edição de vídeo. Analise a transcrição e siga estas instruções: {user_instructions}. Retorne apenas um JSON com uma chave 'cortes', que é uma lista de objetos, cada um com 'texto', 'inicio' e 'fim'."
            else:
                # Prompt geral padrão quando não há instruções.
                return "Você é um assistente de IA que analisa transcrições de vídeo. Sua tarefa é identificar os trechos mais impactantes e relevantes. Ignore introduções, enrolação e enceramentos. Foque em frases de efeito, perguntas retóricas e momentos de virada. Extraia pelo menos 7 cortes. Cada corte deve ter entre 20 e 60 segundos. Retorne apenas um JSON com uma chave 'cortes', que é uma lista de objetos, cada um com 'texto', 'inicio' e 'fim'."
        
        # Para VSL e Corte de Live, usamos o prompt específico.
        base_prompt = prompts[video_type]
        
        # As instruções do usuário são adicionadas ao final do prompt específico.
        if user_instructions:
            return f"{base_prompt}\n\nInstruções adicionais do usuário: {user_instructions}"
        
        return base_prompt

    def _find_closest_word(self, text_snippet: str, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encontra a palavra na transcrição que melhor corresponde ao início do trecho de texto."""
        for i, word_info in enumerate(words):
            # Constrói uma janela de texto a partir da transcrição para comparação
            window = " ".join(w['word'] for w in words[i:i+10])
            if text_snippet.lower().startswith(window[:len(text_snippet)].lower()):
                return word_info
        return words[0] # Fallback

    def _map_text_to_timestamps(self, cutting_guide_text: List[Dict[str, str]], transcription_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mapeia os trechos de texto do guia de cortes para timestamps precisos."""
        timed_cuts = []
        for cut in cutting_guide_text:
            start_text = cut.get('inicio', '')
            end_text = cut.get('fim', '')

            start_word = self._find_closest_word(start_text, transcription_words)
            end_word = self._find_closest_word(end_text, transcription_words)

            timed_cuts.append({
                "texto": cut.get('texto'),
                "start": start_word['start'],
                "end": end_word['end']
            })
        return timed_cuts

    async def generate_cutting_guide(self, transcription_result, video_type: str, user_instructions: str) -> dict:
        logger.info(f"Gerando guia de cortes com o modelo {self.model}.")
        # Agora tratando o resultado como objeto TranscriptionResult, não como dicionário
        transcription_text = transcription_result.text if hasattr(transcription_result, 'text') else ''
        if not transcription_text:
            logger.warning("Texto de transcrição vazio. Não é possível gerar o guia de cortes.")
            return {"cortes": []}

        system_prompt = self._get_system_prompt(video_type, user_instructions)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription_text}
                ],
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            cutting_guide_json = json.loads(response_content)
            logger.info("Guia de cortes recebido da OpenAI.")
            
            # Forçar formato correto baseado no tipo de vídeo
            if video_type == "youtube_live_cut" and 'cortes' in cutting_guide_json and not 'cortes_identificados' in cutting_guide_json:
                # Converter de formato genérico para o formato esperado pelo youtube_live_cut
                cortes_basic = cutting_guide_json.get('cortes', [])
                if cortes_basic and len(cortes_basic) > 0:
                    logger.info(f"Convertendo formato genérico para formato youtube_live_cut com {len(cortes_basic)} cortes")
                    cortes_identificados = [
                        {
                            "corte": i+1,
                            "tema": corte.get('texto', 'Corte sem tema')[:50],
                            "compressao": 0.75,
                            "duracao": "00:00",  # Será calculado depois
                            "gancho_pos": "00:00-00:15",
                            "segmentos": [
                                {
                                    "original": f"{corte.get('inicio')}-{corte.get('fim')}",
                                    "funcao": "desenvolvimento"
                                }
                            ]
                        } for i, corte in enumerate(cortes_basic)
                    ]
                    cutting_guide_json = {"cortes_identificados": cortes_identificados}
            
            # Verificação do formato do JSON
            if 'cortes' in cutting_guide_json:
                logger.info("Guia de cortes JSON recebido com chave 'cortes' e validado.")
            elif 'cortes_identificados' in cutting_guide_json:
                logger.info("Guia de cortes JSON recebido com chave 'cortes_identificados' e validado.")
                # Converter o formato cortes_identificados para o formato cortes
                new_cortes = []
                for corte in cutting_guide_json['cortes_identificados']:
                    for segmento in corte.get('segmentos', []):
                        if 'original' in segmento:
                            inicio, fim = segmento['original'].split('-')
                            new_cortes.append({
                                'texto': segmento.get('texto_completo', ''),
                                'inicio': inicio,
                                'fim': fim
                            })
                cutting_guide_json = {'cortes': new_cortes}
                logger.info(f"Convertidos {len(new_cortes)} segmentos do formato cortes_identificados para o formato cortes.")
            else:
                logger.warning("O JSON recebido da IA não contém nem a chave 'cortes' nem 'cortes_identificados'.")
                cutting_guide_json = {'cortes': []}

            return cutting_guide_json

        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar o JSON da OpenAI: {e}")
            logger.error(f"Resposta recebida: {response_content}")
            raise
        except Exception as e:
            logger.error(f"Erro ao gerar o guia de cortes: {e}", exc_info=True)
            raise

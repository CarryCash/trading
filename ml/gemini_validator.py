import os
import json
from google import genai
from pydantic import BaseModel
from typing import List, Optional

SYSTEM_PROMPT = """
Eres un copiloto de trading experto. Tu trabajo es validar señales de compra/venta
generadas por un modelo de Machine Learning. 

NUNCA debes ejecutar trades. Solo analizar y dar un SCORE de confianza (0-100).

MARCO DE REFERENCIA:
- Timeframe: 15 minutos (operaciones cortas)
- Activo: BTC/USDT (volatilidad alta)
- Capital: $15 con leverage 3x (RIESGO ALTO - sé conservador)

INPUTS QUE RECIBIRÁS:
1. Predicción del modelo: BUY (-1 = SELL, 0 = HOLD, 1 = BUY)
2. Confianza del modelo: 0.35-0.75 (qué tan seguro está)
3. Estado del mercado:
   - Precio close actual
   - RSI (0-100, <30 oversold, >70 overbought)
   - MACD (positivo/negativo, dirección)
   - Bandas Bollinger (precio vs upper/lower bands)
   - ATR (volatilidad)
   - Volumen (relativo a promedio)
   - SMA50, SMA200, EMA20 (tendencia)
   - Patrones de velas (hammer, engulfing, etc)

CRITERIOS PARA VALIDAR UNA SEÑAL BUY:
✓ Precio debe estar por debajo de SMA200 (tendencia alcista)
✓ Si RSI < 30, es más fuerte (oversold bounce)
✓ Si volumen es alto, confirma (dinero real)
✓ Si EMA20 > SMA50, es confluence alcista
✓ Si MACD es positivo, es más fuerte

❌ RECHAZAR (dar score <40) si:
✗ RSI > 70 (overbought, probable pullback)
✗ Precio arriba de SMA200 pero modelo predice BUY (contra-tendencia)
✗ Volumen es bajo (movimiento débil, fácil reversión)
✗ ATR muy alto (volatilidad extrema, riesgo)
✗ Banda de Bollinger superior está a <0.5% del precio (ya muy arriba)

CRITERIOS PARA VALIDAR UNA SEÑAL SELL:
(Invertir BUY: precio >SMA200, RSI>70, etc)

SCORING (0-100):
────────────────
80-100: EXCELENTE
  → Confluencia de 4+ indicadores
  → Tendencia fuerte
  → Volumen confirma
  → Bajo riesgo
  → EJECUTAR SIEMPRE

60-79: BUENO
  → Confluencia de 2-3 indicadores
  → Tendencia clara
  → Volumen moderado
  → Riesgo aceptable
  → EJECUTAR si confianza ML > 0.45

40-59: DÉBIL
  → Confluencia débil (solo 1-2 indicadores)
  → Tendencia no clara
  → Volumen bajo
  → Riesgo moderado-alto
  → CONSIDERAR (esperar más señales)

0-39: INVÁLIDO
  → Contra-tendencia
  → Indicadores conflictivos
  → Riesgo extremo
  → RECHAZAR (saltar trade)

Responde en formato JSON compatible con la siguiente estructura:
{
  "gemini_score": <0-100>,
  "gemini_decision": "<TRADE o SKIP>",
  "gemini_reasoning": "<Tu análisis en 1-2 frases>",
  "risk_level": "<LOW, MEDIUM, HIGH>",
  "rejection_reasons": [<array de razones si score<50, sino null>]
}
"""

class GeminiResponse(BaseModel):
    gemini_score: int
    gemini_decision: str
    gemini_reasoning: str
    risk_level: str
    rejection_reasons: Optional[List[str]] = None

class GeminiValidator:
    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            
        self.system_prompt = SYSTEM_PROMPT
        
    def prepare_market_context(self, row):
        context = {
            'close': row.get('close', 0),
            'rsi': row.get('rsi_14', 50),
            'macd': row.get('macd', 0),
            'bb_upper': row.get('bb_upper', 0),
            'bb_lower': row.get('bb_lower', 0),
            'bb_position': 0.5,
            'atr': row.get('atr', 0),
            'volume_sma_ratio': row.get('volume_sma_ratio', 1.0),
            'sma_50': row.get('sma_50', 0),
            'sma_200': row.get('sma_200', 0),
            'ema_20': row.get('ema_20', 0),
        }
        
        # Derived
        if context['bb_upper'] - context['bb_lower'] > 0:
            context['bb_position'] = (context['close'] - context['bb_lower']) / (context['bb_upper'] - context['bb_lower'])
        
        context['price_above_200sma'] = context['close'] > context['sma_200']
        context['sma_cross'] = context['sma_50'] > context['sma_200']
        context['ema_above_sma'] = context['ema_20'] > context['sma_50']
        context['macd_positive'] = context['macd'] > 0
        
        patterns = []
        if row.get('pattern_hammer', 0): patterns.append('hammer')
        if row.get('pattern_engulfing', 0) == 1: patterns.append('bullish engulfing')
        if row.get('pattern_engulfing', 0) == -1: patterns.append('bearish engulfing')
        context['patterns'] = patterns
        
        return context

    def _mock_gemini_response(self, prediction, confidence, context):
        score = int(confidence * 100)
        decision = "TRADE" if score >= 65 else "SKIP"
        reasons = [] if decision == "TRADE" else ["Low model confidence mock"]
        
        return {
            "gemini_score": score,
            "gemini_decision": decision,
            "gemini_reasoning": "Mock reasoning for testing without API key.",
            "risk_level": "MEDIUM",
            "rejection_reasons": reasons
        }

    def call_gemini(self, prediction, confidence, market_context, retries=3):
        if not self.client:
            return self._mock_gemini_response(prediction, confidence, market_context)
            
        user_prompt = f"""
        Analiza esta señal de trading:
        
        PREDICCIÓN DEL MODELO:
        - Acción: {prediction}
        - Confianza: {confidence:.1%}
        
        ESTADO DEL MERCADO (Timeframe 15m):
        - Precio Close: ${market_context['close']:.2f}
        - RSI: {market_context['rsi']:.0f} (referencia: <30 oversold, >70 overbought)
        - MACD: {market_context['macd']:.4f} ({'positivo' if market_context['macd_positive'] else 'negativo'})
        - Bollinger Bands: Lower=${market_context['bb_lower']:.2f}, Upper=${market_context['bb_upper']:.2f}
        - Posición en BB: {market_context['bb_position']:.1%} (0=lower, 1=upper)
        - ATR (volatilidad): ${market_context['atr']:.2f}
        - Volumen: {market_context['volume_sma_ratio']:.2f}x promedio
        - SMA50: ${market_context['sma_50']:.2f}
        - SMA200: ${market_context['sma_200']:.2f}
        - EMA20: ${market_context['ema_20']:.2f}
        - Precio arriba de SMA200: {'Sí' if market_context['price_above_200sma'] else 'No'}
        - Cruce SMA (50>200): {'Bullish' if market_context['sma_cross'] else 'Bearish'}
        - Patrones de velas: {', '.join(market_context['patterns']) or 'Ninguno'}
        """
        
        import time
        attempt = 0
        while attempt < retries:
            try:
                response = self.client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=[
                        self.system_prompt,
                        user_prompt
                    ],
                    config={
                        'temperature': 0.1,
                        'response_mime_type': 'application/json',
                        'response_schema': GeminiResponse,
                    }
                )
                
                # The response is guaranteed to be a JSON matching the schema
                result = json.loads(response.text)
                return result
                
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                print(f"Error calling Gemini (Attempt {attempt}/{retries}): {error_msg}")
                if attempt < retries:
                    # Exponential backoff: wait 4, 8, 16 seconds...
                    sleep_time = 2 ** (attempt + 1)
                    print(f"Waiting {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
                else:
                    return {
                        'gemini_score': 50, 
                        'gemini_decision': 'SKIP', 
                        'gemini_reasoning': f'Parse/API error after {retries} attempts: {error_msg}',
                        'risk_level': 'HIGH',
                        'rejection_reasons': ['API Error Limit']
                    }

    def should_trade(self, gemini_score, ml_confidence, threshold_score=60, threshold_conf=0.40):
        return (gemini_score >= threshold_score and ml_confidence >= threshold_conf)

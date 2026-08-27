"""
GeminiValidatorV2 — Scorer local para BACKTESTING + cliente Gemini para PRODUCCIÓN.

Arquitectura correcta:
  - Backtest:    score_local(row, prediction, confidence)
                 → Reglas determinísticas en Python. Sin API. Instantáneo.
  - Producción:  call_gemini(...)
                 → 1 llamada real por señal (~1 cada 15m). Bien dentro del tier gratuito.

El scorer local implementa exactamente las mismas reglas del SYSTEM_PROMPT_V2,
permitiendo backtests reproducibles y sin colapsar la API.
"""

import os
import json
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────
# Sistema de scoring local (para backtesting)
# ─────────────────────────────────────────────────────────────

def score_local(prediction: str, confidence: float, ctx: Dict[str, Any]) -> dict:
    """
    Réplica determinística del SYSTEM_PROMPT_V2.
    Igual de rápido que un dict lookup — sin red, sin costos.

    Parámetros
    ----------
    prediction  : 'BUY' o 'SELL'
    confidence  : float confianza del modelo ML (0-1)
    ctx         : dict devuelto por prepare_market_context()

    Retorna
    -------
    dict con gemini_score, gemini_decision, gemini_reasoning, risk_level, rejection_reasons
    """
    rsi              = ctx.get('rsi', 50)
    macd_pos         = ctx.get('macd_positive', False)
    price_vs_sma200  = ctx.get('price_above_200sma', False)
    sma_cross        = ctx.get('sma_cross', False)       # SMA50 > SMA200
    ema_above_sma    = ctx.get('ema_above_sma', False)   # EMA20 > SMA50
    vol_ratio        = ctx.get('volume_sma_ratio', 1.0)
    bb_pos           = ctx.get('bb_position', 0.5)

    score = 0
    reasons_pos = []
    reasons_neg = []

    if prediction == 'BUY':
        if rsi < 45:
            score += 20; reasons_pos.append(f'RSI {rsi:.0f} < 45 (espacio para subir)')
        if macd_pos:
            score += 15; reasons_pos.append('MACD positivo')
        if not price_vs_sma200:
            score += 15; reasons_pos.append('Precio bajo SMA200 (tendencia alcista)')
        if ema_above_sma:
            score += 10; reasons_pos.append('EMA20 > SMA50 (momentum)')
        if vol_ratio >= 0.8:
            score += 10; reasons_pos.append(f'Volumen {vol_ratio:.1f}x OK')
        if bb_pos < 0.85:
            score += 10; reasons_pos.append('No en sobrecompra extrema de BB')
        if price_vs_sma200:
            score += 20; reasons_pos.append('Precio sobre SMA200 (confirmacion bullish)')
        # Penalizaciones
        if rsi > 80:
            score -= 20; reasons_neg.append(f'RSI {rsi:.0f} > 80 (sobrecompra extrema)')
        if vol_ratio < 0.5:
            score -= 10; reasons_neg.append(f'Volumen muy bajo ({vol_ratio:.1f}x)')
        if not sma_cross:
            score -= 5;  reasons_neg.append('SMA50 < SMA200 (tendencia bearish)')

    elif prediction == 'SELL':
        if rsi > 55:
            score += 20; reasons_pos.append(f'RSI {rsi:.0f} > 55 (espacio para bajar)')
        if not macd_pos:
            score += 15; reasons_pos.append('MACD negativo')
        if price_vs_sma200:
            score += 15; reasons_pos.append('Precio sobre SMA200 (tendencia bajista favorece SELL)')
        if not ema_above_sma:
            score += 10; reasons_pos.append('EMA20 < SMA50 (momentum bajista)')
        if vol_ratio >= 0.8:
            score += 10; reasons_pos.append(f'Volumen {vol_ratio:.1f}x OK')
        if bb_pos > 0.15:
            score += 10; reasons_pos.append('No en sobreventa extrema de BB')
        if not price_vs_sma200:
            score += 20; reasons_pos.append('Precio bajo SMA200 (confirmacion bearish)')
        # Penalizaciones
        if rsi < 20:
            score -= 20; reasons_neg.append(f'RSI {rsi:.0f} < 20 (sobreventa extrema)')
        if vol_ratio < 0.5:
            score -= 10; reasons_neg.append(f'Volumen muy bajo ({vol_ratio:.1f}x)')
        if sma_cross:
            score -= 5;  reasons_neg.append('SMA50 > SMA200 (tendencia bullish)')

    score = max(0, min(score, 100))   # clamp 0-100

    # Decision y risk level
    if score >= 75:
        decision   = 'TRADE'
        risk_level = 'LOW'
    elif score >= 50:
        decision   = 'TRADE'
        risk_level = 'MEDIUM'
    elif score >= 25:
        decision   = 'SKIP'
        risk_level = 'MEDIUM'
    else:
        decision   = 'SKIP'
        risk_level = 'HIGH'

    pos_str = ', '.join(reasons_pos) if reasons_pos else 'sin confluence'
    neg_str = '; penalizaciones: ' + ', '.join(reasons_neg) if reasons_neg else ''
    reasoning = f"{prediction}: {pos_str}{neg_str}. Score {score}/100."

    return {
        'gemini_score':      score,
        'gemini_decision':   decision,
        'gemini_reasoning':  reasoning,
        'risk_level':        risk_level,
        'rejection_reasons': reasons_neg if decision == 'SKIP' else None
    }


# ─────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────

class GeminiValidatorV2:
    """
    Validator v2 con scorer local para backtest y cliente Gemini para producción.
    """

    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = api_key
        self.client  = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                print("google-genai no instalado. Solo modo local disponible.")

    # ── Contexto de mercado ────────────────────────────────────

    def prepare_market_context(self, row) -> dict:
        # Helper to safely get a value from a row (dict or pandas Series)
        def g(key, default=0):
            if hasattr(row, 'get'):
                return row.get(key, default) or default
            return getattr(row, key, default) if hasattr(row, key) else default

        ctx = {
            'close':            g('close'),
            'rsi':              g('rsi', 50),       # features_engineered.csv uses 'rsi'
            'macd':             g('macd'),
            'bb_upper':         g('bb_upper'),
            'bb_lower':         g('bb_lower'),
            'bb_position':      0.5,
            'atr':              g('atr'),
            # features_engineered.csv has 'volume_spike_ratio', not 'volume_sma_ratio'
            'volume_sma_ratio': g('volume_spike_ratio', g('volume_sma_ratio', 1.0)),
            'sma_50':           g('sma_50'),
            'sma_200':          g('sma_200'),
            'ema_20':           g('ema_20'),
        }

        bb_range = ctx['bb_upper'] - ctx['bb_lower']
        if bb_range > 0:
            ctx['bb_position'] = (ctx['close'] - ctx['bb_lower']) / bb_range

        # Use pre-computed boolean columns if available, else derive them
        ctx['price_above_200sma'] = bool(g('price_above_200sma', int(ctx['close'] > ctx['sma_200'])))
        ctx['sma_cross']          = bool(g('sma_cross',          int(ctx['sma_50']  > ctx['sma_200'])))
        ctx['ema_above_sma']      = bool(g('ema_above_sma',      int(ctx['ema_20']  > ctx['sma_50'])))
        ctx['macd_positive']      = ctx['macd'] > 0

        patterns = []
        if g('pattern_hammer'):   patterns.append('hammer')
        eng = g('pattern_engulfing')
        if eng == 1:  patterns.append('bullish engulfing')
        if eng == -1: patterns.append('bearish engulfing')
        ctx['patterns'] = patterns
        return ctx

    # ── Scorer local (backtest) ────────────────────────────────

    def validate_local(self, prediction: str, confidence: float, market_context: dict) -> dict:
        """Scorer determinístico — sin API. Usa esto para backtesting."""
        return score_local(prediction, confidence, market_context)

    # ── Cliente Gemini (producción) ────────────────────────────

    def call_gemini(self, prediction: str, confidence: float,
                    market_context: dict, retries: int = 3) -> dict:
        """
        Llama a Gemini 2.5 Flash. Solo para PRODUCCIÓN (1 señal cada ~15m).
        Para backtesting usa validate_local() en su lugar.
        """
        if not self.client:
            print("  [WARN] Sin API key — usando scorer local como fallback.")
            return self.validate_local(prediction, confidence, market_context)

        ctx = market_context
        user_prompt = f"""
Señal de trading a validar:

MODELO: {prediction} | Confianza: {confidence:.1%}

MERCADO (BTC/USDT 15m):
- Close:  ${ctx['close']:.2f}
- RSI:    {ctx['rsi']:.0f}
- MACD:   {ctx['macd']:.4f} ({'pos' if ctx['macd_positive'] else 'neg'})
- BB pos: {ctx['bb_position']:.2f}  (vol: {ctx['volume_sma_ratio']:.2f}x)
- SMA50={ctx['sma_50']:.0f}  SMA200={ctx['sma_200']:.0f}  EMA20={ctx['ema_20']:.0f}
- Sobre SMA200: {'Si' if ctx['price_above_200sma'] else 'No'}
- SMA cross (50>200): {'Bullish' if ctx['sma_cross'] else 'Bearish'}
- Patrones: {', '.join(ctx['patterns']) or 'ninguno'}

Responde SOLO en JSON.
"""
        from pydantic import BaseModel
        from typing import List, Optional as Opt

        class _Resp(BaseModel):
            gemini_score:      int
            gemini_decision:   str
            gemini_reasoning:  str
            risk_level:        str
            rejection_reasons: Opt[List[str]] = None

        system = open(os.path.join(os.path.dirname(__file__),
                                   'gemini_system_prompt.txt')).read() \
                 if os.path.exists(os.path.join(os.path.dirname(__file__),
                                                'gemini_system_prompt.txt')) \
                 else "Eres un validador de señales de trading. Responde en JSON."

        for attempt in range(1, retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[system, user_prompt],
                    config={
                        'temperature': 0.1,
                        'response_mime_type': 'application/json',
                        'response_schema': _Resp,
                    }
                )
                return json.loads(resp.text)
            except Exception as e:
                print(f"  [Gemini error {attempt}/{retries}] {e}")
                if attempt < retries:
                    wait = 2 ** (attempt + 1)
                    print(f"  Esperando {wait}s...")
                    time.sleep(wait)
                else:
                    # Fallback al scorer local si la API falla
                    print("  Fallback a scorer local.")
                    return self.validate_local(prediction, confidence, market_context)

    # ── Decisión de trading ────────────────────────────────────

    def should_trade(self, gemini_score: int, ml_confidence: float) -> bool:
        """
        Lógica tri-nivel:
          75-100 → TRADE siempre
          50- 74 → TRADE si ML confidence >= 0.40
          25- 49 → TRADE si ML confidence >= 0.55
           0- 24 → SKIP siempre
        """
        if gemini_score >= 75:
            return True
        if gemini_score >= 50 and ml_confidence >= 0.40:
            return True
        if gemini_score >= 25 and ml_confidence >= 0.55:
            return True
        return False

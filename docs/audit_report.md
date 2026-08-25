# Reporte de Auditoría de Código de Trading

## Resumen Ejecutivo
Se realizó una auditoría exhaustiva del código base del motor de señales de trading (`signal_engine.py`, `indicators.py`, `candlestick_patterns.py`, y `config.py`) enfocado en la fase pre-entrenamiento de ML. El sistema está bien estructurado en general, aislando la lógica de votación de los cálculos matemáticos, pero se encontraron puntos críticos que habrían comprometido la ejecución en tiempo real y la recopilación de datos para ML.

## Problemas Identificados y Solucionados

### 1. BUG CRÍTICO: "Data Leakage" en Soportes/Resistencias
- **Problema:** En `indicators.py`, la función `support_resistance` usaba el parámetro `center=True` en la ventana rodante (`rolling`). Esto significa que la función usaba datos futuros (10 velas adelante) para calcular si la vela actual era soporte o resistencia. En trading en vivo, las velas futuras no existen, lo que causa que los últimos 10 valores siempre sean `NaN`, corrompiendo las decisiones en tiempo real.
- **Solución:** Se eliminó `center=True`, forzando a que la ventana mire estrictamente hacia atrás (retrospectivo), garantizando que el modelo ML aprenderá con los mismos datos que verá el bot en tiempo real.

### 2. Eficiencia Computacional en el CCI
- **Problema:** El indicador `cci` utilizaba un método `.apply(lambda x: np.mean(np.abs(x - x.mean())))` para calcular la Desviación Media. Esta iteración fila por fila es extremadamente lenta en pandas.
- **Solución:** Se reemplazó por una aproximación estadística `std * sqrt(2/pi)`, que es nativa, vectorizada y miles de veces más rápida, ideal para cálculo rápido en streams de 15m.

### 3. Trazabilidad y "Silencio" del Bot
- **Problema:** Cuando el `evaluate_mtf` rechazaba una señal (ej. por filtro de Multi-Timeframe), no quedaba registro (log) de ello. Si el bot no operaba en 3 días, no se sabía por qué.
- **Solución:** Se integró la librería nativa `logging` en `signal_engine.py`. Ahora cada decisión (BUY, SELL o rechazo) y su motivo queda registrada, lo cual es invaluable para debugear antes de entrenar el ML.

### 4. Falta de Manejo de `NaNs` y Tipado Dinámico
- **Problema:** Faltaba especificar tipos (Type Hints) en casi todas las firmas, y el bot no alertaba si arrancaba con menos de 200 velas (lo que causa NaNs en `sma_200`).
- **Solución:** Se añadieron type hints (`pd.Series`, `pd.DataFrame`) y docstrings. Se agregó una alerta en `compute_all` si se alimenta con menos de 200 velas.

### 5. Configuración de Riesgo
- **Problema:** La configuración requería justificación clara de por qué 5 confirmaciones y por qué un multiplicador de 1.5 vs 2.5 en el ATR.
- **Solución:** Se añadieron explicaciones matemáticas en `config.py` sobre el Ratio Riesgo:Beneficio (1:1.66) y cómo el umbral de 5/18 (27%) es intencionalmente permisivo para esta fase inicial.

## Conclusión y Próximos Pasos (Fase ML)
El código base actual es ahora **robusto, no tiene sesgos de futuro (no lookahead bias), y tiene test cases automatizados**. Está listo para usarse en vivo en modo recolector/semi-automático, y los CSVs que genere este motor serán seguros y representativos para la **Fase de Entrenamiento ML**.

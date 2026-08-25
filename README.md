# ProyectoDawn - Motor de Trading

## Visión General
Este es el motor de señales del bot de trading de criptomonedas. Está diseñado para operar inicialmente de manera semi-automática o automática usando estrategias clásicas, recopilando datos de decisiones para luego entrenar un modelo de Machine Learning (ML).

## Estructura del Proyecto
- `config.py`: Configuraciones globales de capital, temporalidad, umbrales y API Keys (vía `.env`).
- `core/indicators.py`: Cálculo de ~18 indicadores técnicos vectorizados con Pandas.
- `core/candlestick_patterns.py`: Detección de patrones de acción del precio (Hammer, Engulfing, etc).
- `strategy/signal_engine.py`: Motor de votación multicriterio. Recopila los votos de cada indicador.
- `tests/`: Pruebas unitarias para asegurar que los cálculos y la lógica no se rompan.

## Lógica de Decisión (Umbrales)
El `signal_engine` cuenta con un sistema democrático. Las 18 herramientas "votan" `+1` (Alcista), `-1` (Bajista) o `0` (Neutral).
Para que el bot genere una señal de compra o venta, los votos a favor deben superar el `MIN_CONFIRMATIONS_TO_TRADE` definido en `config.py` (actualmente 5). 

Además, se aplica un filtro Multi-Timeframe (MTF): si el ADX o la SMA de un marco temporal mayor (ej. 4H) contradicen la señal de 15m, la señal **se rechaza**.

## Cómo Agregar una Nueva Herramienta
Si deseas agregar un nuevo indicador o patrón y que este vote en las decisiones, sigue estos 3 pasos:

1. **Crear el cálculo matemático**: Agrega la función a `core/indicators.py` o `core/candlestick_patterns.py`.
2. **Exponer la columna**: Si es un indicador, agrégalo a la función `compute_all(df)` dentro de `indicators.py`.
3. **Agrega su Voto**: En `strategy/signal_engine.py`, dirígete a la función de la categoría correcta (ej. `_vote_momentum`) y agrega su lógica para asignar `+1`, `-1` o `0` al diccionario de votos. Automáticamente pasará a ser contado por el motor.

## Pruebas
Antes de subir cualquier cambio, asegúrate de correr los tests:
```bash
pytest tests/
```
Esto prevendrá "data leakage" y errores de lógica.

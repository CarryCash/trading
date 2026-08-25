"""
Dashboard en vivo para el bot de trading de Futures.
Proceso independiente — se comunica con el bot vía data/live_state.json.

Uso:
    python run_dash.py

Abre http://localhost:8050 en tu navegador.
"""
# pyrefly: ignore [missing-import]
import dash
# pyrefly: ignore [missing-import]
from dash import dcc, html
# pyrefly: ignore [missing-import]
from dash.dependencies import Input, Output
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots

from core.live_state import read_state

# ─── Constantes de estilo ─────────────────────────────────────────────
BG_DARK = "#131722"
BG_CARD = "#1e222d"
BG_BORDER = "#2a2e39"
TEXT_PRIMARY = "#d1d4dc"
TEXT_SECONDARY = "#787b86"
COLOR_BULL = "#26a69a"
COLOR_BEAR = "#ef5350"
COLOR_BLUE = "#2962ff"
COLOR_YELLOW = "#f0b90b"
COLOR_CYAN = "#00bcd4"

# ─── App Dash ──────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="ProyectoDawn — Trading Dashboard")

app.layout = html.Div(
    style={"backgroundColor": BG_DARK, "color": TEXT_PRIMARY,
           "fontFamily": '-apple-system, "Segoe UI", Roboto, sans-serif',
           "minHeight": "100vh", "padding": "0", "margin": "0"},
    children=[
        # Auto-refresh cada 2.5 segundos
        dcc.Interval(id="refresh", interval=2500, n_intervals=0),

        # ── BARRA SUPERIOR ──
        html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "center", "padding": "12px 20px",
                   "backgroundColor": BG_CARD, "borderBottom": f"1px solid {BG_BORDER}"},
            children=[
                html.Div(id="header-symbol",
                         style={"fontSize": "18px", "fontWeight": "bold"}),
                html.Div(id="header-mode",
                         style={"fontSize": "14px", "padding": "4px 12px",
                                "borderRadius": "4px", "backgroundColor": COLOR_BLUE}),
                html.Div(id="header-time",
                         style={"fontSize": "14px", "color": TEXT_SECONDARY}),
                html.Div(id="header-capital",
                         style={"fontSize": "14px", "fontWeight": "bold"}),
            ]
        ),

        # ── CUERPO: gráfico principal + panel lateral ──
        html.Div(
            style={"display": "flex", "gap": "0", "height": "calc(100vh - 100px)"},
            children=[
                # Columna izquierda: gráficos (70%)
                html.Div(
                    style={"flex": "7", "display": "flex", "flexDirection": "column"},
                    children=[
                        # Gráfico de velas
                        dcc.Graph(id="candlestick-chart",
                                  style={"flex": "3"},
                                  config={"displayModeBar": True, "scrollZoom": True}),
                        # Curva de equity
                        dcc.Graph(id="equity-chart",
                                  style={"flex": "1"},
                                  config={"displayModeBar": False}),
                    ]
                ),

                # Columna derecha: paneles de información (30%)
                html.Div(
                    style={"flex": "3", "overflowY": "auto",
                           "borderLeft": f"1px solid {BG_BORDER}",
                           "padding": "12px", "display": "flex",
                           "flexDirection": "column", "gap": "12px"},
                    children=[
                        # Tarjeta: Última Señal
                        html.Div(
                            style=_card_style(),
                            children=[
                                html.H3("Última Señal",
                                        style={"margin": "0 0 8px 0", "fontSize": "14px",
                                               "color": TEXT_SECONDARY}),
                                html.Div(id="signal-action",
                                         style={"fontSize": "28px", "fontWeight": "bold",
                                                "marginBottom": "4px"}),
                                html.Div(id="signal-confirmations",
                                         style={"fontSize": "13px", "color": TEXT_SECONDARY,
                                                "marginBottom": "8px"}),
                                html.Div(id="signal-reasons",
                                         style={"fontSize": "11px", "color": TEXT_SECONDARY,
                                                "lineHeight": "1.6", "maxHeight": "120px",
                                                "overflowY": "auto"}),
                            ]
                        ),

                        # Tarjeta: Posición Abierta
                        html.Div(
                            style=_card_style(),
                            children=[
                                html.H3("Posición Abierta",
                                        style={"margin": "0 0 8px 0", "fontSize": "14px",
                                               "color": TEXT_SECONDARY}),
                                html.Div(id="position-info",
                                         style={"fontSize": "13px", "lineHeight": "1.8"}),
                            ]
                        ),

                        # Tarjeta: Sesión
                        html.Div(
                            style=_card_style(),
                            children=[
                                html.H3("Sesión",
                                        style={"margin": "0 0 8px 0", "fontSize": "14px",
                                               "color": TEXT_SECONDARY}),
                                html.Div(id="session-stats",
                                         style={"fontSize": "13px", "lineHeight": "1.8"}),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ]
)


def _card_style() -> dict:
    return {
        "backgroundColor": BG_CARD, "borderRadius": "8px",
        "border": f"1px solid {BG_BORDER}", "padding": "14px",
    }


def _empty_candle_fig():
    """Gráfico vacío con el estilo correcto mientras no hay datos."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font_color=TEXT_PRIMARY,
        xaxis=dict(gridcolor=BG_BORDER, showgrid=True),
        yaxis=dict(gridcolor=BG_BORDER, showgrid=True),
        margin=dict(l=50, r=20, t=30, b=30),
        annotations=[dict(text="Esperando datos del bot…", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False,
                          font=dict(size=16, color=TEXT_SECONDARY))],
    )
    return fig


def _empty_equity_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font_color=TEXT_PRIMARY, height=180,
        xaxis=dict(gridcolor=BG_BORDER), yaxis=dict(gridcolor=BG_BORDER),
        margin=dict(l=50, r=20, t=10, b=30),
    )
    return fig


# ─── CALLBACK PRINCIPAL ───────────────────────────────────────────────
@app.callback(
    [Output("header-symbol", "children"),
     Output("header-mode", "children"),
     Output("header-mode", "style"),
     Output("header-time", "children"),
     Output("header-capital", "children"),
     Output("candlestick-chart", "figure"),
     Output("equity-chart", "figure"),
     Output("signal-action", "children"),
     Output("signal-action", "style"),
     Output("signal-confirmations", "children"),
     Output("signal-reasons", "children"),
     Output("position-info", "children"),
     Output("session-stats", "children")],
    [Input("refresh", "n_intervals")]
)
def update_dashboard(_n):
    state = read_state()
    if state is None:
        mode_style = {"fontSize": "14px", "padding": "4px 12px",
                      "borderRadius": "4px", "backgroundColor": COLOR_BLUE}
        return (
            "Sin conexión al bot", "—", mode_style, "", "",
            _empty_candle_fig(), _empty_equity_fig(),
            "—", {"fontSize": "28px", "fontWeight": "bold"}, "", "", "", ""
        )

    symbol = state.get("symbol", "")
    timeframe = state.get("timeframe", "")
    klines = state.get("klines", [])
    position = state.get("open_position")
    signal = state.get("last_signal", {})
    trades = state.get("trade_history", [])
    equity = state.get("equity_curve", [])
    session = state.get("session_info", {})

    # ── Header ──
    header_symbol = f"{symbol} — {timeframe}"
    mode = session.get("mode", "—")
    mode_bg = COLOR_YELLOW if mode == "ALERTA" else COLOR_BEAR
    mode_style = {"fontSize": "14px", "padding": "4px 12px",
                  "borderRadius": "4px", "backgroundColor": mode_bg,
                  "color": "#000" if mode == "ALERTA" else "#fff"}
    header_time = f"⏱ {session.get('time_remaining', '—')}"
    capital = session.get("capital", 0)
    header_capital = f"💰 {capital:.2f} USDT"

    # ── Gráfico de Velas ──
    candle_fig = _build_candle_chart(klines, trades, position)

    # ── Curva de Equity ──
    equity_fig = _build_equity_chart(equity)

    # ── Panel: Última Señal ──
    action = signal.get("action", "HOLD")
    action_color = COLOR_BULL if action == "BUY" else (COLOR_BEAR if action == "SELL" else TEXT_SECONDARY)
    action_style = {"fontSize": "28px", "fontWeight": "bold", "color": action_color}
    confirmations = f"Confirmaciones: {signal.get('confirmations', 0)}"
    reasons_list = signal.get("reasons", [])
    reasons_el = [html.Div(r) for r in reasons_list[:15]]  # máximo 15 para no saturar

    # ── Panel: Posición Abierta ──
    if position:
        pos_children = [
            _info_row("Tipo", position.get("type", "—"), COLOR_BULL if position.get("type") == "LONG" else COLOR_BEAR),
            _info_row("Entrada", f"${position.get('entry_price', 0):.2f}", COLOR_YELLOW),
            _info_row("Stop Loss", f"${position.get('stop_loss', 0):.2f}", COLOR_BEAR),
            _info_row("Take Profit", f"${position.get('take_profit', 0):.2f}", COLOR_BULL),
            _info_row("Liquidación", f"${position.get('liquidation_price', 0):.2f}", COLOR_BEAR),
            _info_row("Margen", f"${position.get('margin_used', 0):.2f} USDT", TEXT_PRIMARY),
        ]
    else:
        pos_children = [html.Span("Sin posición abierta", style={"color": TEXT_SECONDARY})]

    # ── Panel: Sesión ──
    sell_trades = [t for t in trades if t.get("action") == "SELL" and t.get("pnl") is not None]
    wins = sum(1 for t in sell_trades if t["pnl"] > 0)
    total_closed = len(sell_trades)
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    total_pnl = sum(t["pnl"] for t in sell_trades) if sell_trades else 0
    pnl_color = COLOR_BULL if total_pnl >= 0 else COLOR_BEAR

    session_children = [
        _info_row("Modo", mode, COLOR_YELLOW if mode == "ALERTA" else COLOR_BEAR),
        _info_row("Trades cerrados", str(total_closed), TEXT_PRIMARY),
        _info_row("Win Rate", f"{win_rate:.0f}%", COLOR_BULL if win_rate >= 50 else COLOR_BEAR),
        _info_row("PnL Sesión", f"{total_pnl:+.4f} USDT", pnl_color),
    ]

    return (
        header_symbol, mode, mode_style, header_time, header_capital,
        candle_fig, equity_fig,
        action, action_style, confirmations, reasons_el,
        pos_children, session_children,
    )


def _info_row(label: str, value: str, color: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "justifyContent": "space-between"},
        children=[
            html.Span(label, style={"color": TEXT_SECONDARY}),
            html.Span(value, style={"color": color, "fontWeight": "600"}),
        ]
    )


def _build_candle_chart(klines: list, trades: list, position: dict | None) -> go.Figure:
    if not klines:
        return _empty_candle_fig()

    times = [k.get("open_time", "") for k in klines]
    opens = [k.get("open", 0) for k in klines]
    highs = [k.get("high", 0) for k in klines]
    lows = [k.get("low", 0) for k in klines]
    closes = [k.get("close", 0) for k in klines]

    fig = go.Figure()

    # Velas
    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=COLOR_BULL, decreasing_line_color=COLOR_BEAR,
        increasing_fillcolor=COLOR_BULL, decreasing_fillcolor=COLOR_BEAR,
        name="Precio",
    ))

    # SMA 50
    sma50 = [k.get("sma_50", 0) for k in klines]
    if any(v > 0 for v in sma50):
        fig.add_trace(go.Scatter(
            x=times, y=[v if v > 0 else None for v in sma50],
            mode="lines", line=dict(color=COLOR_BLUE, width=1),
            name="SMA 50", visible="legendonly",
        ))

    # EMA 20
    ema20 = [k.get("ema_20", 0) for k in klines]
    if any(v > 0 for v in ema20):
        fig.add_trace(go.Scatter(
            x=times, y=[v if v > 0 else None for v in ema20],
            mode="lines", line=dict(color=COLOR_YELLOW, width=1),
            name="EMA 20", visible="legendonly",
        ))

    # Bollinger Bands
    bb_upper = [k.get("bb_upper", 0) for k in klines]
    bb_lower = [k.get("bb_lower", 0) for k in klines]
    if any(v > 0 for v in bb_upper):
        fig.add_trace(go.Scatter(
            x=times, y=[v if v > 0 else None for v in bb_upper],
            mode="lines", line=dict(color="#9c27b0", width=1, dash="dot"),
            name="BB Superior", visible="legendonly",
        ))
        fig.add_trace(go.Scatter(
            x=times, y=[v if v > 0 else None for v in bb_lower],
            mode="lines", line=dict(color="#9c27b0", width=1, dash="dot"),
            name="BB Inferior", visible="legendonly",
        ))

    # Marcadores de trades
    buy_trades = [t for t in trades if t.get("action") == "BUY"]
    sell_trades = [t for t in trades if t.get("action") == "SELL"]

    if buy_trades:
        fig.add_trace(go.Scatter(
            x=[t["time"] for t in buy_trades],
            y=[t["price"] for t in buy_trades],
            mode="markers", name="Compra",
            marker=dict(symbol="triangle-up", size=14, color=COLOR_BULL,
                        line=dict(width=1, color="#fff")),
        ))

    if sell_trades:
        # Color según PnL: celeste = ganancia, rojo = pérdida
        colors = [COLOR_CYAN if (t.get("pnl") or 0) > 0 else COLOR_BEAR for t in sell_trades]
        fig.add_trace(go.Scatter(
            x=[t["time"] for t in sell_trades],
            y=[t["price"] for t in sell_trades],
            mode="markers", name="Venta",
            marker=dict(symbol="triangle-down", size=14, color=colors,
                        line=dict(width=1, color="#fff")),
        ))

    # Líneas de posición abierta
    if position:
        for key, label, color, dash_style in [
            ("entry_price", "Entrada", COLOR_YELLOW, "solid"),
            ("stop_loss", "Stop Loss", COLOR_BEAR, "dash"),
            ("take_profit", "Take Profit", COLOR_BULL, "dash"),
            ("liquidation_price", "Liquidación", COLOR_BEAR, "dot"),
        ]:
            val = position.get(key)
            if val and val > 0:
                fig.add_hline(
                    y=val, line_dash=dash_style, line_color=color, line_width=1,
                    annotation_text=f"{label}: ${val:.2f}",
                    annotation_position="right",
                    annotation_font_color=color,
                    annotation_font_size=10,
                )

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font_color=TEXT_PRIMARY,
        xaxis=dict(gridcolor=BG_BORDER, showgrid=True, rangeslider_visible=False),
        yaxis=dict(gridcolor=BG_BORDER, showgrid=True, side="right"),
        margin=dict(l=10, r=80, t=30, b=30),
        legend=dict(
            bgcolor="rgba(30,34,45,0.85)", bordercolor=BG_BORDER,
            borderwidth=1, font=dict(size=11),
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        xaxis_type="category",
    )

    return fig


def _build_equity_chart(equity: list) -> go.Figure:
    if not equity:
        return _empty_equity_fig()

    times = [e.get("time", "") for e in equity]
    values = [e.get("equity", 0) for e in equity]
    start_val = values[0] if values else 0
    colors = [COLOR_BULL if v >= start_val else COLOR_BEAR for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines",
        line=dict(color=COLOR_BULL, width=2),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
        name="Equity",
    ))

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font_color=TEXT_PRIMARY, height=180,
        xaxis=dict(gridcolor=BG_BORDER, showgrid=False, showticklabels=False),
        yaxis=dict(gridcolor=BG_BORDER, showgrid=True, side="right",
                   title="USDT", title_font_size=10),
        margin=dict(l=10, r=80, t=5, b=20),
        showlegend=False,
        xaxis_type="category",
    )
    return fig


# ─── Arranque ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Dashboard en http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)

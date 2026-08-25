"""
Cliente de Binance FUTURES (USDT-M perpetuos). Distinto de spot: usa endpoints
futures_*, maneja leverage y margen, y las órdenes de salida (stop/take-profit) se
colocan como órdenes STOP_MARKET / TAKE_PROFIT_MARKET en vez de OCO.

IMPORTANTE:
- Testnet de Futures es una red separada de la de Spot: https://testnet.binancefuture.com
  Necesitas crear cuenta y API keys ahí específicamente (no son las mismas de
  testnet.binance.vision).
- Margen ISOLATED significa que si te liquidan, solo pierdes el margen de esa
  posición, no todo el balance de la cuenta futures.
- Las API keys reales deben tener SOLO permiso de "Enable Futures", nunca de retiros.
"""
import pandas as pd
# pyrefly: ignore [missing-import]
from binance.client import Client
# pyrefly: ignore [missing-import]
from binance import AsyncClient, BinanceSocketManager

from config import Config


class BinanceFuturesMarketClient:
    """Wrapper síncrono para datos históricos, leverage y órdenes REST en Futures."""

    def __init__(self):
        self.client = Client(
            Config.API_KEY,
            Config.API_SECRET,
            testnet=Config.TESTNET,
        )
        self._configured_symbols = set()

    def ensure_leverage_and_margin(self, symbol: str):
        """Configura leverage y tipo de margen una sola vez por símbolo por sesión."""
        if symbol in self._configured_symbols:
            return
        try:
            self.client.futures_change_margin_type(symbol=symbol, marginType=Config.MARGIN_TYPE)
        except Exception as e:
            # Si ya estaba configurado así, Binance devuelve error "No need to change margin type" -> se ignora
            if "No need to change" not in str(e):
                print(f"[AVISO margen] {e}")
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=Config.LEVERAGE)
        except Exception as e:
            print(f"[AVISO leverage] {e}")
        self._configured_symbols.add(symbol)

    def get_klines_df(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        raw = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["open_time", "open", "high", "low", "close", "volume"]]

    def get_futures_balance(self, asset: str = "USDT") -> float:
        balances = self.client.futures_account_balance()
        for b in balances:
            if b["asset"] == asset:
                return float(b["balance"])
        return 0.0

    def get_mark_price(self, symbol: str) -> float:
        data = self.client.futures_mark_price(symbol=symbol)
        return float(data["markPrice"])

    def place_market_order(self, symbol: str, side: str, quantity: float):
        """side: 'BUY' (abrir long) o 'SELL' (cerrar long / abrir short)."""
        return self.client.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=round(quantity, 3),
        )

    def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float):
        """Orden STOP_MARKET que cierra la posición si el precio toca stop_price."""
        return self.client.futures_create_order(
            symbol=symbol, side=side, type="STOP_MARKET",
            stopPrice=round(stop_price, 2), closePosition=True,
        )

    def place_take_profit(self, symbol: str, side: str, quantity: float, tp_price: float):
        return self.client.futures_create_order(
            symbol=symbol, side=side, type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp_price, 2), closePosition=True,
        )

    def cancel_all_open_orders(self, symbol: str):
        try:
            self.client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception as e:
            print(f"[AVISO cancelar órdenes]: {e}")


class BinanceFuturesStreamClient:
    """WebSocket de velas en tiempo real para Futures (baja latencia)."""

    def __init__(self, symbol: str, interval: str, on_kline_close):
        self.symbol = symbol.lower()
        self.interval = interval
        self.on_kline_close = on_kline_close
        self._client: AsyncClient | None = None

    async def run(self):
        self._client = await AsyncClient.create(
            Config.API_KEY, Config.API_SECRET, testnet=Config.TESTNET
        )
        bsm = BinanceSocketManager(self._client)
        socket = bsm.kline_futures_socket(symbol=self.symbol, interval=self.interval)
        try:
            async with socket as stream:
                while True:
                    msg = await stream.recv()
                    kline = msg.get("k", {})
                    if kline.get("x"):
                        await self.on_kline_close(kline)
        finally:
            await self._client.close_connection()
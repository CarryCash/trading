"""
Escritura y lectura atómica del archivo de estado compartido entre
el bot (main_futures.py) y el dashboard (run_dash.py).

El bot escribe → live_state.json.tmp → os.replace → live_state.json
El dashboard lee → live_state.json

os.replace es atómico a nivel de sistema operativo: el dashboard
nunca lee un archivo a medio escribir.
"""
import json
import os

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(_DIR, "live_state.json")
_TMP_FILE = STATE_FILE + ".tmp"


def write_state(data: dict) -> None:
    """Serializa *data* a JSON y lo escribe atómicamente a STATE_FILE."""
    os.makedirs(_DIR, exist_ok=True)
    payload = json.dumps(data, default=str, ensure_ascii=False)
    with open(_TMP_FILE, "w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(_TMP_FILE, STATE_FILE)


def read_state() -> dict | None:
    """Lee STATE_FILE y devuelve el dict, o None si no existe o está corrupto."""
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

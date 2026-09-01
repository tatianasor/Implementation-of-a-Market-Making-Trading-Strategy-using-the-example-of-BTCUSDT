import websocket
import threading
import json
import pandas as pd
import logging
import time
from datetime import datetime

LOG_FILE = 'historical_data_lin.log'

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

TICK_CSV = 'historical_data_lin.csv'
collect_tick = True

try:
    open(TICK_CSV, 'x').write('timestamp,price,size,side,time\n')
except FileExistsError:
    pass

def save_to_csv(filename, data):
    df = pd.DataFrame([data])
    df.to_csv(filename, mode='a', header=False, index=False)
    logging.info(f"[CSV] Сохранили тик: {data}")

def on_message(ws, message):
    global collect_tick
    logging.info(f"[WebSocket] Сообщение: {message}")
    try:
        data = json.loads(message)
        if 'data' in data and isinstance(data['data'], list):
            for trade in data['data']:
                timestamp = datetime.utcfromtimestamp(int(trade['T']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                row = {
                    'timestamp': timestamp,
                    'price': trade['p'],
                    'size': trade['v'],
                    'side': trade['S'],
                    'time': trade['T']
                }
                if collect_tick:
                    save_to_csv(TICK_CSV, row)
        else:
            logging.warning("[WebSocket] Нет данных в сообщении или формат отличается")
    except Exception as e:
        logging.error(f"[Ошибка] {e}")

def on_error(ws, error):
    logging.error(f"[WebSocket Error] {error}")

def on_close(ws, close_status_code, close_msg):
    logging.warning(f"[WebSocket Closed] Code: {close_status_code}, Msg: {close_msg}")

def on_open(ws):
    logging.info("[WebSocket] Подключение открыто, подписываюсь на publicTrade.BTCUSDT...")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": ["publicTrade.BTCUSDT"]
    }))

def run_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://stream.bybit.com/v5/public/linear",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            logging.info("[WebSocket] Запуск клиента...")
            ws.run_forever()
        except Exception as e:
            logging.error(f"[WebSocket Exception] {e}")
        logging.info("Переподключение через 5 секунд...")
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=run_ws, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Программа завершена.")

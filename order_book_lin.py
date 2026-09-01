import websocket
import threading
import json
import pandas as pd
import time
from datetime import datetime

TICK_CSV = 'order_book_tick_lin.csv'
collect_tick = True

try:
    open(TICK_CSV, 'x').write('timestamp,bid_price,bid_qty,ask_price,ask_qty\n')
except FileExistsError:
    pass

def save_to_csv(filename, data):
    print(f"Saving data: {data}")
    try:
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Data saved successfully: {data}")
    except Exception as e:
        print(f"Error saving data: {e}")

def on_message(ws, message):
    global collect_tick
    print(f"Received message: {message}")
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        print(f"Failed to decode message: {message}")
        return

    if 'data' in data and 'b' in data['data'] and 'a' in data['data']:
        best_bid = data['data']['b'][0]
        best_ask = data['data']['a'][0]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {
            'timestamp': timestamp,
            'bid_price': best_bid[0],
            'bid_qty': best_bid[1],
            'ask_price': best_ask[0],
            'ask_qty': best_ask[1]
        }

        print(f"Prepared row: {row}")
        if collect_tick:
            save_to_csv(TICK_CSV, row)

def run_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(
                "wss://stream.bybit.com/v5/public/linear",
                on_message=on_message,
                on_open=lambda ws: ws.send(json.dumps({
                    "op": "subscribe",
                    "args": ["orderbook.1.BTCUSDT"]
                })),
                on_error=lambda ws, error: print(f"WebSocket error: {error}"),
                on_close=lambda ws, close_status_code, close_msg: print(f"WebSocket closed: {close_status_code}, {close_msg}")
            )
            print("WebSocket connection opened.")
            ws.run_forever()
        except Exception as e:
            print(f"Exception in WebSocket connection: {e}")
        print("Reconnecting in 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=run_ws, daemon=True).start()
    collect_tick = True
    while True:
        time.sleep(1)

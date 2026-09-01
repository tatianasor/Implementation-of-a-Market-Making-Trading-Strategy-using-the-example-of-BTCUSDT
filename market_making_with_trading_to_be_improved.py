import websocket
import threading
import json
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from feature_engineering import generate_features_from_df  # Импортируем функцию для расчета фичей
import os
import numpy as np
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import joblib
from pybit.unified_trading import HTTP
import config
import csv
import time
from datetime import datetime



# Настройка логирования
LOG_FILE = 'combined_script.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Файлы для хранения данных
HISTORICAL_DATA_CSV = 'historical_data_lin.csv'
ORDER_BOOK_TICK_CSV = 'order_book_tick_lin.csv'
HISTORICAL_DATA_SEC_CSV = 'historical_data_sec_lin.csv'
ORDER_BOOK_SEC_CSV = 'order_book_sec_lin.csv'
COMBINED_DATA_CSV = 'data_historical_and_order_book_sec_lin.csv'
FEATURES_ALL_CSV = 'features_all_BTCUSDT_sec_lin.csv'
TRADING_RESULTS_CSV = 'trading_results.csv'

# Глобальные переменные
inventory = 1  # Начальное значение инвентаря
gamma = 0.5 #[0.1-1 по статье]
sigma = 2
T = 1  # Период времени
# dt = 0.1  # Время шага (например, 0.1 сек)
k = 1.5 # [1.5 - 2]
A = 140

# Инициализация файлов
def initialize_files():
    try:
        open(HISTORICAL_DATA_CSV, 'x').write('timestamp,price,size,side,time\n')
    except FileExistsError:
        pass
    try:
        open(ORDER_BOOK_TICK_CSV, 'x').write('timestamp,bid_price,bid_qty,ask_price,ask_qty\n')
    except FileExistsError:
        pass
    try:
        open(FEATURES_ALL_CSV, 'x').close()  # Создаем пустой файл для фичей
    except FileExistsError:
        pass

# WebSocket: Получение данных о сделках
def historical_data_ws():
    def save_to_csv(filename, data):
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=False, index=False)
        logging.info(f"[CSV] Сохранили тик: {data}")

    def on_message(ws, message):
        logging.info(f"[WebSocket] Сообщение: {message}")
        try:
            data = json.loads(message)
            if 'data' in data and isinstance(data['data'], list):
                for trade in data['data']:
                    # Преобразуем временную метку в UTC
                    timestamp = datetime.utcfromtimestamp(int(trade['T']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    row = {
                        'timestamp': timestamp,
                        'price': trade['p'],
                        'size': trade['v'],
                        'side': trade['S'],
                        'time': trade['T']
                    }
                    save_to_csv(HISTORICAL_DATA_CSV, row)
        except Exception as e:
            logging.error(f"[Ошибка] {e}")

    def run_ws():
        while True:
            try:
                ws = websocket.WebSocketApp(
                    "wss://stream.bybit.com/v5/public/linear",
                    on_open=lambda ws: ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["publicTrade.BTCUSDT"]
                    })),
                    on_message=on_message,
                    on_error=lambda ws, error: logging.error(f"[WebSocket Error] {error}"),
                    on_close=lambda ws, close_status_code, close_msg: logging.warning(f"[WebSocket Closed] Code: {close_status_code}, Msg: {close_msg}")
                )
                ws.run_forever()
            except Exception as e:
                logging.error(f"[WebSocket Exception] {e}")

    threading.Thread(target=run_ws, daemon=True).start()

# WebSocket: Получение данных о стакане
def order_book_ws():
    def save_to_csv(filename, data):
        df = pd.DataFrame([data])
        df.to_csv(filename, mode='a', header=False, index=False)

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if 'data' in data and 'b' in data['data'] and 'a' in data['data']:
                best_bid = data['data']['b'][0]
                best_ask = data['data']['a'][0]
                # Временная метка в UTC
                timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                row = {
                    'timestamp': timestamp,
                    'bid_price': best_bid[0],
                    'bid_qty': best_bid[1],
                    'ask_price': best_ask[0],
                    'ask_qty': best_ask[1]
                }
                save_to_csv(ORDER_BOOK_TICK_CSV, row)
        except Exception as e:
            logging.error(f"[Ошибка] {e}")

    def run_ws():
        while True:
            try:
                ws = websocket.WebSocketApp(
                    "wss://stream.bybit.com/v5/public/linear",
                    on_open=lambda ws: ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["orderbook.1.BTCUSDT"]
                    })),
                    on_message=on_message,
                    on_error=lambda ws, error: logging.error(f"[WebSocket Error] {error}"),
                    on_close=lambda ws, close_status_code, close_msg: logging.warning(f"[WebSocket Closed] Code: {close_status_code}, Msg: {close_msg}")
                )
                ws.run_forever()
            except Exception as e:
                logging.error(f"[WebSocket Exception] {e}")

    threading.Thread(target=run_ws, daemon=True).start()

# Обработка исторических данных (секундные свечи OHLCV)
def process_historical_data():
    try:
        df = pd.read_csv(HISTORICAL_DATA_CSV)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Приводим к UTC
        df['second'] = df['timestamp'].dt.floor('s')

        ohlcv = df.groupby('second').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', lambda x: x.astype(float).sum())
        ).reset_index()

        ohlcv.rename(columns={'second': 'timestamp'}, inplace=True)
        ohlcv.to_csv(HISTORICAL_DATA_SEC_CSV, index=False)
        logging.info(f"Секундные свечи сохранены в {HISTORICAL_DATA_SEC_CSV}")
    except Exception as e:
        logging.error(f"[Ошибка обработки historical_data] {e}")

# Обработка данных стакана (секундные значения)
def process_order_book():
    try:
        df = pd.read_csv(ORDER_BOOK_TICK_CSV)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Приводим к UTC
        df = df.sort_values('timestamp')

        df['second'] = df['timestamp'].dt.floor('s')
        second_df = df.groupby('second').tail(1).drop(columns=['second'])

        second_df.to_csv(ORDER_BOOK_SEC_CSV, index=False)
        logging.info(f"Секундные значения стакана сохранены в {ORDER_BOOK_SEC_CSV}")
    except Exception as e:
        logging.error(f"[Ошибка обработки order_book] {e}")

# Объединение данных
def combine_data():
    try:
        historical_data = pd.read_csv(HISTORICAL_DATA_SEC_CSV)
        order_book_data = pd.read_csv(ORDER_BOOK_SEC_CSV)

        # Приводим временные метки к UTC
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp']).dt.tz_localize(None)
        order_book_data['timestamp'] = pd.to_datetime(order_book_data['timestamp']).dt.tz_localize(None)

        # Объединяем данные по временным меткам
        combined_data = pd.merge(historical_data, order_book_data, on='timestamp', how='inner')
        combined_data.sort_values('timestamp', inplace=True)
        combined_data.to_csv(COMBINED_DATA_CSV, index=False)
        logging.info(f"Объединённые данные сохранены в {COMBINED_DATA_CSV}")
    except Exception as e:
        logging.error(f"[Ошибка объединения данных] {e}")

# Отслеживание изменений в COMBINED_DATA_CSV и обновление фичей
last_combined_data = None

def monitor_combined_data_and_update_features():
    global last_combined_data
    last_timestamp = None
    while True:
        try:
            # Проверяем, есть ли новые данные в COMBINED_DATA_CSV
            try:
                combined_data = pd.read_csv(COMBINED_DATA_CSV)
                current_timestamp = combined_data['timestamp'].max()
            except FileNotFoundError:
                current_timestamp = None

            if current_timestamp != last_timestamp:
                logging.info(f"Новые данные обнаружены. Последняя временная метка: {current_timestamp}")
                last_timestamp = current_timestamp

                # Проверка на изменения в данных
                if last_combined_data is not None and combined_data.equals(last_combined_data):
                    logging.info("Данные не изменились. Пропускаем обновление фичей.")
                    continue

                # Обновляем кэш
                last_combined_data = combined_data

                # Проверка наличия обязательных колонок
                required_columns = ['bid_price', 'ask_price', 'bid_qty', 'ask_qty', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in combined_data.columns for col in required_columns):
                    logging.error("[Ошибка] В данных отсутствуют обязательные колонки.")
                    continue

                # Генерация фичей
                features = generate_features_from_df(combined_data)

                # Избегаем дублирования записей
                if os.path.exists(FEATURES_ALL_CSV):
                    existing_features = pd.read_csv(FEATURES_ALL_CSV)
                    features = features[~features['timestamp'].isin(existing_features['timestamp'])]

                # Сохраняем фичи в файл
                features.to_csv(FEATURES_ALL_CSV, mode='a', header=not os.path.exists(FEATURES_ALL_CSV), index=False)
                logging.info(f"Фичи обновлены и сохранены в {FEATURES_ALL_CSV}")

        except KeyboardInterrupt:
            logging.info("Мониторинг завершён.")
            break
        except Exception as e:
            logging.error(f"[Ошибка мониторинга] {e}")



# Загрузка моделей
catboost_model = cb.CatBoostRegressor()
catboost_model.load_model('lasso_catboost_model.cbm')

lightgbm_model = lgb.Booster(model_file='lasso_lightgbm_model.txt')

xgboost_model = xgb.Booster()
xgboost_model.load_model('lasso_xgboost_model.json')

random_forest_model = joblib.load('lasso_rf_model.pkl')

# Фичи, используемые в моделях
selected_features = [
    'macd_hist', 'order_book_imbalance', 'macd', 'moving_average',
 'rsi', 'adx', 'volatility', 'atr'
 ]

session = HTTP(
    testnet=True,
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
)

def update_inventory():
    global inventory
    try:
        # Обновление инвентаря
        balance_data = session.get_wallet_balance(accountType="UNIFIED", coin="BTC")
        inventory = float(next(
            item["equity"] for item in balance_data['result']['list'][0]['coin'] if item["coin"] == "BTC"
        ))
        logging.info(f"[Inventory] Обновлено значение инвентаря: {inventory:.4f} BTC")
    except Exception as e:
        logging.error(f"[Inventory Error] Ошибка при обновлении инвентаря: {e}")

    return

# Торговый алгоритм
def trading_algorithm():
    global inventory
    update_inventory()

    # Загрузка фичей
    df_features = pd.read_csv(FEATURES_ALL_CSV)
    df_features['mid_price_next'] = df_features['mid_price'].shift(-1)
    df_features = df_features.dropna()
    df_features = df_features[selected_features]

    if df_features.empty:
        logging.warning("[Trading] Нет данных для прогноза.")
        return

    latest_row = df_features.iloc[[-1]]

    # Среднее предсказание цены mid_price
    predicted_mid_price = float(np.mean([
        catboost_model.predict(latest_row)[0],
        lightgbm_model.predict(latest_row)[0],
        xgboost_model.predict(xgb.DMatrix(latest_row))[0],
        random_forest_model.predict(latest_row)[0]
    ]))
    logging.info(f"[Trading] Среднее предсказание Mid_price (через 1 сек): {predicted_mid_price:.2f}")

    print(type(predicted_mid_price))
    print(type(inventory))
    print(type(gamma))
    print(type(sigma))
    print(type(T))


    # Расчет оптимального спреда и резервной цены
    reservation_price = predicted_mid_price - inventory * gamma * sigma ** 2 * T
    base_spread = gamma * sigma ** 2 * T + (2 / gamma) * np.log(1 + gamma / k)
    spread = base_spread / 2

    if reservation_price >= predicted_mid_price:
        ask_spread = spread + (reservation_price - predicted_mid_price)
        bid_spread = spread - (reservation_price - predicted_mid_price)
    else:
        ask_spread = spread - (predicted_mid_price - reservation_price)
        bid_spread = spread + (predicted_mid_price - reservation_price)



    # Риск-менеджмент
    quantity = 0.001  # Стандартный объем ордера
    max_inventory = 0.1
    spread_multiplier = 1.0
    features_all = pd.read_csv(FEATURES_ALL_CSV)
    latest_features_all = features_all.iloc[[-1]]


    if abs(inventory) > max_inventory:
        logging.warning(f"[RISK] Превышен лимит инвентаря! Текущий: {inventory:.4f} BTC")
        spread_multiplier *= 1.5

    high_volatility_threshold = np.percentile(features_all['volatility'], 90)
    extreme_volatility_threshold = np.percentile(features_all['volatility'], 95)
    current_volatility = latest_features_all['volatility'].values[0]

    if current_volatility > extreme_volatility_threshold:
        logging.critical("[RISK] Чрезмерная волатильность! Приостанавливаем торговлю.")
        return
    elif current_volatility > high_volatility_threshold:
        logging.warning("[RISK] Высокая волатильность! Расширяем спред.")
        spread_multiplier *= 2.0

    # Корректировка спредов
    order_imbalance = latest_features_all['order_book_imbalance'].values[0]

    if order_imbalance > np.percentile(features_all['order_book_imbalance'], 90):
        logging.warning("[RISK] Дисбаланс в сторону покупателей! Корректируем спред.")
        bid_spread *= 0.8
        ask_spread *= 1.2
    elif order_imbalance < np.percentile(features_all['order_book_imbalance'], 10):
        logging.warning("[RISK] Дисбаланс в сторону продавцов! Корректируем спред.")
        bid_spread *= 1.2
        ask_spread *= 0.8

    if latest_features_all['liquidity_pressure'].values[0] < np.percentile(features_all['liquidity_pressure'], 10):
        logging.warning("[RISK] Низкая ликвидность! Уменьшаем объем ордеров.")
        quantity *= 0.5
        ask_spread *= 0.8



        # Размещение ордеров
    place_order(predicted_mid_price, bid_spread, ask_spread, spread_multiplier, quantity)


# Метод размещения ордеров
def place_order(predicted_mid_price, bid_spread, ask_spread, spread_multiplier, quantity):
    global inventory

    bid_spread *= spread_multiplier
    ask_spread *= spread_multiplier

    bid_price = round(predicted_mid_price - bid_spread, 2)
    ask_price = round(predicted_mid_price + ask_spread, 2)

    try:
        buy_order = session.place_order(
            category="linear",
            symbol="BTCUSDT",
            side="Buy",
            orderType="Limit",
            qty=quantity,
            price=bid_price,
            timeInForce="GTC",
            isLeverage=0,
            orderFilter="Order",
        )
        logging.info(f"[Order] Заявка на покупку размещена: {buy_order}")

        if inventory > 0:
            sell_order = session.place_order(
                category="linear",
                symbol="BTCUSDT",
                side="Sell",
                order_type="Limit",
                qty=quantity,
                price=ask_price,
                time_in_force="GTC",
                isLeverage=0,
                orderFilter="Order"
            )
            logging.info(f"[Order] Заявка на продажу размещена: {sell_order}")

    except Exception as e:
        logging.error(f"[Bybit API Error] Ошибка при размещении ордеров: {e}")

    # Получение баланса и запись результатов

    try:
        balance_data = session.get_wallet_balance(accountType="UNIFIED", coin="BTC")
        cum_realized_pnl = float(next(
            (item["cumRealisedPnl"] for item in balance_data['result']['list'][0]['coin'] if item["coin"] == "BTC"),
            0.0  # Значение по умолчанию, если нет данных
        ))
        logging.info(f"[Realized_PnL] Обновлено значение Cum_Realized PnL: {cum_realized_pnl :.6f}")

        with open(TRADING_RESULTS_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Если файл пустой, записать заголовки
                writer.writerow([
                    "Timestamp", "Inventory", "CumRealized_PnL", "Predicted_Mid_Price", "bid_apread", "ask_spread"
                ])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{inventory:.8f}",  # Форматируем inventory
                f"{cum_realized_pnl:.8f}",  # Форматируем cum_realized_pnl
                f"{predicted_mid_price:.2f}"  # Записываем предсказанный mid_price
            ])

        logging.info(
            f"[Trading Results] Inventory={inventory:.8f}, CumRealizedPnL={cum_realized_pnl:.8f}, Predicted_Mid_Price={predicted_mid_price:.2f}, bid_apread={bid_spread:.8f}, ask_spread={ask_spread:.8f}"
        )


    except Exception as e:
        logging.error(f"[Bybit API Error] Ошибка получения данных: {e}")

    except Exception as e:
        logging.error(f"[Trading Error] Ошибка в торговом алгоритме: {e}")

if __name__ == "__main__":

    # Ограничение времени работы программы (15 минут)
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=15)
    initialize_files()
    historical_data_ws()
    order_book_ws()
    threading.Thread(target=monitor_combined_data_and_update_features, daemon=True).start()

    try:
        while datetime.now() < end_time:
            process_historical_data()
            process_order_book()
            combine_data()
            trading_algorithm()
            time.sleep(0.1)  # Ждем 0.1 секунду перед следующим шагом
    except KeyboardInterrupt:
        logging.info("Программа завершена пользователем.")
    finally:
        logging.info("Программа завершена после 15 минут работы.")
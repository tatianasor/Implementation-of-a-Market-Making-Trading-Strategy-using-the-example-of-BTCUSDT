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
inventory = 0            # Начальное значение инвентаря
q_target = 0             # Начальное значение инвентаря
T = 1                   # Временной горизонт (секунды)
phi = 0.005             # 0.001–0.01 Штраф за отклонение от целевого уровня инвентаря
kappa = 200    # 100-300 - Чувствительность к цене

# lamb = 0.73             # 0.1–1.0 Интенсивность

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
LightGBM_model = lgb.Booster(model_file='lasso_lightgbm_model.txt')
print(LightGBM_model.num_trees())  # Должно вывести количество деревьев
print(LightGBM_model)  # Должен вывести информацию о модели

# catboost_model = cb.CatBoostRegressor()
# catboost_model.load_model('lasso_catboost_model.cbm')

# Фичи, используемые в моделях
selected_features = [
    'macd_hist', 'order_book_imbalance', 'macd', 'moving_average',
 'rsi', 'volatility', 'obv', 'adx','atr'
 ]


session = HTTP(
    testnet=True,
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
)

def update_inventory():
    global inventory
    try:
        # Запрашиваем позицию BTCUSDT (Linear)
        position_data = session.get_positions(category="linear", symbol="BTCUSDT")

        # Проверяем, есть ли открытые позиции
        if "result" in position_data and "list" in position_data["result"] and position_data["result"]["list"]:
            inventory = float(position_data['result']['list'][0]['size'])
            logging.info(f"[Inventory] Обновлено значение size: {inventory:.6f} BTC")
        else:
            inventory = 0.0  # Если позиций нет, инвентарь = 0
            logging.info("[Inventory] Нет открытых позиций, size установлен в 0.0 BTC")

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

    missing_features = [feat for feat in selected_features if feat not in df_features.columns]
    if missing_features:
        logging.error(f"[Ошибка] В данных отсутствуют колонки: {missing_features}")
        return

    if df_features[selected_features].isnull().values.any():
        logging.warning("[Warning] В данных есть пропущенные значения!")
        df_features = df_features.dropna()


    latest_row = df_features[selected_features].iloc[[-1]]
    predicted_mid_price = float(LightGBM_model.predict(latest_row.to_numpy())[0])
    logging.info(f"[Trading] Среднее предсказание Mid_price (через 1 сек): {predicted_mid_price:.2f}")

    # Расчет интенсивности поступления лимитных ордеров (λ) - lambda 50% lim orders
    data2 = pd.read_csv('historical_data_lin.csv')
    data2['timestamp'] = pd.to_datetime(data2['timestamp'])
    ticks_per_second = data2.groupby(data2['timestamp'].dt.floor('s'))['size'].sum()
    # Среднее количество ордеров в секунду за последние 30 секунд
    trades_per_second = ticks_per_second.tail(30).mean()
    lamb = trades_per_second / 2  # Поступление лимитных ордеров (50% лимитных ордеров)





    # Расчет оптимального спреда и резервной цены (Cartea)
    d_omega_dq = phi * (inventory - q_target)
    denominator = phi + d_omega_dq
    log_term = np.log(lamb / denominator)
    spread = max(0, (log_term / kappa))  # Глубина не может быть отрицательной

    ask_spread = spread
    bid_spread = spread

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
        position_data = session.get_positions(category="linear", symbol="BTCUSDT")['result']['list'][0]

        positionValue = float(position_data['positionValue'])
        avgPrice = float(position_data['avgPrice'])
        markPrice = float(position_data['markPrice'])
        side = position_data['side']  # строка "Buy" или "Sell"
        unrealisedPnl = float(position_data['unrealisedPnl'])
        curRealisedPnl = float(position_data['curRealisedPnl'])
        cumRealisedPnl = float(position_data['cumRealisedPnl'])

        combined_data = pd.read_csv(COMBINED_DATA_CSV)
        last_bid_price = combined_data['bid_price'].iloc[-1]
        last_ask_price = combined_data['ask_price'].iloc[-1]

        logging.info(f"[positionValue] {positionValue:.6f}")
        logging.info(f"[avgPrice] {avgPrice:.6f}")
        logging.info(f"[markPrice] {markPrice:.6f}")
        logging.info(f"[side] {side}")
        logging.info(f"[unrealisedPnl] {unrealisedPnl:.6f}")
        logging.info(f"[curRealisedPnl] {curRealisedPnl:.6f}")
        logging.info(f"[cumRealisedPnl] {cumRealisedPnl:.6f}")

        # Проверяем, существует ли файл и содержит ли он заголовки
        file_exists = os.path.exists(TRADING_RESULTS_CSV)

        with open(TRADING_RESULTS_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Inventory", "PositionValue", "AvgPrice", "MarkPrice", "Side",
                    "UnrealisedPnl", "CurRealisedPnl", "CumRealisedPnl",
                    "Predicted_Mid_Price", "Bid_Price", "Ask_Price", "last_bid_price", "last_ask_price"
                ])

            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                f"{inventory:.8f}",
                f"{positionValue:.8f}",
                f"{avgPrice:.8f}",
                f"{markPrice:.8f}",
                f"{side}",
                f"{unrealisedPnl:.8f}",
                f"{curRealisedPnl:.8f}",
                f"{cumRealisedPnl:.8f}",
                f"{predicted_mid_price:.2f}",
                f"{bid_price:.2f}",
                f"{ask_price:.2f}",
                f"{last_bid_price:.2f}",
                f"{last_ask_price:.2f}"
            ])

        logging.info(f"[Trading Results] Данные успешно записаны в {TRADING_RESULTS_CSV}")

    except Exception as e:
        logging.error(f"[Bybit API Error] Ошибка получения данных: {e}")


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
            time.sleep(0.005)  # Ждем 0.1 секунду перед следующим шагом
    except KeyboardInterrupt:
        logging.info("Программа завершена пользователем.")
    finally:
        logging.info("Программа завершена после 15 минут работы.")
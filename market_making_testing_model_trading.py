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

    predicted_mid_price = 78500
    logging.info(f"[Trading] Среднее предсказание Mid_price (через 1 сек): {predicted_mid_price:.2f}")

    bid_price = 0.05
    ask_price = 0.05
    quantity = 0.005

    # Размещение ордеров
    place_order(predicted_mid_price, bid_price, ask_price, quantity)

# Метод размещения ордеров
def place_order(predicted_mid_price, bid_price, ask_price, quantity):
    global inventory

    try:
            buy_order = session.place_order(
                category="linear",
                symbol="BTCUSDT",
                side="Buy",
                orderType="Limit",
                qty= quantity,
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
    try:
            trading_algorithm()
            time.sleep(0.005)  # Ждем 0.1 секунду перед следующим шагом
    except KeyboardInterrupt:
        logging.info("Программа завершена пользователем.")


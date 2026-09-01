
import pandas as pd
import numpy as np
import talib as ta

# Функции для расчета признаков из L2 Order Book (глубина стакана)
def calculate_mid_price(data):
    return (data['bid_price'].astype(float) + data['ask_price'].astype(float)) / 2

def calculate_bid_ask_spread(data):
    return data['ask_price'].astype(float) - data['bid_price'].astype(float)

def calculate_order_book_imbalance(data):
    total_bid_quantity = data['bid_qty'].astype(float)
    total_ask_quantity = data['ask_qty'].astype(float)
    imbalance = (total_bid_quantity - total_ask_quantity) / (total_bid_quantity + total_ask_quantity)
    return imbalance.replace([np.inf, -np.inf], np.nan)

def calculate_relative_spread(data):
    spread = calculate_bid_ask_spread(data)
    mid_price = calculate_mid_price(data)
    return spread / mid_price

def calculate_liquidity_pressure(data):
    total_bid_quantity = data['bid_qty'].astype(float)
    total_ask_quantity = data['ask_qty'].astype(float)
    return total_bid_quantity / (total_bid_quantity + total_ask_quantity)

# Функции для расчета признаков из исторических данных (OHLCV)
def calculate_volatility(data, window=30):
    return data['close'].astype(float).rolling(window=window).std()

def calculate_moving_average(data, window=30):
    return data['close'].astype(float).rolling(window=window).mean()

def calculate_rsi(data, window=14):
    return ta.RSI(data['close'].values.astype(float), timeperiod=window)

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['close'].astype(float).rolling(window=window).mean()
    rolling_std = data['close'].astype(float).rolling(window=window).std()
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_macd(data):
    macd, macd_signal, macd_hist = ta.MACD(data['close'].values.astype(float), fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, macd_signal, macd_hist

def calculate_adx(data, window=14):
    return ta.ADX(data['high'].values.astype(float), data['low'].values.astype(float), data['close'].values.astype(float), timeperiod=window)

def calculate_ema(data, window=14):
    return ta.EMA(data['close'].values.astype(float), timeperiod=window)

def calculate_ichimoku(data):
    high_9 = data['high'].astype(float).rolling(window=9).max()
    low_9 = data['low'].astype(float).rolling(window=9).min()
    conversion_line = (high_9 + low_9) / 2

    high_26 = data['high'].astype(float).rolling(window=26).max()
    low_26 = data['low'].astype(float).rolling(window=26).min()
    base_line = (high_26 + low_26) / 2
    return conversion_line, base_line

def calculate_atr(data, window=14):
    return ta.ATR(data['high'].values.astype(float), data['low'].values.astype(float), data['close'].values.astype(float), timeperiod=window)

def calculate_obv(data):
    return ta.OBV(data['close'].values.astype(float), data['volume'].values.astype(float))

# Основной блок
def generate_features_from_df(data):
    # Предобработка
    data[['bid_price', 'ask_price', 'bid_qty', 'ask_qty']] = data[['bid_price', 'ask_price', 'bid_qty', 'ask_qty']].ffill()
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].ffill()
    data = data.replace([np.inf, -np.inf], np.nan).ffill()

    # Генерация признаков
    data['mid_price'] = calculate_mid_price(data)
    data['bid_ask_spread'] = calculate_bid_ask_spread(data)
    data['order_book_imbalance'] = calculate_order_book_imbalance(data)
    data['relative_spread'] = calculate_relative_spread(data)
    data['liquidity_pressure'] = calculate_liquidity_pressure(data)

    data['volatility'] = calculate_volatility(data)
    data['moving_average'] = calculate_moving_average(data)
    data['rsi'] = calculate_rsi(data)

    upper_band, lower_band = calculate_bollinger_bands(data)
    data['bollinger_upper'] = upper_band
    data['bollinger_lower'] = lower_band

    macd, macd_signal, macd_hist = calculate_macd(data)
    data['macd'] = macd
    data['macd_signal'] = macd_signal
    data['macd_hist'] = macd_hist

    data['adx'] = calculate_adx(data)
    data['ema'] = calculate_ema(data)
    conversion_line, base_line = calculate_ichimoku(data)
    data['ichimoku_conversion'] = conversion_line
    data['ichimoku_base'] = base_line
    data['atr'] = calculate_atr(data)
    data['obv'] = calculate_obv(data)

    # Проверка на NaN
    print("NaN в фичах:")
    print(data.isnull().sum())

    # Очистка
    data = data.dropna()
    print(f'Финальный размер после очистки: {data.shape}')

    # Удаляем ненужные столбцы перед возвратом
    data = data.drop(columns=['open', 'high', 'low', 'close', 'volume', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty'], errors='ignore')

    return data

# Главная функция для генерации фичей
def feature_engineering():
    try:
        # Чтение объединённых данных
        combined_data = pd.read_csv('data_historical_and_order_book_sec_lin.csv')
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])

        # Генерация фичей
        features = generate_features_from_df(combined_data)

        # Сохранение фичей
        features.to_csv('features_all_BTCUSDT_sec_lin.csv', index=False)
        print(f"Фичи сохранены в features_all_BTCUSDT_sec_lin.csv")
    except Exception as e:
        print(f"[Ошибка feature engineering] {e}")

# Вызов функции feature_engineering
if __name__ == "__main__":
    feature_engineering()


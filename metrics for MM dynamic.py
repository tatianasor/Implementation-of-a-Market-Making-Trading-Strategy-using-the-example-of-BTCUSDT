import pandas as pd
import numpy as np

HISTORICAL_DATA_CSV = 'historical_data_lin.csv'
ORDER_BOOK_TICK_CSV = 'order_book_tick_lin.csv'
HISTORICAL_DATA_SEC_CSV = 'historical_data_sec_lin.csv'
ORDER_BOOK_SEC_CSV = 'order_book_sec_lin.csv'
COMBINED_DATA_CSV = 'data_historical_and_order_book_sec_lin.csv'
FEATURES_ALL_CSV = 'features_all_BTCUSDT_sec_lin.csv'
TRADING_RESULTS_CSV = 'trading_results.csv'

data = pd.read_csv('data_historical_and_order_book_sec_lin.csv')
data2 = pd.read_csv('historical_data_lin.csv')

# Расчет sigma (секундная волатильность)
mid_price = (data['bid_price'] + data['ask_price']) / 2
log_returns = np.log(mid_price / mid_price.shift(1)).dropna()
mid_price_returns = (mid_price / mid_price.shift(1)).dropna()
sigma_sec = log_returns.std()
print(f"Секундная волатильность (sigma): {sigma_sec:.6f}")


# Расчет интенсивности поступления лимитных ордеров (λ) - lambda 50% lim orders
data2['timestamp'] = pd.to_datetime(data2['timestamp'])
ticks_per_second = data2.groupby(data2['timestamp'].dt.floor('s'))['size'].sum()
trades_per_second = ticks_per_second.mean()
lambda_value = trades_per_second / 2
print(f"Поступление лимитных ордеров(lambda): {lambda_value:.6f}")




# Расчет sigma (секундная волатильность) - последние 30
data = pd.read_csv('data_historical_and_order_book_sec_lin.csv')
data = data.tail(30)  # последние 30 значений
mid_price = (data['bid_price'] + data['ask_price']) / 2
log_returns = np.log(mid_price / mid_price.shift(1)).dropna()
mid_price_returns = (mid_price / mid_price.shift(1)).dropna()
sigma_sec = log_returns.std()
print(f"Секундная волатильность (sigma) для последних 30 значений: {sigma_sec:.6f}")



# Расчет интенсивности поступления лимитных ордеров (λ) - lambda 50% lim orders - последние 30
data2 = pd.read_csv('historical_data_lin.csv')
data2['timestamp'] = pd.to_datetime(data2['timestamp'])
ticks_per_second = data2.groupby(data2['timestamp'].dt.floor('s'))['size'].sum()
# Среднее количество ордеров в секунду за последние 30 секунд
trades_per_second = ticks_per_second.tail(30).mean()
lambda_value = trades_per_second / 2  # Поступление лимитных ордеров (50% лимитных ордеров)
print(f"Поступление лимитных ордеров (lambda) для последних 30 значений: {lambda_value:.6f}")


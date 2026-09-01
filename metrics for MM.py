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
sigma_sec = mid_price_returns.std()
print(f"Секундная волатильность (sigma): {sigma_sec:.6f}")

# Расчет k (интенсивность прибыли)
data2['timestamp'] = pd.to_datetime(data2['timestamp'])
ticks_per_second = data2.groupby(data2['timestamp'].dt.floor('s'))['size'].sum()
trades_per_second = ticks_per_second.mean()
avg_spread = (data['ask_price'] - data['bid_price']).mean()
k = trades_per_second * np.exp(-avg_spread / sigma_sec)
print(f"Интенсивность прибыли (k): {k:.6f}")

# Расчет интенсивности поступления лимитных ордеров (λ) - lambda 50% lim orders
lambda_value = trades_per_second / 2
print(f"Поступление лимитных ордеров(lambda): {lambda_value:.6f}")




import pandas as pd
import numpy as np

# Загрузка данных
data = pd.read_csv(ORDER_BOOK_TICK_CSV)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['second'] = data['timestamp'].dt.floor('s')
data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2

if (data['bid_price'] <= 0).any() or (data['ask_price'] <= 0).any():
    raise ValueError("Данные содержат нулевые или отрицательные значения bid_price или ask_price.")
if len(data) < 100:
    raise ValueError("Недостаточно данных для расчета.")

# Интенсивность поступления ордеров
lambda_value = len(data) / (data['timestamp'].max() - data['timestamp'].min()).total_seconds()

# Функция для расчета вероятности
def compute_probability(delta):
    price_reaches_level = data['mid_price'] >= (data['bid_price'] + delta)
    probability_per_second = price_reaches_level.groupby(data['second']).mean()
    return probability_per_second.mean()

# Уточнение между двумя дельтами
low = 0.05000000000999999666
high = 0.05000000001100000230
threshold = 0.5  # Целевая граница вероятности
precision = 1e-15  # Насколько точно искать границу
max_iterations = 100
transition_delta = None

for _ in range(max_iterations):
    mid = (low + high) / 2
    prob = compute_probability(mid)

    if abs(high - low) < precision:
        transition_delta = mid
        break

    if prob > threshold:
        low = mid
    else:
        high = mid

# Финальная дельта и вероятность
transition_delta = (low + high) / 2
transition_probability = compute_probability(transition_delta)

print(f"Transition delta found at: {transition_delta:.20f}, probability: {transition_probability:.6f}")

# Обновляем список дельт, добавив найденную
deltas = [0.05, 0.050000000005, 0.05000000001, 0.05000000003, 0.05000000005, transition_delta]

# Пересчитываем вероятности для каждой дельты
probabilities = []
for delta in deltas:
    probability = compute_probability(delta)
    probabilities.append(probability)

# Логарифмическая зависимость
filtered_probabilities = [p for p in probabilities if p > 0]
if len(filtered_probabilities) != len(probabilities):
    print("Предупреждение: Некоторые вероятности равны 0 и были исключены из расчета.")

log_probabilities = np.log(np.array(filtered_probabilities) / lambda_value)

# Линейная регрессия для оценки κ
coeffs = np.polyfit(deltas[:len(filtered_probabilities)], log_probabilities, 1)
kappa = -coeffs[0]

print(f"Calculated Sensitivity to Price (κ): {kappa:.4f}")
print("Probabilities for each delta:", list(zip(deltas, probabilities)))












import numpy as np
import pandas as pd

# Загрузка данных
data = pd.read_csv(ORDER_BOOK_TICK_CSV)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['second'] = data['timestamp'].dt.floor('s')
data['mid_price'] = (data['bid_price'] + data['ask_price']) / 2

# Проверка данных на нулевые или отрицательные значения
if (data['bid_price'] <= 0).any() or (data['ask_price'] <= 0).any():
    raise ValueError("Данные содержат нулевые или отрицательные значения bid_price или ask_price.")
if len(data) < 100:
    raise ValueError("Недостаточно данных для расчета.")

# Рассчитываем логарифмические доходности для средней цены (mid_price)
log_returns = np.log(data['mid_price'] / data['mid_price'].shift(1)).dropna()

# Волатильность за секунду
sigma_sec = log_returns.std()

# Расчет интенсивности поступления ордеров
lambda_value = len(data) / (data['timestamp'].max() - data['timestamp'].min()).total_seconds()

# Простой расчет κ
# Логарифмические доходности и волатильность для регрессии
log_returns = log_returns.values
volatility = sigma_sec

# Моделируем зависимость изменения цены от волатильности
# Можем предположить, что изменения цены пропорциональны волатильности (это стандартный подход для Market Making)
X = np.vstack([volatility * np.ones_like(log_returns), log_returns]).T

# Линейная регрессия
coeffs = np.linalg.lstsq(X, log_returns, rcond=None)[0]

# Примерно κ = коэффициент, соответствующий волатильности
kappa = coeffs[0]

# Выводим результат
print(f"Calculated Sensitivity to Price (κ): {kappa:.4f}")



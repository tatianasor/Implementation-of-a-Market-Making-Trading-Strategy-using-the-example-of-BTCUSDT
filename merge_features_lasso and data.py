import pandas as pd
import pandas as pd
import numpy as np


# Читаем оба файла полностью
data_historical_full = pd.read_csv("data_historical_and_order_book_sec_lin.csv")
selected_features_full = pd.read_csv("features_all_BTCUSDT_sec_lin.csv")
results_full = pd.read_csv("predictions_lstm_torch.csv")

results_full2 = results_full[(results_full['model'] == 'LSTM') & (results_full['feature_set'] == 'lasso')]
results_full2 = results_full2.tail(14662).reset_index(drop=True)
results_full2.to_csv("predictions_lstm_best_test.csv", index=False)

# Вывод количества финальных строк
print(f"Количество финальных строк LSTM: {len(results_full2)}")
print(f"Количество финальных строк фичей: {len(selected_features_full)}")
print(f"Количество финальных строк данных: {len(data_historical_full)}")

# Берем последние 14662 строки из каждого файла
data_historical = data_historical_full.tail(14662).reset_index(drop=True)

selected_features_1 = selected_features_full.head(0).reset_index(drop=True)
selected_features_end = selected_features_full.tail(14662).reset_index(drop=True)
selected_features = pd.concat([selected_features_1, selected_features_end], axis=0)

# Вывод количества финальных строк
print(f"Количество финальных строк LSTM: {len(results_full2)}")
print(f"Количество финальных строк фичей: {len(selected_features)}")
print(f"Количество финальных строк данных: {len(data_historical)}")

# Объединяем данные по столбцам
combined_data = pd.concat([data_historical, selected_features, results_full2], axis=1)

# Берем последние строки (test)
# combined_data = combined_data.tail(85877)

# Сохраняем результат
combined_data.to_csv("combined_output.csv", index=False)



df = combined_data

# Проверка на наличие нужных колонок
assert 'y_true' in df.columns and 'y_pred' in df.columns, "Отсутствуют колонки y_true и/или y_pred"

# Расчёт ошибок
df['mae'] = np.abs(df['y_true'] - df['y_pred'])
df['mse'] = (df['y_true'] - df['y_pred'])**2

# Перцентильные уровни
percentiles = [5, 10, 25, 50, 75, 80, 100]

# Вывод перцентилей MAE
mae_percentiles = df['mae'].quantile([p / 100 for p in percentiles])
print("📊 MAE percentiles:")
print(mae_percentiles)

# Вывод перцентилей MSE
mse_percentiles = df['mse'].quantile([p / 100 for p in percentiles])
print("\n📊 MSE percentiles:")
print(mse_percentiles)




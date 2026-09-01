import pandas as pd

# Файлы
INPUT_CSV = 'historical_data_lin.csv'
OUTPUT_CSV = 'historical_data_sec_lin.csv'

# Чтение исходных данных
df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Округляем до секунды
df['second'] = df['timestamp'].dt.floor('s')

# Группировка и построение OHLCV
ohlcv = df.groupby('second').agg(
    open=('price', 'first'),
    high=('price', 'max'),
    low=('price', 'min'),
    close=('price', 'last'),
    volume=('size', lambda x: x.astype(float).sum())
).reset_index()

# Переименовываем колонку second в timestamp
ohlcv.rename(columns={'second': 'timestamp'}, inplace=True)

# Сохраняем результат
ohlcv.to_csv(OUTPUT_CSV, index=False)

print(f'Готово! Секундные свечи сохранены в {OUTPUT_CSV}')

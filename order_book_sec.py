import pandas as pd

TICK_CSV = 'order_book_tick_lin.csv'
SEC_CSV = 'order_book_sec_lin.csv'

# Читаем данные из тик-файла
df = pd.read_csv(TICK_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Группируем по секундам и берем последний тик каждой секунды
df['second'] = df['timestamp'].dt.floor('s')
second_df = df.groupby('second').tail(1).drop(columns=['second'])

# Сохраняем результат в новый файл
second_df.to_csv(SEC_CSV, index=False)

print(f'Обработка завершена: {len(second_df)} записей сохранено в {SEC_CSV}')

# Теперь возвращаем данные, которые могут быть использованы в feature_engineering
order_book = second_df

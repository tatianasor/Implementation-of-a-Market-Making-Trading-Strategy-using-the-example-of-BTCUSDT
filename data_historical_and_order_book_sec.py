import pandas as pd

# Пути к файлам
ORDER_BOOK_CSV = 'order_book_sec_lin.csv'
HISTORICAL_DATA_CSV = 'historical_data_sec_lin.csv'
OUTPUT_CSV = 'data_historical_and_order_book_sec_lin.csv'

# Чтение данных из CSV
order_book_df = pd.read_csv(ORDER_BOOK_CSV)
historical_data_df = pd.read_csv(HISTORICAL_DATA_CSV)

# Выводим общее количество строк в каждом исходном файле
print(f"Общее количество строк в {ORDER_BOOK_CSV}: {len(order_book_df)}")
print(f"Общее количество строк в {HISTORICAL_DATA_CSV}: {len(historical_data_df)}")

# Преобразуем timestamp в datetime
order_book_df['timestamp'] = pd.to_datetime(order_book_df['timestamp'])
historical_data_df['timestamp'] = pd.to_datetime(historical_data_df['timestamp'])

# Проверка на пропущенные временные метки
missing_time_historical = historical_data_df[~historical_data_df['timestamp'].isin(order_book_df['timestamp'])]
missing_time_order_book = order_book_df[~order_book_df['timestamp'].isin(historical_data_df['timestamp'])]

# Выводим информацию о пропущенных временных метках
print(f"Пропущенные временные метки в historical_data: {len(missing_time_historical)}")
print(f"Пропущенные временные метки в order_book: {len(missing_time_order_book)}")

# Заполняем NaN значения в order_book_df (например, заполняем предыдущими значениями)
order_book_df[['bid_price', 'bid_qty', 'ask_price', 'ask_qty']] = order_book_df[['bid_price', 'bid_qty', 'ask_price', 'ask_qty']].ffill()

# Объединяем два датафрейма по столбцу timestamp (используем left join от historical_data)
merged_df = pd.merge(historical_data_df, order_book_df, on='timestamp', how='left')

# Сохраняем объединенные данные в новый CSV файл
merged_df.to_csv(OUTPUT_CSV, index=False)

# Выводим общее количество строк в объединенном файле
print(f'Обработка завершена: {len(merged_df)} записей сохранено в {OUTPUT_CSV}')

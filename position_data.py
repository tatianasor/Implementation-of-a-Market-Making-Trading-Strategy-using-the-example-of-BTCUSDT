import config
from pybit.unified_trading import HTTP
import csv
from market_making_stoikov import inventory

# Использование значений API-ключа и секрета из файла config.py
session = HTTP(
    testnet=True,
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
)

# Запрос баланса
balance_data = session.get_wallet_balance(accountType="UNIFIED", coin="BTC")
try:
    total_available_balance = balance_data['result']['list'][0]['totalAvailableBalance']
    print(f"Total Available Balance: {total_available_balance} USD")
except (KeyError, IndexError):
    print("Ошибка: Не удалось получить totalAvailableBalance")
    total_available_balance = None  # Добавляем обработку ошибки


# Запрос баланса
try:
    # Достаем equity из coin в list
    inventory = next(
        item["equity"] for item in balance_data['result']['list'][0]['coin'] if item["coin"] == "BTC"
    )
    print(f"Equity: {inventory} BTC")
except (KeyError, IndexError, StopIteration):
    print("Ошибка: Не удалось получить equity")

# Запись данных в новый CSV-файл
with open('trading_result.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Total Available Balance (USD)', 'Inventory (BTC)'])
    if total_available_balance is not None and inventory is not None:
        writer.writerow([total_available_balance, inventory])


print(session.get_wallet_balance(
    accountType="UNIFIED",
    coin="BTC",
))



from pybit.unified_trading import HTTP
session = HTTP(
    testnet=True,
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
)
print(session.get_wallet_balance(
    accountType="UNIFIED",
    coin="BTC",
))


print(session.get_positions(
    category="linear",
    symbol="BTCUSDT",
))


position_data = session.get_positions(category="linear", symbol="BTCUSDT")
try:
    avgPrice = position_data['result']['list'][0]['avgPrice']
    print(f"avgPrice: {avgPrice} USD")
except (KeyError, IndexError):
    print("Ошибка: Не удалось получить avgPrice")
    avgPrice = None


import pandas as pd
import numpy as np
import random

# Константы
quantity = 0.001
gamma = 1.0
T = 1
k = 1.5
max_inventory = 0.1
spread_multiplier = 1.0

# Cartea параметры
phi = 0.005
kappa = 200
q_target = 0
lamb = 0.729211

# Загрузка данных
df = pd.read_csv("combined_output.csv")
df = df.reset_index(drop=True)
df = df.iloc[60:].reset_index(drop=True)

# Инициализация
stoikov_log, cartea_log = [], []
stoikov_inventory, stoikov_cash = 0, 0
cartea_inventory, cartea_cash = 0, 0

# MAE
df['mae'] = np.abs(df['y_true'] - df['y_pred'])
mae_percentiles = df['mae'].quantile([0.5, 0.6, 0.7, 0.8, 0.9])

for i in range(30, len(df)):
    window = df.iloc[i - 30:i]
    current = df.iloc[i]
    predicted_mid_price = current['y_pred']
    true_price = current['y_true']
    error = predicted_mid_price - true_price
    mae = abs(error)

    # === Общие показатели ===
    sigma = (window['mid_price'] / window['mid_price'].shift(1)).dropna().std()

    # ===== STOIKOV =====

    r_price_stoikov = predicted_mid_price - stoikov_inventory * gamma * sigma ** 2 * T
    base_spread = gamma * sigma ** 2 * T + (2 / gamma) * np.log(1 + gamma / k)
    spread = base_spread / 2
    if r_price_stoikov >= predicted_mid_price:
        ask_spread = spread + (r_price_stoikov - predicted_mid_price)
        bid_spread = spread - (r_price_stoikov - predicted_mid_price)
    else:
        ask_spread = spread - (predicted_mid_price - r_price_stoikov)
        bid_spread = spread + (predicted_mid_price - r_price_stoikov)

    # === Cartea: динамический λ === bid_size and ask_size
    d_omega_dq = phi * (cartea_inventory - q_target)
    denominator = phi + d_omega_dq if phi + d_omega_dq > 0 else phi
    log_term = np.log(lamb / denominator) if lamb > 0 else 0
    cartea_spread = max(0, log_term / kappa)
    cartea_bid_spread = cartea_spread
    cartea_ask_spread = cartea_spread


    # === Общая логика размещения заявок для обеих стратегий ===
    base_prob = 0.7
    bid_prob = ask_prob = base_prob

    mae_50, mae_60, mae_70, mae_80, mae_90 = [mae_percentiles[q] for q in [0.5, 0.6, 0.7, 0.8, 0.9]]
    if error > 0:
        if mae > mae_90:
            ask_prob *= 0.6
        elif mae > mae_80:
            ask_prob *= 0.7
        elif mae > mae_70:
            ask_prob *= 0.8
        elif mae > mae_60:
            ask_prob *= 0.85
        elif mae > mae_50:
            ask_prob *= 0.9
    elif error < 0:
        if mae > mae_90:
            bid_prob *= 0.6
        elif mae > mae_80:
            bid_prob *= 0.7
        elif mae > mae_70:
            bid_prob *= 0.8
        elif mae > mae_60:
            bid_prob *= 0.85
        elif mae > mae_50:
            bid_prob *= 0.9

    # === Цены заявок ===
    stoikov_bid = predicted_mid_price - bid_spread * spread_multiplier
    stoikov_ask = predicted_mid_price + ask_spread * spread_multiplier
    cartea_bid = predicted_mid_price - cartea_bid_spread
    cartea_ask = predicted_mid_price + cartea_ask_spread

    # === Эмуляция сделок (Stoikov) ===
    executed = False
    if random.random() < bid_prob and current['low'] <= stoikov_bid:
        stoikov_inventory += quantity
        stoikov_cash -= stoikov_bid * quantity
        executed = True
    if random.random() < ask_prob and current['high'] >= stoikov_ask:
        stoikov_inventory -= quantity
        stoikov_cash += stoikov_ask * quantity
        executed = True

    stoikov_log.append({
        'timestamp': current['timestamp'],
        'bid_price': stoikov_bid,
        'ask_price': stoikov_ask,
        'inventory': stoikov_inventory,
        'cash': stoikov_cash,
        'total_pnl': stoikov_cash + stoikov_inventory * predicted_mid_price
    })

    # === Эмуляция сделок (Cartea) ===
    executed = False
    if random.random() < bid_prob and current['low'] <= cartea_bid:
        cartea_inventory += quantity
        cartea_cash -= cartea_bid * quantity
        executed = True
    if random.random() < ask_prob and current['high'] >= cartea_ask:
        cartea_inventory -= quantity
        cartea_cash += cartea_ask * quantity
        executed = True

    cartea_log.append({
        'timestamp': current['timestamp'],
        'bid_price': cartea_bid,
        'ask_price': cartea_ask,
        'inventory': cartea_inventory,
        'cash': cartea_cash,
        'total_pnl': cartea_cash + cartea_inventory * predicted_mid_price
    })

# === Сохранение результатов ===
pd.DataFrame(stoikov_log).to_csv("backtest_stoikov2.csv", index=False)
pd.DataFrame(cartea_log).to_csv("backtest_cartea2.csv", index=False)

print("Backtest завершен. Результаты сохранены в backtest_stoikov.csv и backtest_cartea.csv")

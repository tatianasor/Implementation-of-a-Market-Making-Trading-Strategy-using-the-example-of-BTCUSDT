import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna

# === Настройки ===
target_column = 'mid_price'
csv_filename = "predictions_lstm_torch.csv"
feature_sets = ['lasso', 'rf', 'pca']
lookback = 60  # Размер окна (количество временных шагов)
batch_size = 32
epochs = 30
learning_rate = 0.0001
patience = 5  # Параметр для ранней остановки

# === Метрики ===
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R²': r2}

# === Проверка CSV для предсказаний ===
if not os.path.exists(csv_filename):
    with open(csv_filename, "w") as f:
        f.write("model,feature_set,y_true,y_pred\n")

# === Подготовка данных для LSTM ===
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X, self.y = [], []
        for i in range(len(X) - lookback):
            self.X.append(X[i:i + lookback])
            self.y.append(y[i + lookback])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# === Создание модели LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# === Инициализация весов модели ===
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)

# === Обучение и оценка ===
def train_evaluate_lstm(X_train, y_train, X_val, y_val, X_test, y_test, prefix, optuna_trial=None):
    results = []

    print("Train data stats:")
    print(y_train.describe())
    print("\nValidation data stats:")
    print(y_val.describe())
    print("\nTest data stats:")
    print(y_test.describe())

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    y_train_scaled += np.random.normal(0, 0.01, size=y_train_scaled.shape)
    y_val_scaled += np.random.normal(0, 0.01, size=y_val_scaled.shape)
    y_test_scaled += np.random.normal(0, 0.01, size=y_test_scaled.shape)

    # Вывод статистики после масштабирования
    print("\nScaled Train Data Stats:")
    print(pd.Series(y_train_scaled).describe())
    print("\nScaled Validation Data Stats:")
    print(pd.Series(y_val_scaled).describe())
    print("\nScaled Test Data Stats:")
    print(pd.Series(y_test_scaled).describe())

    # Создание датасетов
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, lookback)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, lookback)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, lookback)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Определение гиперпараметров
    input_size = X_train.shape[1]
    hidden_size = optuna_trial.suggest_int("hidden_size", 32, 128, step=32) if optuna_trial else 64
    num_layers = optuna_trial.suggest_int("num_layers", 1, 3) if optuna_trial else 1
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.5)
    model.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Обучение модели с ранней остановкой
    best_val_loss = float('inf')
    no_improvement = 0

    print(f"\n⏳ Обучение LSTM ({prefix})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), f"best_model_{prefix}.pth")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f"best_model_{prefix}.pth"))
    model.eval()

    # Предсказания на тестовой выборке
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())

    y_true_rescaled = y_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
    y_pred_rescaled = y_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

    test_metrics = calculate_metrics(y_true_rescaled, y_pred_rescaled)

    predictions_df = pd.DataFrame({
        "model": ["LSTM"] * len(y_true_rescaled),
        "feature_set": [prefix] * len(y_true_rescaled),
        "y_true": y_true_rescaled,
        "y_pred": y_pred_rescaled
    })
    predictions_df.to_csv(csv_filename, mode='a', header=False, index=False)

    print(f"📊 LSTM на {prefix}: R²={test_metrics['R²']:.4f}, MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

    results.append({
        'Model': 'LSTM',
        'Test_RMSE': test_metrics['RMSE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_R²': test_metrics['R²']
    })

    return results

# === Оптимизация с помощью Optuna ===
def optimize_lstm(X_train, y_train, X_val, y_val, X_test, y_test, prefix):
    def objective(trial):
        results = train_evaluate_lstm(X_train, y_train, X_val, y_val, X_test, y_test, prefix, optuna_trial=trial)
        return results[0]['Test_RMSE']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1)

    print(f"Best trial: {study.best_trial.params}")
    return study.best_trial

# === Основной цикл по фичсетам ===
all_results = {}

for prefix in feature_sets:
    print(f"\n=== Обработка фичей: {prefix.upper()} ===")

    try:
        df_train = pd.read_csv(f'selected_features_{prefix}_train.csv')
        df_val = pd.read_csv(f'selected_features_{prefix}_val.csv')
        df_test = pd.read_csv(f'selected_features_{prefix}_test.csv')
    except FileNotFoundError:
        print(f"❌ Файлы для {prefix} не найдены. Пропуск...")
        continue

    y_train = df_train[target_column]
    y_val = df_val[target_column]
    y_test = df_test[target_column]

    X_train = df_train.drop(columns=[target_column])
    X_val = df_val.drop(columns=[target_column])
    X_test = df_test.drop(columns=[target_column])

    # Оптимизация гиперпараметров
    best_trial = optimize_lstm(X_train, y_train, X_val, y_val, X_test, y_test, prefix)

    # Повторное обучение модели с лучшими параметрами
    final_results = train_evaluate_lstm(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        prefix,
        optuna_trial=best_trial
    )

    # Сохраняем метрики в словарь
    all_results[prefix.upper()] = final_results[0]  # results — это список из одного словаря

# === Сохраняем метрики ===
with pd.ExcelWriter('lstm_optimization_results_torch.xlsx') as writer:
    for name, result in all_results.items():
        df = pd.DataFrame([result])
        df.set_index('Model', inplace=True)
        sheet_name = f'{name[:28]}_Results'
        df.to_excel(writer, sheet_name=sheet_name)

print("\n✅ Обучение завершено. Метрики сохранены. Предсказания добавлены в predictions_lstm_torch.csv.")

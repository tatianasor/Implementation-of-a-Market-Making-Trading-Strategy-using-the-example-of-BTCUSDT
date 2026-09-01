import os
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution
from ngboost import NGBRegressor

# === Настройки ===
target_column = 'mid_price'
csv_filename = "predictions.csv"
feature_sets = ['lasso', 'rf', 'pca']


# === Метрики ===
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R²': r2}


# === Проверка CSV для предсказаний ===
if not os.path.exists(csv_filename):
    with open(csv_filename, "w") as f:
        f.write("model,feature_set,y_true,y_pred\n")


# === Кросс-валидация с Walk-Forward ===
def walk_forward_split(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# === Подбор гиперпараметров с Optuna ===
def optuna_search(model, X, y, param_distributions):
    tscv = TimeSeriesSplit(n_splits=5)
    optuna_search = OptunaSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=30,
        n_trials=20,
        random_state=42,
        verbose=0
    )
    optuna_search.fit(X, y)
    return optuna_search.best_estimator_, optuna_search.best_params_


# === Nested Cross-Validation ===
def nested_cross_val(X, y, base_model, param_dist, outer_splits=3, inner_splits=3):
    outer_cv = TimeSeriesSplit(n_splits=outer_splits)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\n🔁 Nested Fold {fold + 1}/{outer_splits}")

        best_model, _ = optuna_search(base_model, X_train, y_train, param_dist)
        y_pred = best_model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        print(f"📈 Fold {fold + 1} | R²={metrics['R²']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
        all_metrics.append(metrics)

    print("\n📊 Средние метрики по Nested CV:")
    for metric in ['RMSE', 'MSE', 'MAE', 'R²']:
        scores = [m[metric] for m in all_metrics]
        print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


# === Обучение и оценка ===
def train_evaluate_save(X_train, y_train, X_val, y_val, X_test, y_test, prefix):
    results = []

    # Сохраняем имена признаков
    feature_names = X_train.columns.tolist()

    # Объединяем обучающую и валидационную выборки
    full_X = pd.concat([X_train, X_val])
    full_y = pd.concat([y_train, y_val])

    # Масштабирование данных
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    full_X_scaled = scaler.fit_transform(full_X)
    X_test_scaled = scaler.transform(X_test)

    # Преобразуем обратно в DataFrame с сохранением имен признаков
    full_X_scaled = pd.DataFrame(full_X_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    model_configs = {
        'LightGBM': (
            lgb.LGBMRegressor(verbose=-1),
            {
                'n_estimators': IntDistribution(100, 1000),
                'learning_rate': FloatDistribution(0.001, 0.1),
                'max_depth': IntDistribution(3, 15),
                'num_leaves': IntDistribution(20, 100),
                'min_child_samples': IntDistribution(5, 50)
            }
        ),
        'CatBoost': (
            cb.CatBoostRegressor(verbose=0),
            {
                'iterations': IntDistribution(200, 1000),
                'learning_rate': FloatDistribution(0.001, 0.1),
                'depth': IntDistribution(4, 10)
            }
        ),
        'RandomForest': (
            RandomForestRegressor(),
            {
                'n_estimators': IntDistribution(100, 500),
                'max_depth': IntDistribution(5, 20),
                'min_samples_split': IntDistribution(2, 10),
                'min_samples_leaf': IntDistribution(1, 5)
            }
        ),
        'XGBoost': (
            xgb.XGBRegressor(verbosity=0),
            {
                'n_estimators': IntDistribution(100, 1000),
                'learning_rate': FloatDistribution(0.001, 0.1),
                'max_depth': IntDistribution(3, 15),
                'subsample': FloatDistribution(0.5, 1.0),
                'colsample_bytree': FloatDistribution(0.5, 1.0)
            }
        ),
        'NGBoost': (
            NGBRegressor(verbose=False),
            {
                'n_estimators': IntDistribution(100, 1000),
                'learning_rate': FloatDistribution(0.001, 0.1),
                'minibatch_frac': FloatDistribution(0.5, 1.0)
            }
        )
    }

    for model_name, (base_model, param_dist) in model_configs.items():
        print(f"\n🔍 Optuna-поиск для {model_name} ({prefix})...")
        model, best_params = optuna_search(base_model, full_X_scaled, full_y, param_dist)

        print(f"✅ Лучшие параметры {model_name}: {best_params}")

        # Обучаем модель на полных данных
        model.fit(full_X_scaled, full_y)

        # Предсказания на тестовой выборке
        y_pred = model.predict(X_test_scaled)
        test_metrics = calculate_metrics(y_test, y_pred)

        # Сохраняем модель
        model_path = f'{prefix}_{model_name.lower()}_model'
        if isinstance(model, lgb.LGBMRegressor):
            model.booster_.save_model(f'{model_path}.txt')
        elif isinstance(model, cb.CatBoostRegressor):
            model.save_model(f'{model_path}.cbm')
        else:
            joblib.dump(model, f'{model_path}.pkl')

        # Записываем предсказания в CSV
        predictions_df = pd.DataFrame({
            "model": [model_name] * len(y_test),
            "feature_set": [prefix] * len(y_test),
            "y_true": y_test.values,
            "y_pred": y_pred
        })
        predictions_df.to_csv(csv_filename, mode='a', header=False, index=False)

        # Выводим метрики
        print(
            f"📊 {model_name} на {prefix}: R²={test_metrics['R²']:.4f}, MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}")

        # Сохраняем результаты
        results.append({
            'Model': model_name,
            'Best Params': best_params,
            'Test_RMSE': test_metrics['RMSE'],
            'Test_MAE': test_metrics['MAE'],
            'Test_R²': test_metrics['R²']
        })

    return results


# === Основной цикл по фичсетам ===
all_results = {}

for prefix in feature_sets:
    print(f"\n=== Обработка фичей: {prefix.upper()} ===")

    try:
        df_train = pd.read_csv(f'selected_features_{prefix}_train.csv')
        df_val = pd.read_csv(f'selected_features_{prefix}_val.csv')
        df_test = pd.read_csv(f'selected_features_{prefix}_test.csv')
    except FileNotFoundError:
        print(f"Файлы для {prefix} не найдены. Пропуск...")
        continue

    y_train = df_train[target_column]
    y_val = df_val[target_column]
    y_test = df_test[target_column]

    X_train = df_train.drop(columns=[target_column])
    X_val = df_val.drop(columns=[target_column])
    X_test = df_test.drop(columns=[target_column])

    # nested_cross_val можно вызвать, например, так:
    # nested_cross_val(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), lgb.LGBMRegressor(), {...}, 3, 3)

    results = train_evaluate_save(X_train, y_train, X_val, y_val, X_test, y_test, prefix)
    all_results[prefix.upper()] = results

# === Сохраняем метрики ===
with pd.ExcelWriter('model_optimization_results.xlsx') as writer:
    for name, result in all_results.items():
        df = pd.DataFrame(result)
        df.set_index('Model', inplace=True)
        sheet_name = f'{name[:28]}_Results'
        df.to_excel(writer, sheet_name=sheet_name)

print("\n✅ Обучение завершено. Метрики сохранены. Предсказания добавлены в predictions.csv.")

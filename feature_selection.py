import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import os

# === Загрузка данных ===
data = pd.read_csv('features_all_BTCUSDT_sec_lin.csv')

# Смещаем целевую переменную на один шаг вперёд
mid_price = data['mid_price'].shift(-1).dropna()
X = data.iloc[:-1].drop(columns=['timestamp', 'mid_price'], errors='ignore')

# Разделение на train/val/test
split1 = int(len(X) * 0.7)
split2 = int(len(X) * 0.85)

X_train, y_train = X.iloc[:split1], mid_price.iloc[:split1]
X_val, y_val = X.iloc[split1:split2], mid_price.iloc[split1:split2]
X_test, y_test = X.iloc[split2:], mid_price.iloc[split2:]

# === Корреляционный фильтр на train ===
def filter_correlated_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop, errors='ignore'), to_drop

X_train_filtered, dropped_corr = filter_correlated_features(X_train)

# Применяем тот же список фич к val и test
X_val_filtered = X_val[X_train_filtered.columns]
X_test_filtered = X_test[X_train_filtered.columns]

summary = {
    'Correlation': {
        'Removed': len(dropped_corr),
        'Remaining': X_train_filtered.shape[1],
        'Remaining Features': list(X_train_filtered.columns)
    }
}

# === Random Forest Importance ===
rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
rf.fit(X_train_filtered, y_train)
importance_rf = pd.Series(rf.feature_importances_, index=X_train_filtered.columns).sort_values(ascending=False)
top_rf = importance_rf.head(10).index

summary['RandomForest'] = {
    'Selected': len(top_rf),
    'Top features': list(top_rf),
    'Remaining Features': list(X_train_filtered.columns)
}

# Сохраняем выборки RF
for part, X_part, y_part in [('train', X_train_filtered, y_train),
                             ('val', X_val_filtered, y_val),
                             ('test', X_test_filtered, y_test)]:
    df_rf = pd.concat([y_part.reset_index(drop=True), X_part[top_rf].reset_index(drop=True)], axis=1)
    df_rf.to_csv(f'selected_features_rf_{part}.csv', index=False)

# === Lasso ===
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_filtered, y_train)
importance_lasso = pd.Series(lasso.coef_, index=X_train_filtered.columns)
selected_lasso = importance_lasso[importance_lasso != 0].sort_values(ascending=False).index

summary['Lasso'] = {
    'Selected': len(selected_lasso),
    'Top features': list(selected_lasso[:10]),
    'Remaining Features': list(X_train_filtered.columns)
}

# Сохраняем выборки Lasso
for part, X_part, y_part in [('train', X_train_filtered, y_train),
                             ('val', X_val_filtered, y_val),
                             ('test', X_test_filtered, y_test)]:
    df_lasso = pd.concat([y_part.reset_index(drop=True), X_part[selected_lasso].reset_index(drop=True)], axis=1)
    df_lasso.to_csv(f'selected_features_lasso_{part}.csv', index=False)

# === PCA ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var = np.cumsum(pca.explained_variance_ratio_)
summary['PCA'] = {
    'Explained variance (%)': [round(r * 100, 2) for r in explained_var],
    'Remaining Features': [f'PC{i + 1}' for i in range(X_train_pca.shape[1])]
}

# Сохраняем выборки PCA
pca_columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
for part, X_part_pca, y_part in [('train', X_train_pca, y_train),
                                 ('val', X_val_pca, y_val),
                                 ('test', X_test_pca, y_test)]:
    df_pca = pd.concat([y_part.reset_index(drop=True),
                        pd.DataFrame(X_part_pca, columns=pca_columns)], axis=1)
    df_pca.to_csv(f'selected_features_pca_{part}.csv', index=False)

# === Сохраняем сводку ===
print("\n=== Сводка по отборам признаков ===")
for method, stats in summary.items():
    print(f"\nМетод: {method}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

summary_df = pd.DataFrame.from_dict(summary, orient='index')
summary_df.to_excel('feature_selection_summary.xlsx')

print("\nВсе результаты сохранены.")

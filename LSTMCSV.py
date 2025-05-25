import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# === 核心逻辑：行业模型 + 超额收益 + tanh + EWM + 图像真实还原 ===

# Step 1: 加载与筛选数据
df = pd.read_csv("data/interim/cleaned/cleaned_all_companies_long.csv")
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values(['Company', 'Dates'])

train_companies = ["ADBE", "AMD", "AMZN", "AVGO", "CSCO", "GOOGL", "IBM", "INTC", "META", "MSFT", "NVDA"]
all_companies = train_companies + ["CRM", "ORCL", "QCOM", "TXN"]
df = df[df['Company'].isin(all_companies)]

# Step 2: 加载因子
factors_df = pd.read_csv("data/result/selected_factors.csv", index_col=0)
features = factors_df.index.tolist()

# Step 3: 构造目标变量（超额收益 + EWM）
df['log_return'] = df.groupby('Company')['PX_LAST'].transform(lambda x: np.log(x).diff())
df['market_return'] = df.groupby('Dates')['log_return'].transform('mean')
df['excess_return'] = df['log_return'] - df['market_return']
df['target'] = (
    df.groupby('Company')['excess_return']
    .transform(lambda x: x.shift(-1).ewm(span=20).mean())
    .shift(-9).clip(-0.3, 0.3)
)

# Step 4: 衍生变量补充
if 'VIX_PE_interact' in features and 'VIX_PE_interact' not in df.columns:
    df['VIX_PE_interact'] = df['VIX Index'] * df['PE_RATIO']
if 'volatility_10d' in features and 'volatility_10d' not in df.columns:
    df['volatility_10d'] = df.groupby('Company')['PX_LAST'].rolling(10).std().reset_index(level=0, drop=True)
if 'return_lag1' in features and 'return_lag1' not in df.columns:
    df['return_lag1'] = df.groupby('Company')['log_return'].shift(1)

df = df.dropna(subset=features + ['target'])

# Step 5: 训练测试划分
train_df = df[(df['Company'].isin(train_companies)) & (df['Dates'].dt.year <= 2023)]
test_df = df[df['Dates'].dt.year == 2024]

# Step 6: 标准化 + 归一化
scaler = StandardScaler()
target_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])
y_train = target_scaler.fit_transform(train_df[['target']]).flatten()
y_test = target_scaler.transform(test_df[['target']]).flatten()

# Step 7: 滑窗序列
WINDOW_SIZE = 30
def create_sequences(X, y, window=30):
    return np.array([X[i:i+window] for i in range(len(X)-window)]), np.array([y[i+window] for i in range(len(y)-window)])
X_train_seq, y_train_seq = create_sequences(X_train, y_train, WINDOW_SIZE)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, WINDOW_SIZE)

# Step 8: 构建 LSTM 模型
tf.random.set_seed(42)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, len(features))),
    BatchNormalization(), Dropout(0.15),
    LSTM(32),
    Dense(64, activation='relu'), Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh')  
])
model.compile(optimizer=Adam(0.0003), loss='mse', metrics=['mae'])

# Step 9: 训练模型
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.1,
    epochs=80, batch_size=16,
    callbacks=[es], verbose=2
)

# Step 10: 每家公司输出评估与图像
import os, matplotlib.pyplot as plt
model_path = "data/result/final_mixed_model"
result_dir = "data/result/final_mixed_company_result"
os.makedirs(model_path, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
model.save(os.path.join(model_path, "lstm_final_mixed_model.h5"))

company_metrics = []
for company in test_df['Company'].unique():
    df_c = test_df[test_df['Company'] == company].copy().sort_values('Dates')
    X_c = scaler.transform(df_c[features])
    y_c = df_c['target'].values
    y_c_scaled = target_scaler.transform(y_c.reshape(-1, 1)).flatten()
    X_c_seq, y_c_seq = create_sequences(X_c, y_c_scaled, WINDOW_SIZE)

    if len(X_c_seq) == 0: continue
    y_pred_scaled = model.predict(X_c_seq).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(np.array(y_c_seq).reshape(-1, 1)).flatten()

    # 保存预测值 CSV，以便后续回测选股
    date_seq = df_c['Dates'].iloc[WINDOW_SIZE:].reset_index(drop=True)
    prediction_df = pd.DataFrame({
        'Dates': date_seq,
        'Real': y_true,
        'Predicted': y_pred
    })
    prediction_df.to_csv(os.path.join(result_dir, f"{company}_prediction.csv"), index=False)

    # 评估与图像
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    company_metrics.append({'Company': company, 'MAE': mae, 'R2': r2, 'Direction_Accuracy': acc})

    # 图像保存
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Real", linewidth=1.5, color='royalblue')
    plt.plot(y_pred, label="Predicted", linewidth=1.5, color='darkorange')
    plt.fill_between(range(len(y_true)), y_true, y_pred, color='gray', alpha=0.2)
    plt.title(f"Mixed LSTM Prediction vs Real - {company}", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{company}_final_mixed_prediction.png"))
    plt.close()

# 保存评估结果
pd.DataFrame(company_metrics).to_csv(os.path.join(result_dir, "company_final_mixed_metrics.csv"), index=False)
print(f"最终模型与图像结果已保存：{result_dir}")
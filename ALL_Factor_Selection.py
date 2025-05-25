import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# === Step 1: 加载清洗数据 ===
df = pd.read_csv("data/interim/cleaned/cleaned_all_companies_long.csv")
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
df = df.sort_values(['Company', 'Dates'])

# === Step 2: 构造衍生特征（基于历史数据，无未来泄漏）===
df['return_lag1'] = df.groupby('Company')['PX_LAST'].pct_change(1)
df['momentum_5d'] = df.groupby('Company')['PX_LAST'].pct_change(5)
df['momentum_10d'] = df.groupby('Company')['PX_LAST'].pct_change(10)
df['volatility_5d'] = df.groupby('Company')['PX_LAST'].rolling(5).std().reset_index(level=0, drop=True)
df['volatility_10d'] = df.groupby('Company')['PX_LAST'].rolling(10).std().reset_index(level=0, drop=True)
df['PE_log'] = np.log1p(df['PE_RATIO'])
df['VIX_PE_interact'] = df['VIX Index'] * df['PE_RATIO']
df['USDJPY_ret'] = df['USDJPY Curncy'].pct_change(1)

# === Step 3: 设定训练公司（用于选因子 + 模型训练）===
train_companies = [
    "ADBE", "AMD", "AMZN", "AVGO", "CSCO", "GOOGL", "IBM",
    "INTC", "META", "MSFT", "NVDA", "CRM"
]  # 🧠 AAPL, ORCL, TXN 保留用于泛化验证

# === Step 4: 筛选训练数据（公司 + 年份 2020–2022）===
train_df = df[
    (df['Company'].isin(train_companies)) &
    (df['Dates'].dt.year <= 2022)
].copy()

# === Step 5: 构造目标变量（10日对数收益滑动平均）===
train_df['log_return'] = train_df.groupby('Company')['PX_LAST'].transform(lambda x: np.log(x).diff())
train_df['return'] = (
    train_df.groupby('Company')['log_return']
    .shift(-1)
    .rolling(10)
    .mean()
    .shift(-9)
    .clip(-0.3, 0.3)
)
train_df = train_df.dropna(subset=['return'])

# === Step 6: 特征列（排除元数据列）===
exclude_cols = ['Dates', 'Company', 'PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW',
                'PX_VOLUME.1', 'log_return', 'return']
features = [col for col in train_df.columns if col not in exclude_cols]

# === Step 7: 特征标准化 + 随机森林选因子 ===
X = train_df[features].fillna(0)
y = train_df['return'].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_scaled, y)

# === Step 8: 提取并保存 Top 20 因子 ===
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
top_20_factors = importances.head(20)

print("📌 行业统一选出的 Top20 因子：")
print(top_20_factors)

top_20_factors.to_frame(name='importance').to_csv("data/result/selected_factors.csv", index=True)
print("已保存 Top 20 因子到 data/result/selected_factors.csv")

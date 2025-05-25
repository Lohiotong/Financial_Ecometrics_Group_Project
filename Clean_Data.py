import pandas as pd

# === Step 1: 读入原始数据 ===
df = pd.read_csv("data/interim/transformed/all_companies_long_sorted.csv")
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')

# === Step 2: 删除缺失值比例 > 30% 的列 ===
missing_ratio = df.isnull().mean()
df = df.drop(columns=missing_ratio[missing_ratio > 0.3].index)
print(f"删除高缺失列后剩余列数: {df.shape[1]}")

# === Step 3: 数值列填充中位数 ===
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# === Step 4: 删除日期/公司名为空的行（极少）===
df = df.dropna(subset=['Dates', 'Company'])

# === Step 5: 排序（便于后续滑窗） ===
df = df.sort_values(by=['Company', 'Dates']).reset_index(drop=True)

# === Step 6: 保存清洗后的数据 ===
df.to_csv("data/interim/cleaned/cleaned_all_companies_long.csv", index=False)
print("已保存清洗后的数据：data/interim/cleaned/cleaned_all_companies_long.csv")

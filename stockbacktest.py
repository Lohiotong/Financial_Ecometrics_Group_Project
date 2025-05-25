import pandas as pd
import os
import matplotlib.pyplot as plt

# === 参数设置 ===
result_dir = "data/result/final_mixed_company_result"
metric_path = os.path.join(result_dir, "company_final_mixed_metrics.csv")
threshold = 0.001  # 选股阈值

# === Step 1: 读取公司评估指标 ===
metrics_df = pd.read_csv(metric_path)

# === Step 2: 回测函数 ===
def backtest_stock(df, threshold=0.001):
    df = df.sort_values('Dates').reset_index(drop=True)
    df['Signal'] = (df['Predicted'] > threshold).astype(int)
    df['Strategy_Return'] = df['Real'] * df['Signal']
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Benchmark'] = (1 + df['Real']).cumprod()
    return df

# === Step 3: 汇总全部回测结果 ===
selected = []
results = []
all_results = []

for fname in os.listdir(result_dir):
    if not fname.endswith("_prediction.csv"):
        continue

    company = fname.split("_")[0]
    df_pred = pd.read_csv(os.path.join(result_dir, fname))
    if not {'Dates', 'Real', 'Predicted'}.issubset(df_pred.columns):
        continue

    df_pred['Dates'] = pd.to_datetime(df_pred['Dates'])
    df_backtest = backtest_stock(df_pred, threshold=threshold)

    # 保存图像
    plt.figure(figsize=(10, 4))
    plt.plot(df_backtest['Dates'], df_backtest['Cumulative_Benchmark'], label='Benchmark', color='gray', alpha=0.6)
    plt.plot(df_backtest['Dates'], df_backtest['Cumulative_Strategy'], label='Strategy', color='blue')
    plt.title(f"{company} Backtest")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{company}_backtest.png"))
    plt.close()

    # 收益指标
    final_benchmark = df_backtest['Cumulative_Benchmark'].iloc[-1]
    final_strategy = df_backtest['Cumulative_Strategy'].iloc[-1]
    outperformance = final_strategy - final_benchmark

    results.append({
        'Company': company,
        'Final_Strategy': final_strategy,
        'Final_Benchmark': final_benchmark,
        'Outperformance': outperformance
    })

    if outperformance > 0:
        selected.append(company)

    # === 合并公司日级值 ===
    df_temp = df_backtest[['Dates', 'Cumulative_Strategy', 'Cumulative_Benchmark']].copy()
    df_temp['Company'] = company
    df_temp.rename(columns={
        'Cumulative_Strategy': 'Strategy_Value',
        'Cumulative_Benchmark': 'Benchmark_Value'
    }, inplace=True)
    all_results.append(df_temp)

# === Step 4: 保存回测表现 ===
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(result_dir, "backtest_summary.csv"), index=False)

df_selected = pd.DataFrame({'Selected_Stocks': selected})
df_selected.to_csv(os.path.join(result_dir, "selected_stocks.csv"), index=False)

print("\u2705 \u56de\u6d4b\u5b8c\u6210，结果保存在：", result_dir)

# Step 5: 构造全公司预测净值序列
df_all_predictions = pd.concat(all_results)  # all_results 中每项是 df，每家公司含 Strategy_Value, Benchmark_Value

# 创建组合净值（只对 selected 股票）
selected = pd.read_csv("data/result/final_mixed_company_result/selected_stocks.csv")['Selected_Stocks'].tolist()
portfolio_df = df_all_predictions[df_all_predictions['Company'].isin(selected)].copy()

# 每日平均净值
portfolio_df['Strategy_Group'] = portfolio_df.groupby('Dates')['Strategy_Value'].transform('mean')
portfolio_df['Benchmark_Group'] = portfolio_df.groupby('Dates')['Benchmark_Value'].transform('mean')
portfolio_df_daily = portfolio_df[['Dates', 'Strategy_Group', 'Benchmark_Group']].drop_duplicates()

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(portfolio_df_daily['Dates'], portfolio_df_daily['Strategy_Group'], label='LSTM Strategy Portfolio', linewidth=2)
plt.plot(portfolio_df_daily['Dates'], portfolio_df_daily['Benchmark_Group'], label='Benchmark Portfolio', linewidth=2, linestyle='--')
plt.title("LSTM strategy profile vs Benchmark Portfolio net value")
plt.xlabel("Date")
plt.ylabel("Net Value")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("data/result/portfolio_net_value.png")
plt.show()
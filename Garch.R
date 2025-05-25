# 安装并加载必要包
library(quantmod)
library(rugarch)
library(xts)
library(patchwork)

# 读取数据
data <- read.csv('/data/interim/cleaned/cleaned_all_companies_long.csv')
data$Dates <- as.Date(data$Dates)

# 公司列表
company_names <- c('ORCL', 'QCOM', 'TXN')

# 创建图形列表
plot_list<-list()

# 循环处理每家公司
for (company in company_names) {
  # 提取该公司数据
  company_data <- subset(data, Company == company)
  price <- company_data[, c('Dates', 'PX_LAST')]
  
  # 分割训练集和测试集
  price_train <- subset(price, Dates >= as.Date('2020-01-01') & Dates < as.Date('2024-01-01'))
  price_test  <- subset(price, Dates >= as.Date('2024-01-01') & Dates < as.Date('2025-01-01'))
  
  # 转换为对数收益率序列
  price_xts <- xts(price_train$PX_LAST, order.by = price_train$Dates)
  log_ret <- diff(log(price_xts))[-1]
  
  # ----------------------------
  # 自动选择最优 GARCH 模型阶数
  # ----------------------------
  best_aic <- Inf
  best_spec <- NULL
  best_order <- c(0, 0, 0, 0)
  
  for (arma_p in 0:1) {
    for (arma_q in 0:1) {
      for (garch_p in 1:2) {
        for (garch_q in 1:2) {
          try({
            spec_try <- ugarchspec(
              variance.model = list(model = 'sGARCH', garchOrder = c(garch_p, garch_q)),
              mean.model     = list(armaOrder = c(arma_p, arma_q), include.mean = TRUE),
              distribution.model = "std"
            )
            fit_try <- ugarchfit(spec_try, data = log_ret, solver = 'hybrid', silent = TRUE)
            aic_val <- infocriteria(fit_try)[1]
            
            if (aic_val < best_aic) {
              best_aic <- aic_val
              best_spec <- spec_try
              best_order <- c(arma_p, arma_q, garch_p, garch_q)
            }
          }, silent = TRUE)
        }
      }
    }
  }
  
  cat(" 最优模型阶数: ARMA(", best_order[1], ",", best_order[2], "), GARCH(", best_order[3], ",", best_order[4], ")\n")
  cat(" 最小AIC: ", best_aic, "\n")
  
  # ----------------------------
  # 拟合最佳模型并预测
  # ----------------------------
  fit <- ugarchfit(best_spec, data = log_ret)
  n_forecast <- nrow(price_test)
  forecast <- ugarchforecast(fit, n.ahead = n_forecast)
  predicted_ret <- fitted(forecast)
  
  # 从最后一个已知价格开始累积 log return → 预测价格
  last_log_price <- log(as.numeric(tail(price_train$PX_LAST, 1)))
  log_price_path <- cumsum(predicted_ret) + last_log_price
  predicted_price <- exp(log_price_path)
  
  # 构造绘图数据
  plot_df <- data.frame(
    Dates = price_test$Dates,
    Actual = price_test$PX_LAST,
    Predicted = as.numeric(predicted_price)
  )
  
  # 计算RMSE
  rmse <- sqrt(mean((plot_df$Actual - plot_df$Predicted)^2))
  cat(company, " GARCH预测RMSE:", round(rmse, 4), "\n")
  
  # ----------------------------
  # 绘图
  # ----------------------------
  p <- ggplot(plot_df, aes(x = Dates)) +
    geom_line(aes(y = Actual, color = "Actual Price"), linewidth = 1) +
    geom_line(aes(y = Predicted, color = "GARCH Predicted Price"), linetype = "dashed", linewidth = 1) +
    scale_color_manual(values = c("Actual Price" = "black", "GARCH Predicted Price" = "blue")) +
    labs(title = paste0(company, " Actual vs GARCH Predicted Price (2024)\nBest ARMA(", best_order[1], ",", best_order[2], 
                        "), GARCH(", best_order[3], ",", best_order[4], ")"),
         x = "Date", y = "Price", color = "Legend") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5,size=16,face='bold'),
          axis.title = element_text(size = 16),                                
          axis.text = element_text(size = 16),                                           
          legend.title = element_text(size = 14),                              
          legend.text = element_text(size = 14))
  
  # 将图添加到列表中
  plot_list[[company]] <- p
  ggsave(filename = paste0(company, "_GARCH_Best_Model.png"),
         plot = p, width = 18, height = 10, dpi = 300, bg = "white")
}

# ----------------------------
# 拼接图形（从上到下）
# ----------------------------
combined_plot <- plot_list[[1]] / plot_list[[2]] / plot_list[[3]]

# 显示组合图
print(combined_plot)

# 保存组合图
ggsave("GARCH_Combined_Actual_vs_Predicted.png", plot = combined_plot,
       width = 18, height = 10, dpi = 300, bg = "white")

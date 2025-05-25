# 加载必要的库
library(forecast)
library(dplyr)
library(ggplot2)
library(gridExtra)  

# 第 1 步：读取 CSV 数据
data <- read.csv("cleaned_data.csv", stringsAsFactors = FALSE)
data$Dates <- as.Date(data$Dates)

# 第 2 步：选择目标公司
target_companies <- c("ORCL", "QCOM", "TXN")

# 过滤目标公司
data_filtered <- data %>% filter(Company %in% target_companies)

# 创建一个空列表来存储每个公司的图形
plots <- list()

# 第 3 步：对每家公司进行建模和预测
for (company in target_companies) {
  cat("Processing:", company, "\n")
  
  # 提取当前公司数据
  company_data <- data_filtered %>% filter(Company == company)
  
  # 按日期排序（保险起见）
  company_data <- company_data %>% arrange(Dates)
  
  # 划分训练集和测试集
  train_data <- company_data %>% filter(Dates >= as.Date("2020-01-01") & Dates <= as.Date("2023-12-31"))
  test_data <- company_data %>% filter(Dates >= as.Date("2024-01-01"))
  
  train_prices <- train_data$PX_LAST
  test_prices <- test_data$PX_LAST
  
  # 手动指定 ARIMA 模型的 p, d, q 参数
  p <- 1  # 自回归项
  d <- 2  # 差分阶数
  q <- 1  # 移动平均项
  
  # 拟合 ARIMA 模型
  model <- arima(train_prices, order = c(p, d, q))
  
  # 打印模型参数
  cat("Manual ARIMA Parameters for", company, ":", p, d, q, "\n")
  
  # 预测
  h <- length(test_prices)
  fcst <- forecast(model, h = h)
  
  # 构建数据框用于绘图
  full_actual <- data.frame(
    Date = c(train_data$Dates, test_data$Dates),
    Actual = c(train_prices, test_prices)
  )
  
  full_predicted <- data.frame(
    Date = c(train_data$Dates, test_data$Dates),
    Predicted = c(rep(NA, length(train_prices)), fcst$mean)
  )
  
  # 合并实际值与预测值
  plot_data <- full_actual %>%
    left_join(full_predicted, by = "Date") %>%
    mutate(Company = company)
  
  # 绘制图形
  plot <- ggplot(plot_data, aes(x = Date)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1, linetype = "dashed") +
    geom_vline(xintercept = as.numeric(as.Date("2024-01-01")), linetype = "dotdash", color = "gray30") +
    labs(title = paste(company, "- Closing Prices: Actual vs Predicted (ARIMA, 2020–2024)"),
         x = "Date", y = "Closing Price", color = "Legend") +
    scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
    theme_minimal()
  
  # 将当前图形添加到图形列表中
  plots[[company]] <- plot
}

# 使用 grid.arrange() 将所有图形显示在同一张图中
grid.arrange(grobs = plots, ncol = 1)  # ncol=1 表示垂直排列，若想水平排列可以设置 nrow=1

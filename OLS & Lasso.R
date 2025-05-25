# 加载必要包
library(gamlr)
library(ggplot2)
library(Matrix)

# 读取数据
data <- read.csv("data/interim/transformed/all_companies_long_sorted.csv")
selected_feature <- read.csv("/data/result/selected_factors.csv")

# 标准化列名
selected_feature_names <- make.names(selected_feature$X)

# 提取指定特征列 + 添加交互项
data_selected <- data[, selected_feature_names[selected_feature_names %in% names(data)], drop = FALSE]
data_selected$VIX_PE_interact <- data$VIX.Index * data$PE_RATIO

# 添加目标变量和辅助信息
data_selected$PX_LAST <- data$PX_LAST
data_selected$Dates <- as.Date(data$Dates)
data_selected$Company <- as.character(data$Company)  # 设为字符，避免因子干扰

# 保留测试目标公司
target_companies <- c("ORCL", "QCOM", "TXN")

# 拆分训练和测试集
train_data <- subset(data_selected, Dates >= as.Date('2020-01-01') & Dates < as.Date('2024-01-01') & !(Company %in% target_companies))
test_data  <- subset(data_selected, Dates >= as.Date('2024-01-01') & Dates < as.Date('2025-01-01') & Company %in% target_companies)

# 保留 Company 字段用于后续可视化
test_company <- test_data$Company

# 建模前删除 Company 列（避免因子冲突）
train_data$Company <- NULL
test_data$Company <- NULL

# ==== OLS 模型 ====
model_ols <- glm(PX_LAST ~ ., data = train_data)
ols_pred <- predict(model_ols, newdata = test_data)

# ==== LASSO 模型 ====
x_train <- sparse.model.matrix(PX_LAST ~ ., data = train_data)[, -1]
y_train <- train_data$PX_LAST
x_test  <- sparse.model.matrix(PX_LAST ~ ., data = test_data)[, -1]

set.seed(123)
model_lasso <- cv.gamlr(x_train, y_train, nfold = 10)
lasso_pred <- predict(model_lasso, x_test)

# ==== 模型评估 ====
y_test <- test_data$PX_LAST
rmse_ols <- sqrt(mean((y_test - ols_pred)^2))
r2_ols <- 1 - sum((y_test - ols_pred)^2) / sum((y_test - mean(y_test))^2)

rmse_lasso <- sqrt(mean((y_test - lasso_pred)^2))
r2_lasso <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)

cat("OLS RMSE:", round(rmse_ols, 4), "R²:", round(r2_ols, 4), "\n")
cat("LASSO RMSE:", round(rmse_lasso, 4), "R²:", round(r2_lasso, 4), "\n")

# ==== 组合绘图数据 ====
test_data$Company <- test_company
test_data$OLS_PRED <- ols_pred
test_data$LASSO_PRED <- as.numeric(lasso_pred)

# ==== 可视化 ====
p <- ggplot(test_data, aes(x = Dates)) +
  geom_line(aes(y = PX_LAST, color = "Actual Value"), size = 1) +
  geom_line(aes(y = OLS_PRED, color = "OLS Prediction"), linetype = "dashed") +
  geom_line(aes(y = LASSO_PRED, color = "LASSO Prediction"), linetype = "dotted") +
  facet_wrap(~ Company, scales = "free_y",ncol=1) +
  labs(title = "OLS vs LASSO Prediction for ORCL, QCOM, TXN (2024)",
       y = "PX_LAST", x = "Date", color = "Legend") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(filename = "OLS_LASSO_Comparison_TargetCompanies.png", plot = p,
       width = 12, height = 8, dpi = 300, bg = "white")

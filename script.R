# -----------------------------------------------
# parkinsons_modeling_analysis.R
# Final Project: Predicting motor_UPDRS via Voice Biomarkers
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(ggplot2)

# Load and Clean Data
data <- read.csv("/Users/user/ADS-503-Group-Project/parkinsons_telemonitoring.csv/parkinsons_updrs.data")
data$subject <- as.factor(data$subject.)
data$sex <- as.factor(data$sex)

# Drop irrelevant columns
model_data <- data %>% select(-subject., -test_time, -Jitter.DDP)

# Split Data
set.seed(42)
trainIndex <- createDataPartition(model_data$motor_UPDRS, p = 0.8, list = FALSE)
trainData <- model_data[trainIndex, ]
testData  <- model_data[-trainIndex, ]
y_test    <- testData$motor_UPDRS

# Prepare data for LASSO
x_train <- model.matrix(motor_UPDRS ~ ., trainData)[,-1]
y_train <- trainData$motor_UPDRS
x_test  <- model.matrix(motor_UPDRS ~ ., testData)[,-1]

# LASSO Model
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
best_lambda <- lasso_model$lambda.min
pred_lasso  <- predict(lasso_model, s = best_lambda, newx = x_test)
lasso_rmse  <- sqrt(mean((pred_lasso - y_test)^2))

# LASSO with caret
train_control <- trainControl(method = "cv", number = 10)
lasso_caret <- train(
  motor_UPDRS ~ .,
  data = trainData,
  method = "glmnet",
  trControl = train_control,
  tuneLength = 10
)
pred_lasso_caret <- predict(lasso_caret, newdata = testData)
lasso_caret_rmse <- RMSE(pred_lasso_caret, y_test)

# Ranger RF Model
set.seed(42)
rf_ranger <- train(
  motor_UPDRS ~ .,
  data = trainData,
  method = "ranger",
  trControl = train_control,
  tuneLength = 3,
  importance = "impurity"
)
pred_rf_ranger <- predict(rf_ranger, newdata = testData)
rf_ranger_rmse <- RMSE(pred_rf_ranger, y_test)

# Save Results
results_df <- tibble(
  Model = c("LASSO", "Ranger RF"),
  RMSE  = c(round(lasso_caret_rmse, 4), round(rf_ranger_rmse, 4))
)

write.csv(results_df, "model_comparison_results.csv", row.names = FALSE)
saveRDS(lasso_caret, "lasso_model.rds")
saveRDS(rf_ranger, "rf_model.rds")
saveRDS(testData, "test_data.rds")

cat("✅ All tasks complete. Models and results exported.\n")
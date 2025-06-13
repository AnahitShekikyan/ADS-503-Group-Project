# -----------------------------------------------
# Final Project: Predicting motor_UPDRS via Voice Biomarkers
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(e1071)     # SVM
library(gbm)       # Gradient Boosting
library(kknn)      # kNN

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

# Prepare training control
train_control <- trainControl(method = "cv", number = 10)

# -------------------------
# Train Models
# -------------------------

# 1. LASSO
lasso_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "glmnet",
  trControl = train_control,
  tuneLength = 10
)
pred_lasso <- predict(lasso_model, newdata = testData)
rmse_lasso <- RMSE(pred_lasso, y_test)

# 2. Random Forest (Ranger)
rf_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "ranger",
  trControl = train_control,
  tuneLength = 3,
  importance = "impurity"
)
pred_rf <- predict(rf_model, newdata = testData)
rmse_rf <- RMSE(pred_rf, y_test)

# 3. SVM
svm_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "svmRadial",
  trControl = train_control
)
pred_svm <- predict(svm_model, newdata = testData)
rmse_svm <- RMSE(pred_svm, y_test)

# 4. Gradient Boosting
gbm_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "gbm",
  trControl = train_control,
  verbose = FALSE
)
pred_gbm <- predict(gbm_model, newdata = testData)
rmse_gbm <- RMSE(pred_gbm, y_test)

# 5. kNN
knn_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "kknn",
  trControl = train_control
)
pred_knn <- predict(knn_model, newdata = testData)
rmse_knn <- RMSE(pred_knn, y_test)

# -------------------------
# Save & Export Results
# -------------------------

results_df <- tibble(
  Model = c("LASSO", "Ranger RF", "SVM", "GBM", "kNN"),
  RMSE  = c(round(rmse_lasso, 4), round(rmse_rf, 4), round(rmse_svm, 4),
            round(rmse_gbm, 4), round(rmse_knn, 4))
)

# Save results
write.csv(results_df, "model_comparison_results.csv", row.names = FALSE)
saveRDS(lasso_model, "lasso_model.rds")
saveRDS(rf_model, "rf_model.rds")
saveRDS(svm_model, "svm_model.rds")
saveRDS(gbm_model, "gbm_model.rds")
saveRDS(knn_model, "knn_model.rds")
saveRDS(testData, "test_data.rds")

cat("✅ All tasks complete. Models trained, evaluated, and saved.\n")

# -----------------------------------------------
# Final Project: Predicting motor_UPDRS via Voice Biomarkers
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(glmnet)
  library(ranger)
  library(e1071)
  library(gbm)
  library(kknn)
  library(ggplot2)
  library(parallel)
  library(doParallel)
  if (!require(gridExtra)) install.packages("gridExtra", dependencies = TRUE)
  library(gridExtra)
})

# Enable parallel processing
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

log_message <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
}

# -------------------------
# Data Loading & Cleaning
# -------------------------

log_message("Loading and preprocessing data...")
data_path <- "/Users/user/ADS-503-Group-Project/parkinsons_telemonitoring.csv/parkinsons_updrs.data"
stopifnot(file.exists(data_path))

data <- read.csv(data_path)
data$subject <- as.factor(data$subject.)
data$sex <- as.factor(data$sex)
model_data <- data %>% select(-subject., -test_time, -Jitter.DDP)
model_data <- na.omit(model_data)

# -------------------------
# Train/Test Split
# -------------------------

log_message("Splitting data into training and test sets...")
set.seed(42)
trainIndex <- createDataPartition(model_data$motor_UPDRS, p = 0.8, list = FALSE)
trainData <- model_data[trainIndex, ]
testData  <- model_data[-trainIndex, ]
y_test    <- testData$motor_UPDRS

train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# -------------------------
# Train Models + Track Runtime
# -------------------------

results <- list()
predictions <- list()
run_times <- list()

# LASSO
start_time <- Sys.time()
log_message("Training LASSO...")
lasso_model <- train(motor_UPDRS ~ ., data = trainData, method = "glmnet", trControl = train_control, tuneLength = 10)
run_times[["LASSO"]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pred_lasso <- predict(lasso_model, newdata = testData)
rmse_lasso <- RMSE(pred_lasso, y_test)
results[["LASSO"]] <- list(model = lasso_model, rmse = rmse_lasso, preds = pred_lasso)
predictions[["LASSO"]] <- pred_lasso

# Random Forest
start_time <- Sys.time()
log_message("Training Ranger RF...")
rf_model <- train(motor_UPDRS ~ ., data = trainData, method = "ranger",
                  trControl = train_control,
                  tuneGrid = expand.grid(mtry = c(2, 5, 10), splitrule = "variance", min.node.size = 5),
                  num.trees = 100)
run_times[["Ranger RF"]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pred_rf <- predict(rf_model, newdata = testData)
rmse_rf <- RMSE(pred_rf, y_test)
results[["Ranger RF"]] <- list(model = rf_model, rmse = rmse_rf, preds = pred_rf)
predictions[["Ranger RF"]] <- pred_rf

# SVM
start_time <- Sys.time()
log_message("Training SVM...")
svm_model <- train(motor_UPDRS ~ ., data = trainData, method = "svmRadial", trControl = train_control, tuneLength = 5)
run_times[["SVM"]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pred_svm <- predict(svm_model, newdata = testData)
rmse_svm <- RMSE(pred_svm, y_test)
results[["SVM"]] <- list(model = svm_model, rmse = rmse_svm, preds = pred_svm)
predictions[["SVM"]] <- pred_svm

# GBM
start_time <- Sys.time()
log_message("Training GBM...")
gbm_model <- train(motor_UPDRS ~ ., data = trainData, method = "gbm", trControl = train_control, tuneLength = 5, verbose = FALSE)
run_times[["GBM"]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pred_gbm <- predict(gbm_model, newdata = testData)
rmse_gbm <- RMSE(pred_gbm, y_test)
results[["GBM"]] <- list(model = gbm_model, rmse = rmse_gbm, preds = pred_gbm)
predictions[["GBM"]] <- pred_gbm

# kNN
start_time <- Sys.time()
log_message("Training kNN...")
knn_model <- train(motor_UPDRS ~ ., data = trainData, method = "kknn", trControl = train_control, tuneLength = 5)
run_times[["kNN"]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
pred_knn <- predict(knn_model, newdata = testData)
rmse_knn <- RMSE(pred_knn, y_test)
results[["kNN"]] <- list(model = knn_model, rmse = rmse_knn, preds = pred_knn)
predictions[["kNN"]] <- pred_knn

# -------------------------
# Ensemble Prediction
# -------------------------

valid_preds <- predictions[!sapply(predictions, is.null)]

if (length(valid_preds) >= 2) {
  log_message("Creating ensemble model (average of predictions)...")
  ensemble_preds <- rowMeans(do.call(cbind, valid_preds))
  rmse_ensemble <- RMSE(ensemble_preds, y_test)
  results[["Ensemble"]] <- list(model = "Average", rmse = rmse_ensemble, preds = ensemble_preds)
  predictions[["Ensemble"]] <- ensemble_preds
  log_message(paste0("Ensemble RMSE: ", round(rmse_ensemble, 4)))
}

# -------------------------
# Save Results, Predictions, Runtime
# -------------------------

results_df <- tibble(
  Model = names(results),
  RMSE  = sapply(results, function(x) x$rmse)
)
write.csv(results_df, "model_comparison_results.csv", row.names = FALSE)

runtime_df <- tibble(
  Model = names(run_times),
  Runtime_Seconds = unlist(run_times)
)
write.csv(runtime_df, "model_runtimes.csv", row.names = FALSE)

saveRDS(testData, "test_data.rds")
all_preds <- data.frame(ID = seq_along(y_test), Actual = y_test)
for (name in names(predictions)) {
  all_preds[[name]] <- predictions[[name]]
}
write.csv(all_preds, "test_predictions.csv", row.names = FALSE)

for (model_name in names(results)) {
  model_object <- results[[model_name]]$model
  if (inherits(model_object, "train")) {
    saveRDS(model_object, paste0(tolower(gsub(" ", "_", model_name)), "_model.rds"))
  }
}

stopCluster(cl)
log_message("All tasks complete. Models trained, evaluated, and saved.")

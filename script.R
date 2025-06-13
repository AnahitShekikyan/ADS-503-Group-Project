# -----------------------------------------------
# Final Project: Predicting motor_UPDRS via Voice Biomarkers
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

# Load Required Libraries
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
# Train Models
# -------------------------

results <- list()
predictions <- list()

# LASSO
log_message("Training LASSO...")
lasso_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "glmnet",
  trControl = train_control,
  tuneLength = 10
)
pred_lasso <- predict(lasso_model, newdata = testData)
rmse_lasso <- RMSE(pred_lasso, y_test)
results[["LASSO"]] <- list(model = lasso_model, rmse = rmse_lasso, preds = pred_lasso)
predictions[["LASSO"]] <- pred_lasso

# Random Forest
log_message("Training Ranger RF...")
rf_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "ranger",
  trControl = train_control,
  tuneGrid = expand.grid(
    mtry = c(2, 5, 10),
    splitrule = "variance",
    min.node.size = 5
  ),
  num.trees = 100
)
pred_rf <- predict(rf_model, newdata = testData)
rmse_rf <- RMSE(pred_rf, y_test)
results[["Ranger RF"]] <- list(model = rf_model, rmse = rmse_rf, preds = pred_rf)
predictions[["Ranger RF"]] <- pred_rf

# SVM
log_message("Training SVM...")
svm_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "svmRadial",
  trControl = train_control,
  tuneLength = 5
)
pred_svm <- predict(svm_model, newdata = testData)
rmse_svm <- RMSE(pred_svm, y_test)
results[["SVM"]] <- list(model = svm_model, rmse = rmse_svm, preds = pred_svm)
predictions[["SVM"]] <- pred_svm

# GBM
log_message("Training GBM...")
gbm_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "gbm",
  trControl = train_control,
  tuneLength = 5,
  verbose = FALSE
)
pred_gbm <- predict(gbm_model, newdata = testData)
rmse_gbm <- RMSE(pred_gbm, y_test)
results[["GBM"]] <- list(model = gbm_model, rmse = rmse_gbm, preds = pred_gbm)
predictions[["GBM"]] <- pred_gbm

# kNN
log_message("Training kNN...")
knn_model <- train(
  motor_UPDRS ~ ., data = trainData,
  method = "kknn",
  trControl = train_control,
  tuneLength = 5
)
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
} else {
  log_message("Not enough valid predictions for ensemble. Skipping ensemble model.")
}

# -------------------------
# Diagnostic & Residual Plots
# -------------------------

log_message("Generating diagnostic and residual plots...")

plot_pred_vs_actual <- function(preds, name) {
  ggplot(data.frame(Actual = y_test, Predicted = preds), aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(color = "red", linetype = "dashed") +
    ggtitle(paste("Predicted vs Actual -", name)) +
    theme_minimal()
}

plot_residuals <- function(preds, name) {
  residuals <- y_test - preds
  ggplot(data.frame(Residuals = residuals, Predicted = preds), aes(x = Predicted, y = Residuals)) +
    geom_point(alpha = 0.5) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    ggtitle(paste("Residual Plot -", name)) +
    theme_minimal()
}

pdf("model_diagnostics.pdf", width = 8, height = 12)
for (name in names(predictions)) {
  p1 <- plot_pred_vs_actual(predictions[[name]], name)
  p2 <- plot_residuals(predictions[[name]], name)
  grid.arrange(p1, p2, ncol = 1)
}
dev.off()

# -------------------------
# Feature Importance Plots
# -------------------------

log_message("Extracting and plotting feature importance...")

importance_plots <- list()
for (model_name in c("Ranger RF", "GBM")) {
  model_obj <- results[[model_name]]$model
  if ("varImp" %in% methods(class = class(model_obj))) {
    vi <- varImp(model_obj)
    df <- as.data.frame(vi$importance)
    df$Feature <- rownames(df)
    df <- df %>% arrange(desc(Overall)) %>% slice(1:10)
    
    p <- ggplot(df, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_bar(stat = "identity") +
      coord_flip() +
      ggtitle(paste("Top 10 Important Features -", model_name)) +
      theme_minimal()
    
    importance_plots[[model_name]] <- p
  }
}

if (length(importance_plots) > 0) {
  pdf("feature_importance_plots.pdf", width = 8, height = 6)
  for (p in importance_plots) print(p)
  dev.off()
}

# -------------------------
# Save Results & Predictions
# -------------------------

log_message("Saving results, predictions, and model objects...")

results_df <- tibble(
  Model = names(results),
  RMSE  = sapply(results, function(x) x$rmse)
)
write.csv(results_df, "model_comparison_results.csv", row.names = FALSE)
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

# -------------------------
# Shutdown Parallel
# -------------------------

stopCluster(cl)
log_message("All tasks complete. Models trained, evaluated, and saved.")

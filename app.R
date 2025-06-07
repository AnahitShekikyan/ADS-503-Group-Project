# -----------------------------------------------
# app.R - Parkinson's Dashboard Shiny App
# Final Project: Predicting motor_UPDRS via Voice Biomarkers
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

library(shiny)
library(ggplot2)
library(readr)
library(dplyr)
library(DT)

# Load models and data
lasso_model <- readRDS("/Users/user/lasso_model.rds")
rf_model    <- readRDS("/Users/user/rf_model.rds")
results_df  <- read_csv("/Users/user/model_comparison_results.csv")
testData    <- readRDS("/Users/user/test_data.rds")

# Prepare LASSO input
x_test <- model.matrix(motor_UPDRS ~ ., testData)[, -1]
y_test <- testData$motor_UPDRS

# Predictions
pred_lasso <- predict(lasso_model, newdata = testData)
pred_rf    <- predict(rf_model, newdata = testData)

# UI
ui <- fluidPage(
  titlePanel("Parkinson's UPDRS Prediction Dashboard"),
  tabsetPanel(
    tabPanel("Model Comparison",
             fluidRow(
               column(6,
                      plotOutput("lassoPlot"),
                      verbatimTextOutput("lassoRMSE")
               ),
               column(6,
                      plotOutput("rfPlot"),
                      verbatimTextOutput("rfRMSE")
               )
             )
    ),
    tabPanel("Results Table",
             DTOutput("resultsTable")
    )
  )
)

# Server
server <- function(input, output) {
  
  output$lassoPlot <- renderPlot({
    df <- data.frame(True = y_test, Predicted = as.numeric(pred_lasso))
    ggplot(df, aes(x = True, y = Predicted)) +
      geom_point(color = "darkblue", alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "LASSO: Predicted vs Actual", x = "True motor_UPDRS", y = "Predicted") +
      theme_minimal()
  })
  
  output$rfPlot <- renderPlot({
    df <- data.frame(True = y_test, Predicted = as.numeric(pred_rf))
    ggplot(df, aes(x = True, y = Predicted)) +
      geom_point(color = "darkgreen", alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "Ranger RF: Predicted vs Actual", x = "True motor_UPDRS", y = "Predicted") +
      theme_minimal()
  })
  
  output$lassoRMSE <- renderPrint({
    paste("LASSO RMSE:", results_df %>% filter(Model == "LASSO") %>% pull(RMSE))
  })
  
  output$rfRMSE <- renderPrint({
    paste("Ranger RF RMSE:", results_df %>% filter(Model == "Ranger RF") %>% pull(RMSE))
  })
  
  output$resultsTable <- renderDT({
    datatable(results_df, options = list(dom = 't', paging = FALSE), rownames = FALSE)
  })
}

# Run
shinyApp(ui = ui, server = server)

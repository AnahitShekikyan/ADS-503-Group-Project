# -----------------------------------------------
# University of San Diego - Final Project
# Team: Arjun Venkatesh, Vinh Dao, Anahit Shekikyan
# -----------------------------------------------

library(shiny)
library(shinydashboard)
library(bslib)
library(ggplot2)
library(readr)
library(dplyr)
library(DT)
library(shinyWidgets)

# Load models and data
lasso_model <- readRDS("/Users/user/lasso_model.rds")
rf_model    <- readRDS("/Users/user/rf_model.rds")
svm_model   <- readRDS("/Users/user/svm_model.rds")
gbm_model   <- readRDS("/Users/user/gbm_model.rds")
knn_model   <- readRDS("/Users/user/knn_model.rds")

results_df  <- read_csv("/Users/user/model_comparison_results.csv")
testData    <- readRDS("/Users/user/test_data.rds")

# Prepare prediction outputs
y_test <- testData$motor_UPDRS
preds <- list(
  LASSO = predict(lasso_model, newdata = testData),
  RF    = predict(rf_model,    newdata = testData),
  SVM   = predict(svm_model,   newdata = testData),
  GBM   = predict(gbm_model,   newdata = testData),
  kNN   = predict(knn_model,   newdata = testData)
)

# Custom theme
usd_colors <- bs_theme(
  bootswatch = "flatly",
  primary = "#002855",    # Navy Blue
  secondary = "#007FAE",  # Torero Blue
  base_font = font_google("Roboto")
)

# UI
ui <- navbarPage(
  title = div(
    img(src = "usd-logo-primary-thumb.png", height = "40px", style = "margin-right:10px;"),
    span("Parkinson's UPDRS Prediction Dashboard", style = "color:#002855; font-weight:bold;")
  ),
  theme = usd_colors,
  
  # Add custom CSS to override navbar background
  header = tags$style(HTML("
    .navbar-default {
      background-color: white !important;
      border-color: #e7e7e7 !important;
    }
    .navbar-default .navbar-nav > li > a,
    .navbar-default .navbar-brand {
      color: #002855 !important;
    }
    .navbar-default .navbar-nav > li > a:hover,
    .navbar-default .navbar-brand:hover {
      color: #005a9c !important;
    }
  ")),
  
  tabPanel("Overview",
           fluidPage(
             h2("Welcome to the Parkinson's Prediction Dashboard"),
             p("This interactive application allows you to explore various machine learning models used to predict motor_UPDRS scores based on vocal biomarkers."),
             br(),
             fluidRow(
               column(6, valueBoxOutput("bestModelBox")),
               column(6, valueBoxOutput("lowestRMSEBox"))
             ),
             br(),
             plotOutput("overviewPlot")
           )
  ),
  
  tabPanel("Model Predictions",
           sidebarLayout(
             sidebarPanel(
               selectInput("model", "Select Model:", choices = names(preds)),
               sliderInput("ageFilter", "Filter by Age:", 
                           min = min(testData$age), max = max(testData$age),
                           value = c(min(testData$age), max(testData$age)))
             ),
             mainPanel(
               h4(textOutput("modelTitle")),
               plotOutput("predPlot"),
               verbatimTextOutput("rmseText")
             )
           )
  ),
  
  tabPanel("RMSE Table",
           fluidPage(
             h4("Model Performance Summary"),
             DTOutput("resultsTable")
           )
  ),
  
  tabPanel("Feature Explorer",
           fluidPage(
             selectInput("feature", "Choose Feature:", choices = colnames(testData)[sapply(testData, is.numeric)]),
             plotOutput("featurePlot")
           )
  )
)

# Server
server <- function(input, output) {
  
  output$bestModelBox <- renderValueBox({
    best <- results_df[which.min(results_df$RMSE), ]
    valueBox(best$Model, subtitle = "Best Performing Model", color = "navy")
  })
  
  output$lowestRMSEBox <- renderValueBox({
    rmse <- min(results_df$RMSE)
    valueBox(round(rmse, 3), subtitle = "Lowest RMSE", color = "blue")
  })
  
  output$overviewPlot <- renderPlot({
    ggplot(results_df, aes(x = Model, y = RMSE, fill = Model)) +
      geom_col() +
      labs(title = "Model RMSE Comparison", y = "RMSE") +
      theme_minimal()
  })
  
  output$modelTitle <- renderText({
    paste(input$model, " - Predicted vs Actual")
  })
  
  output$predPlot <- renderPlot({
    df <- data.frame(True = y_test, Predicted = as.numeric(preds[[input$model]]), Age = testData$age)
    df <- df %>% filter(Age >= input$ageFilter[1], Age <= input$ageFilter[2])
    
    if (nrow(df) == 0) {
      showNotification("No data in this age range", type = "warning")
      return(NULL)
    }
    
    ggplot(df, aes(x = True, y = Predicted)) +
      geom_point(color = "#007FAE", alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#002855") +
      labs(x = "True motor_UPDRS", y = "Predicted", title = paste(input$model, "Predicted vs Actual")) +
      theme_minimal()
  })
  
  output$rmseText <- renderPrint({
    rmse <- results_df %>% filter(Model == input$model | Model == paste(input$model, "(caret)")) %>% pull(RMSE)
    paste("RMSE:", round(rmse, 4))
  })
  
  output$resultsTable <- renderDT({
    datatable(results_df, options = list(pageLength = 5), rownames = FALSE)
  })
  
  output$featurePlot <- renderPlot({
    feature <- input$feature
    ggplot(testData, aes_string(x = feature, y = "motor_UPDRS")) +
      geom_point(color = "#002855", alpha = 0.6) +
      geom_smooth(method = "lm", color = "#007FAE") +
      labs(title = paste("motor_UPDRS vs", feature), y = "motor_UPDRS") +
      theme_minimal()
  })
}

# Run app
shinyApp(ui = ui, server = server)

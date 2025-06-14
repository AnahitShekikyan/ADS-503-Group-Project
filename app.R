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
library(rmarkdown)

# Load trained models and data
lasso_model <- readRDS("/Users/user/lasso_model.rds")
rf_model    <- readRDS("/Users/user/ranger_rf_model.rds")
svm_model   <- readRDS("/Users/user/svm_model.rds")
gbm_model   <- readRDS("/Users/user/gbm_model.rds")
knn_model   <- readRDS("/Users/user/knn_model.rds")

results_df  <- read_csv("/Users/user/model_comparison_results.csv")
runtime_data <- read_csv("/Users/user/model_runtimes.csv")
testData    <- readRDS("/Users/user/test_data.rds")
y_test      <- testData$motor_UPDRS

# Model predictions
preds <- list(
  LASSO       = predict(lasso_model, newdata = testData),
  `Ranger RF` = predict(rf_model,    newdata = testData),
  SVM         = predict(svm_model,   newdata = testData),
  GBM         = predict(gbm_model,   newdata = testData),
  kNN         = predict(knn_model,   newdata = testData)
)

# Custom theme
usd_colors <- bs_theme(
  bootswatch = "flatly",
  primary = "#002855",
  secondary = "#007FAE",
  base_font = font_google("Roboto")
)

# UI
ui <- navbarPage(
  title = div(
    img(src = "usd-logo-primary-thumb.png", height = "40px", 
        style = "margin-right:10px; background:white; border-radius:4px; padding:4px;"),
    span("Parkinson's UPDRS Prediction Dashboard", 
         style = "color:#002855; font-weight:bold; font-size: 22px;")
  ),
  theme = usd_colors,
  
  tabPanel("Overview",
           fluidPage(
             br(),
             h2("Welcome", style = "color:#002855; font-weight:bold"),
             p("Explore machine learning models that predict Parkinson's motor symptoms using vocal biomarkers."),
             br(),
             fluidRow(
               column(6, valueBoxOutput("bestModelBox")),
               column(6, valueBoxOutput("lowestRMSEBox"))
             ),
             br(),
             selectInput("overviewFilter", "Filter by Model:", choices = c("All", names(preds)), selected = "All"),
             plotOutput("overviewPlot", height = "300px"),
             br(),
             h5("Download full analysis report:"),
             downloadButton("downloadReport", "Download HTML Report")
           )
  ),
  
  tabPanel("Model Predictions",
           fluidPage(
             br(),
             sidebarLayout(
               sidebarPanel(
                 selectInput("model", "Select Model:", choices = names(preds)),
                 sliderInput("ageFilter", "Filter by Age:",
                             min = min(testData$age), max = max(testData$age),
                             value = c(min(testData$age), max(testData$age))
                 ),
                 sliderInput("predRange", "Filter by Predicted Value:",
                             min = floor(min(unlist(preds))), max = ceiling(max(unlist(preds))),
                             value = c(floor(min(unlist(preds))), ceiling(max(unlist(preds))))
                 ),
                 sliderInput("pointAlpha", "Point Transparency:", min = 0.1, max = 1, step = 0.1, value = 0.6),
                 checkboxInput("showResiduals", "Show Residual Plot", value = FALSE),
                 downloadButton("downloadModelData", "Download Predictions CSV")
               ),
               mainPanel(
                 h4(textOutput("modelTitle")),
                 plotOutput("predPlot", height = "400px"),
                 verbatimTextOutput("rmseText")
               )
             )
           )
  ),
  
  tabPanel("RMSE Table",
           fluidPage(
             br(),
             h4("Model Performance Summary"),
             DTOutput("resultsTable")
           )
  ),
  
  tabPanel("Feature Explorer",
           fluidPage(
             br(),
             fluidRow(
               column(4,
                      selectInput("feature", "Choose Feature:", 
                                  choices = colnames(testData)[sapply(testData, is.numeric)]),
                      selectInput("smoother", "Smoothing Method:", 
                                  choices = c("Linear (lm)" = "lm", "LOESS" = "loess", "None" = "none"), selected = "lm"),
                      sliderInput("featureAlpha", "Point Transparency:", min = 0.1, max = 1, step = 0.1, value = 0.6),
                      checkboxInput("jitterPoints", "Apply Jitter", value = FALSE)
               ),
               column(8,
                      plotOutput("featurePlot", height = "400px")
               )
             )
           )
  ),
  
  tabPanel("Run Time",
           fluidPage(
             br(),
             h4("Model Training Duration"),
             DTOutput("runtimeTable"),
             br(),
             plotOutput("runtimePlot", height = "300px")
           )
  ),
  
  tabPanel("Project Details",
           fluidPage(
             br(),
             h3("Workflow and Technical Overview", style = "color:#002855; font-weight:bold"),
             p("This project predicts Parkinson's motor symptom severity (motor_UPDRS) using vocal biomarkers. The pipeline includes:"),
             tags$ul(
               tags$li("Data Cleaning: Removed subject IDs, timestamps, and irrelevant jitter features"),
               tags$li("Feature Selection: Included all relevant numerical voice metrics"),
               tags$li("Model Training: LASSO, Ranger RF, SVM, GBM, and kNN using 10-fold CV via caret"),
               tags$li("Model Evaluation: RMSE calculated on 20% test holdout set"),
               tags$li("Visual Analysis: Predicted vs actual, residuals, feature trends"),
               tags$li("Deployment: Fully interactive dashboard built in R Shiny")
             ),
             p("Technologies: caret, ranger, glmnet, gbm, e1071, kknn, bslib, DT, shinyWidgets, rmarkdown.")
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
    df <- if (input$overviewFilter == "All") results_df else filter(results_df, Model == input$overviewFilter)
    ggplot(df, aes(x = Model, y = RMSE, fill = Model)) +
      geom_col() +
      theme_minimal() +
      labs(title = "Model RMSE Comparison", y = "RMSE") +
      theme(legend.position = "none")
  })
  
  output$modelTitle <- renderText({
    paste(input$model, "- Predicted vs Actual / Residual")
  })
  
  output$predPlot <- renderPlot({
    df <- data.frame(True = y_test, Predicted = as.numeric(preds[[input$model]]), Age = testData$age)
    df <- df %>%
      filter(Age >= input$ageFilter[1], Age <= input$ageFilter[2]) %>%
      filter(Predicted >= input$predRange[1], Predicted <= input$predRange[2])
    
    if (nrow(df) == 0) {
      showNotification("No data in this filter range", type = "warning")
      return(NULL)
    }
    
    if (input$showResiduals) {
      df$Residual <- df$True - df$Predicted
      ggplot(df, aes(x = Predicted, y = Residual)) +
        geom_point(alpha = input$pointAlpha, color = "#D7263D") +
        geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
        labs(title = paste("Residuals for", input$model), y = "Residual (True - Predicted)", x = "Predicted") +
        theme_minimal()
    } else {
      ggplot(df, aes(x = True, y = Predicted)) +
        geom_point(alpha = input$pointAlpha, color = "#007FAE") +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#002855") +
        labs(x = "True motor_UPDRS", y = "Predicted", title = paste(input$model, "Predicted vs Actual")) +
        theme_minimal()
    }
  })
  
  output$rmseText <- renderPrint({
    rmse <- results_df %>% filter(Model == input$model) %>% pull(RMSE)
    if (length(rmse) == 0) return("RMSE: Not available")
    paste("RMSE:", round(rmse, 4))
  })
  
  output$resultsTable <- renderDT({
    datatable(results_df, options = list(pageLength = 5), rownames = FALSE)
  })
  
  output$featurePlot <- renderPlot({
    p <- ggplot(testData, aes_string(x = input$feature, y = "motor_UPDRS"))
    
    if (input$jitterPoints) {
      p <- p + geom_jitter(alpha = input$featureAlpha, color = "#002855", width = 0.3, height = 0.3)
    } else {
      p <- p + geom_point(alpha = input$featureAlpha, color = "#002855")
    }
    
    if (input$smoother != "none") {
      p <- p + geom_smooth(method = input$smoother, color = "#007FAE", se = TRUE)
    }
    
    p + theme_minimal() +
      labs(title = paste("motor_UPDRS vs", input$feature), x = input$feature, y = "motor_UPDRS")
  })
  
  output$runtimeTable <- renderDT({
    datatable(runtime_data, options = list(pageLength = 5), rownames = FALSE)
  })
  
  output$runtimePlot <- renderPlot({
    ggplot(runtime_data, aes(x = reorder(Model, Runtime_Seconds), y = Runtime_Seconds, fill = Model)) +
      geom_bar(stat = "identity") +
      labs(title = "Training Time by Model", y = "Runtime (seconds)", x = "Model") +
      theme_minimal() +
      theme(legend.position = "none")
  })
  
  output$downloadModelData <- downloadHandler(
    filename = function() {
      paste0("predictions_", gsub(" ", "_", tolower(input$model)), ".csv")
    },
    content = function(file) {
      df <- data.frame(True = y_test, Predicted = preds[[input$model]])
      write.csv(df, file, row.names = FALSE)
    }
  )
  
  output$downloadReport <- downloadHandler(
    filename = function() {
      "Parkinsons_Model_Report.html"
    },
    content = function(file) {
      rmarkdown::render("parkinsons_report.Rmd",
                        params = list(
                          results_df = results_df,
                          test_data = testData,
                          predictions = preds
                        ),
                        output_file = file,
                        envir = new.env(parent = globalenv())
      )
    }
  )
}

# Run app
shinyApp(ui = ui, server = server)

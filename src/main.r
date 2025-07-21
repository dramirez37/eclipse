# ------------------------------------------------------------------
# Script: Eclipse Software Bugs with Walk-Forward Optimization
# Author: David Ramirez
# Date: July 21, 2025
# Description: This script uses a multi-class XGBoost model with a
#              full suite of advanced, dynamically generated features.
# ------------------------------------------------------------------


# 1. SETUP & LOAD LIBRARIES
# ==================================================================
rm(list = ls())
gc(full = TRUE)
set.seed(42)

packages <- c("dplyr", "readr", "data.table", "xgboost", "lubridate", "pROC", "caret", "text2vec", "tm")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(dplyr)
library(readr)
library(data.table)
library(xgboost)
library(lubridate)
library(pROC)
library(caret)
library(text2vec)
library(tm)


# 2. ROBUST DATA LOADING
# ==================================================================
data_path <- "/workspaces/metrics2/data/"

suppressWarnings({
    assigned_to <- read_csv(paste0(data_path, "assigned_to.csv"), show_col_types = FALSE)
    bug_status <- read_csv(paste0(data_path, "bug_status.csv"), show_col_types = FALSE)
    component <- read_csv(paste0(data_path, "component.csv"), show_col_types = FALSE)
    op_sys <- read_csv(paste0(data_path, "op_sys.csv"), show_col_types = FALSE)
    priority <- read_csv(paste0(data_path, "priority.csv"), show_col_types = FALSE)
    resolution <- read_csv(paste0(data_path, "resolution.csv"), show_col_types = FALSE)
    severity <- read_csv(paste0(data_path, "severity.csv"), show_col_types = FALSE)
    cc <- read_csv(paste0(data_path, "cc.csv"), show_col_types = FALSE)
    short_desc <- read_csv(paste0(data_path, "short_desc.csv"), show_col_types = FALSE)
    version <- read_csv(paste0(data_path, "version.csv"), show_col_types = FALSE)
    product <- read_csv(paste0(data_path, "product.csv"), show_col_types = FALSE)
    reports <- read_csv(paste0(data_path, "reports.csv"), show_col_types = FALSE)
})


# 3. FEATURE ENGINEERING & PREPARATION
# ==================================================================
final_resolution <- resolution %>% arrange(.data$id, .data$when) %>% group_by(.data$id) %>% summarise(final_resolution = tolower(last(what)), .groups = "drop")
get_initial_attribute <- function(df, col_name) {df %>% arrange(.data$id, .data$when) %>% group_by(.data$id) %>% summarise(!!paste0(col_name, "_initial") := first(what), .groups = "drop")}
cc_count <- cc %>% group_by(.data$id) %>% summarise(cc_count = n_distinct(what), .groups = "drop")

priority_initial <- get_initial_attribute(priority, "priority")
severity_initial <- get_initial_attribute(severity, "severity")
product_initial <- get_initial_attribute(product, "product")
component_initial <- get_initial_attribute(component, "component")
op_sys_initial <- get_initial_attribute(op_sys, "op_sys")
version_initial <- get_initial_attribute(version, "version")
assigned_to_initial <- get_initial_attribute(assigned_to, "assigned_to")

reassignment_counts <- assigned_to %>% group_by(.data$id) %>% summarise(reassignment_count = n_distinct(what) - 1, .groups = "drop")
reopening_counts <- bug_status %>% filter(what == "REOPENED") %>% group_by(.data$id) %>% summarise(reopening_count = n(), .groups = "drop")

all_events <- bind_rows(select(bug_status, .data$id, .data$when), select(priority, .data$id, .data$when), select(assigned_to, .data$id, .data$when)) %>% distinct()
initial_response_time <- reports %>%
  select(.data$id, .data$opening) %>%
  left_join(all_events, by = "id") %>%
  filter(.data$when > .data$opening) %>%
  group_by(.data$id) %>%
  summarise(first_event_time = min(when), .groups = "drop") %>%
  left_join(select(reports, .data$id, .data$opening), by = "id") %>%
  mutate(initial_response_hr = as.numeric(first_event_time - opening) / 3600) %>%
  select(.data$id, .data$initial_response_hr)


# 4. ASSEMBLE, FILTER, AND PREPARE MASTER DATASET
# ==================================================================
master_data <- reports %>%
  select(.data$id, .data$opening, .data$reporter) %>%
  left_join(final_resolution, by = "id") %>%
  left_join(short_desc %>% arrange(.data$id, .data$when) %>% group_by(.data$id) %>% summarise(short_desc = first(what), .groups="drop"), by="id") %>%
  left_join(priority_initial, by = "id") %>%
  left_join(severity_initial, by = "id") %>%
  left_join(product_initial, by = "id") %>%
  left_join(component_initial, by = "id") %>%
  left_join(op_sys_initial, by = "id") %>%
  left_join(version_initial, by = "id") %>%
  left_join(assigned_to_initial, by = "id") %>%
  left_join(cc_count, by = "id") %>%
  left_join(initial_response_time, by = "id") %>%
  left_join(reassignment_counts, by = "id") %>%
  left_join(reopening_counts, by = "id") %>%
  mutate(cc_count = ifelse(is.na(.data$cc_count), 0, .data$cc_count),
         initial_response_hr = ifelse(is.na(.data$initial_response_hr), -1, .data$initial_response_hr),
         short_desc = ifelse(is.na(.data$short_desc), "", .data$short_desc),
         reassignment_count = ifelse(is.na(.data$reassignment_count), 0, .data$reassignment_count),
         reopening_count = ifelse(is.na(.data$reopening_count), 0, .data$reopening_count))

final_data <- master_data %>%
  filter(!is.na(.data$final_resolution) & .data$final_resolution %in% c("fixed", "invalid", "wontfix", "duplicate", "worksforme", "incomplete")) %>%
  mutate(
    opening_date = as_datetime(.data$opening),
    opening_month = month(.data$opening_date),
    opening_wday = wday(.data$opening_date, label = TRUE),
    assignee_component = paste(assigned_to_initial, component_initial, sep = "_")
  ) %>%
  arrange(.data$opening_date)

final_data$final_resolution <- as.factor(final_data$final_resolution)
class_levels <- levels(final_data$final_resolution)
num_classes <- length(class_levels)
final_data$target_numeric <- as.integer(final_data$final_resolution) - 1


# 5. WALK-FORWARD VALIDATION
# ==================================================================
n_folds <- 3
total_obs <- nrow(final_data)
if (total_obs < (n_folds + 1) * 2) stop("Not enough data to create even one fold after filtering.")
fold_size <- floor(total_obs / (n_folds + 1))

auc_scores <- c()
cat("Starting walk-forward validation with Multi-Class objective...\n")

for (i in 1:n_folds) {
  train_end_idx <- i * fold_size
  validation_end_idx <- train_end_idx + fold_size
  if (validation_end_idx > total_obs || train_end_idx == 0) next

  train_fold <- final_data[1:train_end_idx, ]
  validation_fold <- final_data[(train_end_idx + 1):validation_end_idx, ]

  cat(sprintf("\n--- Fold %d: Training on %d samples, validating on %d samples ---\n", i, nrow(train_fold), nrow(validation_fold)))
  
  # --- Dynamic Feature Engineering for the Fold ---
  # Binning
  top_reporters <- train_fold %>% count(reporter, sort = TRUE) %>% top_n(100) %>% pull(reporter)
  top_assignees <- train_fold %>% count(assigned_to_initial, sort = TRUE) %>% top_n(100) %>% pull(assigned_to_initial)
  train_fold <- train_fold %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees, assigned_to_initial, "OTHER"))
  validation_fold <- validation_fold %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees, assigned_to_initial, "OTHER"))
  
  # Global Reputation
  global_rate <- sum(train_fold$final_resolution == "fixed") / nrow(train_fold)
  reporter_rates <- train_fold %>% group_by(reporter_binned) %>% summarise(reporter_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_rates <- train_fold %>% group_by(assignee_binned) %>% summarise(assignee_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  train_fold <- train_fold %>% left_join(reporter_rates, by="reporter_binned") %>% left_join(assignee_rates, by="assignee_binned")
  validation_fold <- validation_fold %>% left_join(reporter_rates, by="reporter_binned") %>% left_join(assignee_rates, by="assignee_binned")
  
  # ** NEW: Component-Specific Reputation **
  assignee_comp_rates <- train_fold %>% group_by(assignee_component) %>% summarise(assignee_comp_success_rate = sum(final_resolution == "fixed") / n(), .groups="drop")
  train_fold <- train_fold %>% left_join(assignee_comp_rates, by="assignee_component")
  validation_fold <- validation_fold %>% left_join(assignee_comp_rates, by="assignee_component")

  # Impute NAs and create reputation interaction
  train_fold <- train_fold %>% mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate, reporter_success_rate), assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate, assignee_success_rate), assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate, assignee_comp_success_rate), reputation_interaction = reporter_success_rate * assignee_success_rate)
  validation_fold <- validation_fold %>% mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate, reporter_success_rate), assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate, assignee_success_rate), assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate, assignee_comp_success_rate), reputation_interaction = reporter_success_rate * assignee_success_rate)
  
  # --- Dynamic NLP with TF-IDF ---
  it_train <- itoken(train_fold$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  it_validate <- itoken(validation_fold$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  vocab <- create_vocabulary(it_train, stopwords = stopwords("en")) %>% prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.5)
  vectorizer <- vocab_vectorizer(vocab)
  tfidf <- TfIdf$new()
  dtm_train <- create_dtm(it_train, vectorizer) %>% fit_transform(tfidf)
  dtm_validate <- create_dtm(it_validate, vectorizer) %>% transform(tfidf)
  
  # --- Prepare structured data ---
  feature_cols_structured <- setdiff(names(train_fold), c("opening", "opening_date", "final_resolution", "target_numeric", "short_desc", "reporter", "assigned_to_initial"))
  train_structured <- as.data.table(train_fold)[, ..feature_cols_structured]
  validation_structured <- as.data.table(validation_fold)[, ..feature_cols_structured]
  
  for(col in names(train_structured)) {
    if(is.character(train_structured[[col]]) || is.factor(train_structured[[col]])) {
      all_levels <- unique(c(train_structured[[col]], validation_structured[[col]]))
      train_structured[, (col) := as.integer(factor(.SD[[col]], levels=all_levels))]
      validation_structured[, (col) := as.integer(factor(.SD[[col]], levels=all_levels))]
    }
  }

  train_combined_matrix <- cbind(as.matrix(train_structured), dtm_train)
  validation_combined_matrix <- cbind(as.matrix(validation_structured), dtm_validate)

  dtrain <- xgb.DMatrix(data = train_combined_matrix, label = train_fold$target_numeric)
  dvalidation <- xgb.DMatrix(data = validation_combined_matrix, label = validation_fold$target_numeric)

  params <- list(objective = "multi:softprob", eval_metric = "mlogloss", num_class = num_classes, eta = 0.05, max_depth = 6)
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500, watchlist = list(eval = dvalidation), early_stopping_rounds = 30, verbose = 0)

  pred_probs <- predict(xgb_model, dvalidation, reshape = TRUE)
  colnames(pred_probs) <- class_levels
  
  roc_obj <- multiclass.roc(validation_fold$final_resolution, pred_probs, quiet=TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj, levels = c("fixed"), quiet = TRUE))
  auc_scores <- c(auc_scores, auc_val)
  cat(sprintf("Fold %d OvR AUC for 'fixed' class: %.4f\n", i, auc_val))

  is_fixed_response <- ifelse(validation_fold$final_resolution == "fixed", 1, 0)
  fixed_predictor_probs <- pred_probs[, "fixed"]
  plot_roc <- roc(response = is_fixed_response, predictor = fixed_predictor_probs, quiet = TRUE)
  png(paste0("roc_plot_fold_", i, ".png")); plot(plot_roc, main = paste("ROC for 'fixed' vs. Rest - Fold", i, "\nAUC =", round(auc_val, 4)), print.auc=TRUE); dev.off()
}


# 6. FINAL RESULTS & ANALYSIS
# ==================================================================
if (length(auc_scores) > 0) {
  cat("\n--- Walk-Forward Validation Results ---\n")
  cat(sprintf("Average One-vs-Rest AUC for 'fixed' class: %.4f\n", mean(auc_scores)))
  cat(sprintf("Standard Deviation of AUC: %.4f\n\n", sd(auc_scores)))

  cat("--- Training final model on all data... ---\n")
  
  # Create final features on all data
  top_reporters_final <- final_data %>% count(reporter, sort = TRUE) %>% top_n(100) %>% pull(reporter)
  top_assignees_final <- final_data %>% count(assigned_to_initial, sort = TRUE) %>% top_n(100) %>% pull(assigned_to_initial)
  final_data_full <- final_data %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters_final, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees_final, assigned_to_initial, "OTHER"))
  
  reporter_rates_final <- final_data_full %>% group_by(reporter_binned) %>% summarise(reporter_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_rates_final <- final_data_full %>% group_by(assignee_binned) %>% summarise(assignee_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_comp_rates_final <- final_data_full %>% group_by(assignee_component) %>% summarise(assignee_comp_success_rate = sum(final_resolution == "fixed") / n(), .groups="drop")
  global_rate_final <- sum(final_data_full$final_resolution == "fixed") / nrow(final_data_full)
  
  final_data_full <- final_data_full %>%
    left_join(reporter_rates_final, by = "reporter_binned") %>%
    left_join(assignee_rates_final, by = "assignee_binned") %>%
    left_join(assignee_comp_rates_final, by = "assignee_component") %>%
    mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate_final, reporter_success_rate), 
           assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate_final, assignee_success_rate),
           assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate_final, assignee_comp_success_rate),
           reputation_interaction = reporter_success_rate * assignee_success_rate)
  
  # Final NLP
  it_final <- itoken(final_data_full$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  vocab_final <- create_vocabulary(it_final, stopwords = stopwords("en")) %>% prune_vocabulary(term_count_min = 10)
  vectorizer_final <- vocab_vectorizer(vocab_final)
  tfidf_final <- TfIdf$new()
  dtm_final <- create_dtm(it_final, vectorizer_final) %>% fit_transform(tfidf_final)
  
  # Final structured data
  feature_cols_structured_final <- setdiff(names(final_data_full), c("opening", "opening_date", "final_resolution", "target_numeric", "short_desc", "reporter", "assigned_to_initial"))
  final_structured_data <- as.data.table(final_data_full)[, ..feature_cols_structured_final]
  for(col in names(final_structured_data)) { if(is.character(final_structured_data[[col]]) || is.factor(final_structured_data[[col]])) { set(final_structured_data, j = col, value = as.integer(as.factor(final_structured_data[[col]]))) } }
  
  final_combined_matrix <- cbind(as.matrix(final_structured_data), dtm_final)
  dfinal <- xgb.DMatrix(data = final_combined_matrix, label = final_data_full$target_numeric)
  
  final_params <- list(objective="multi:softprob", eval_metric="mlogloss", num_class=num_classes, eta = 0.05, max_depth = 6)
  final_model <- xgb.train(params = final_params, data = dfinal, nrounds = if(!is.null(xgb_model)) xgb_model$best_iteration else 100, verbose = 0)

  importance <- xgb.importance(model = final_model)
  cat("\n--- Top 15 Feature Importance from Final Model ---\n")
  print(importance[1:15,])

  cat("\n--- Confusion Matrix for Final Model (on all data) ---\n")
  final_preds_probs <- predict(final_model, dfinal, reshape = TRUE)
  final_preds_numeric <- max.col(final_preds_probs) - 1
  predicted_labels <- factor(class_levels[final_preds_numeric + 1], levels = class_levels)
  actual_labels <- factor(final_data_full$final_resolution, levels = class_levels)
  conf_matrix <- confusionMatrix(data = predicted_labels, reference = actual_labels)
  print(conf_matrix)

} else {
  cat("\nWalk-forward validation did not produce any results. Please check data quality and fold sizes.\n")
}
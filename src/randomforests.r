# ------------------------------------------------------------------
# Script: Eclipse Software Bugs with Walk-Forward Optimization
# Author: David Ramirez
# Date: July 22, 2025
# Description: This script uses the memory-efficient 'ranger' package
#              for a Random Forest model with a full suite of features.
# ------------------------------------------------------------------


# 1. SETUP & LOAD LIBRARIES
# ==================================================================
rm(list = ls())
gc(full = TRUE)
set.seed(42)

# ** FIX: Replace randomForest with the more memory-efficient ranger package **
packages <- c("dplyr", "readr", "data.table", "ranger", "lubridate", "pROC", "caret", "text2vec", "tm", "igraph", "Matrix")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(dplyr)
library(readr)
library(data.table)
library(ranger)
library(lubridate)
library(pROC)
library(caret)
library(text2vec)
library(tm)
library(igraph)
library(Matrix)


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
bug_activity <- reports %>%
  select(.data$id, .data$opening) %>%
  left_join(all_events, by = "id") %>%
  mutate(time_diff_hr = as.numeric(.data$when - .data$opening) / 3600)

first_response <- bug_activity %>%
  filter(.data$time_diff_hr > 0) %>%
  group_by(.data$id) %>%
  summarise(initial_response_hr = min(time_diff_hr), .groups = "drop")

events_in_first_72h <- bug_activity %>%
  filter(.data$time_diff_hr > 0 & .data$time_diff_hr <= 72) %>%
  group_by(.data$id) %>%
  summarise(events_in_72h = n(), .groups = "drop")


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
  left_join(first_response, by = "id") %>%
  left_join(events_in_first_72h, by = "id") %>%
  left_join(reassignment_counts, by = "id") %>%
  left_join(reopening_counts, by = "id") %>%
  mutate(cc_count = ifelse(is.na(.data$cc_count), 0, .data$cc_count),
         initial_response_hr = ifelse(is.na(.data$initial_response_hr), -1, .data$initial_response_hr),
         events_in_72h = ifelse(is.na(.data$events_in_72h), 0, .data$events_in_72h),
         short_desc = ifelse(is.na(.data$short_desc), "", .data$short_desc),
         reassignment_count = ifelse(is.na(.data$reassignment_count), 0, .data$reassignment_count),
         reopening_count = ifelse(is.na(.data$reopening_count), 0, .data$reopening_count))

final_data <- master_data %>%
  filter(!is.na(.data$final_resolution) & .data$final_resolution %in% c("fixed", "invalid", "wontfix", "duplicate", "worksforme", "incomplete")) %>%
  mutate(
    opening_date = as_datetime(.data$opening),
    opening_month = month(.data$opening_date),
    opening_wday = wday(.data$opening_date, label = TRUE),
    assignee_component = paste(assigned_to_initial, component_initial, sep = "_"),
    has_patch = grepl("patch|\\.diff", short_desc, ignore.case = TRUE),
    has_stack_trace = grepl("exception|stack trace|npe|nullpointer", short_desc, ignore.case = TRUE),
    is_performance = grepl("slow|performance|memory|leak", short_desc, ignore.case = TRUE)
  ) %>%
  arrange(.data$opening_date)

final_data$final_resolution <- as.factor(final_data$final_resolution)
class_levels <- levels(final_data$final_resolution)
num_classes <- length(class_levels)


# 5. WALK-FORWARD VALIDATION
# ==================================================================
n_folds <- 3
total_obs <- nrow(final_data)
if (total_obs < (n_folds + 1) * 2) stop("Not enough data to create even one fold after filtering.")
fold_size <- floor(total_obs / (n_folds + 1))

auc_scores <- c()
cat("Starting walk-forward validation with Random Forest (ranger)...\n")

for (i in 1:n_folds) {
  train_end_idx <- i * fold_size
  validation_end_idx <- train_end_idx + fold_size
  if (validation_end_idx > total_obs || train_end_idx == 0) next

  train_fold <- final_data[1:train_end_idx, ]
  validation_fold <- final_data[(train_end_idx + 1):validation_end_idx, ]

  cat(sprintf("\n--- Fold %d: Training on %d samples, validating on %d samples ---\n", i, nrow(train_fold), nrow(validation_fold)))
  
  # Dynamic Feature Engineering for the Fold
  train_fold <- train_fold %>% mutate(reporter = as.character(reporter), assigned_to_initial = as.character(assigned_to_initial))
  validation_fold <- validation_fold %>% mutate(reporter = as.character(reporter), assigned_to_initial = as.character(assigned_to_initial))
  
  edges <- train_fold %>% select(reporter, assigned_to_initial) %>% filter(!is.na(reporter) & !is.na(assigned_to_initial))
  g <- graph_from_data_frame(edges, directed = FALSE)
  pagerank_scores <- page_rank(g)$vector
  pagerank_df <- data.frame(person = names(pagerank_scores), pagerank = pagerank_scores)
  train_fold <- train_fold %>% left_join(pagerank_df, by = c("reporter" = "person")) %>% rename(reporter_pagerank = pagerank) %>% left_join(pagerank_df, by = c("assigned_to_initial" = "person")) %>% rename(assignee_pagerank = pagerank)
  validation_fold <- validation_fold %>% left_join(pagerank_df, by = c("reporter" = "person")) %>% rename(reporter_pagerank = pagerank) %>% left_join(pagerank_df, by = c("assigned_to_initial" = "person")) %>% rename(assignee_pagerank = pagerank)
  train_fold <- train_fold %>% mutate(reporter_pagerank = ifelse(is.na(reporter_pagerank), 0, reporter_pagerank), assignee_pagerank = ifelse(is.na(assignee_pagerank), 0, assignee_pagerank))
  validation_fold <- validation_fold %>% mutate(reporter_pagerank = ifelse(is.na(reporter_pagerank), 0, reporter_pagerank), assignee_pagerank = ifelse(is.na(assignee_pagerank), 0, assignee_pagerank))
  
  top_reporters <- train_fold %>% count(reporter, sort = TRUE) %>% top_n(100) %>% pull(reporter)
  top_assignees <- train_fold %>% count(assigned_to_initial, sort = TRUE) %>% top_n(100) %>% pull(assigned_to_initial)
  train_fold <- train_fold %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees, assigned_to_initial, "OTHER"))
  validation_fold <- validation_fold %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees, assigned_to_initial, "OTHER"))
  global_rate <- sum(train_fold$final_resolution == "fixed") / nrow(train_fold)
  reporter_rates <- train_fold %>% group_by(reporter_binned) %>% summarise(reporter_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_rates <- train_fold %>% group_by(assignee_binned) %>% summarise(assignee_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_comp_rates <- train_fold %>% group_by(assignee_component) %>% summarise(assignee_comp_success_rate = sum(final_resolution == "fixed") / n(), .groups="drop")
  train_fold <- train_fold %>% left_join(reporter_rates, by="reporter_binned") %>% left_join(assignee_rates, by="assignee_binned") %>% left_join(assignee_comp_rates, by="assignee_component")
  validation_fold <- validation_fold %>% left_join(reporter_rates, by="reporter_binned") %>% left_join(assignee_rates, by="assignee_binned") %>% left_join(assignee_comp_rates, by="assignee_component")
  train_fold <- train_fold %>% mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate, reporter_success_rate), assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate, assignee_success_rate), assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate, assignee_comp_success_rate), reputation_interaction = reporter_success_rate * assignee_success_rate)
  validation_fold <- validation_fold %>% mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate, reporter_success_rate), assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate, assignee_success_rate), assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate, assignee_comp_success_rate), reputation_interaction = reporter_success_rate * assignee_success_rate)

  it_train <- itoken(train_fold$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  it_validate <- itoken(validation_fold$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  vocab <- create_vocabulary(it_train, stopwords = stopwords("en")) %>% prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.5)
  vectorizer <- vocab_vectorizer(vocab)
  tfidf <- TfIdf$new()
  dtm_train <- create_dtm(it_train, vectorizer) %>% fit_transform(tfidf)
  dtm_validate <- create_dtm(it_validate, vectorizer) %>% transform(tfidf)
  
  feature_cols_structured <- setdiff(names(train_fold), c("opening", "opening_date", "final_resolution", "short_desc", "reporter", "assigned_to_initial"))
  train_structured <- as.data.table(train_fold)[, ..feature_cols_structured]
  validation_structured <- as.data.table(validation_fold)[, ..feature_cols_structured]
  for(col in names(train_structured)) {
    if(is.character(train_structured[[col]]) || is.factor(train_structured[[col]]) || is.logical(train_structured[[col]])) {
      all_levels <- unique(c(train_structured[[col]], validation_structured[[col]]))
      train_structured[, (col) := as.integer(factor(.SD[[col]], levels=all_levels, ordered = FALSE))]
      validation_structured[, (col) := as.integer(factor(.SD[[col]], levels=all_levels, ordered = FALSE))]
    }
  }

  train_combined <- cbind(as.data.frame(as.matrix(train_structured)), as.data.frame(as.matrix(dtm_train)))
  validation_combined <- cbind(as.data.frame(as.matrix(validation_structured)), as.data.frame(as.matrix(dtm_validate)))
  
  # Impute NAs
  train_combined[is.na(train_combined)] <- 0
  validation_combined[is.na(validation_combined)] <- 0
  
  # Add target variable to training data for ranger formula
  train_combined$final_resolution <- train_fold$final_resolution

  # Train and Predict with ranger
  ranger_model <- ranger(
    dependent.variable.name = "final_resolution",
    data = train_combined,
    num.trees = 250,
    probability = TRUE,
    importance = 'impurity'
  )
  
  pred_obj <- predict(ranger_model, data = validation_combined)
  pred_probs <- pred_obj$predictions
  
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
  
  # Final feature engineering for the full dataset...
  final_data_full <- final_data %>% mutate(reporter = as.character(reporter), assigned_to_initial = as.character(assigned_to_initial))
  edges_final <- final_data_full %>% select(reporter, assigned_to_initial) %>% filter(!is.na(reporter) & !is.na(assigned_to_initial))
  g_final <- graph_from_data_frame(edges_final, directed = FALSE)
  pagerank_final <- page_rank(g_final)$vector
  pagerank_df_final <- data.frame(person = names(pagerank_final), pagerank = pagerank_final)
  final_data_full <- final_data_full %>% left_join(pagerank_df_final, by = c("reporter" = "person")) %>% rename(reporter_pagerank = pagerank) %>% left_join(pagerank_df_final, by = c("assigned_to_initial" = "person")) %>% rename(assignee_pagerank = pagerank)
  final_data_full <- final_data_full %>% mutate(reporter_pagerank = ifelse(is.na(reporter_pagerank), 0, reporter_pagerank), assignee_pagerank = ifelse(is.na(assignee_pagerank), 0, assignee_pagerank))
  top_reporters_final <- final_data_full %>% count(reporter, sort = TRUE) %>% top_n(100) %>% pull(reporter)
  top_assignees_final <- final_data_full %>% count(assigned_to_initial, sort = TRUE) %>% top_n(100) %>% pull(assigned_to_initial)
  final_data_full <- final_data_full %>% mutate(reporter_binned = ifelse(reporter %in% top_reporters_final, reporter, "OTHER"), assignee_binned = ifelse(assigned_to_initial %in% top_assignees_final, assigned_to_initial, "OTHER"))
  global_rate_final <- sum(final_data_full$final_resolution == "fixed") / nrow(final_data_full)
  reporter_rates_final <- final_data_full %>% group_by(reporter_binned) %>% summarise(reporter_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_rates_final <- final_data_full %>% group_by(assignee_binned) %>% summarise(assignee_success_rate = sum(final_resolution == "fixed") / n(), .groups = "drop")
  assignee_comp_rates_final <- final_data_full %>% group_by(assignee_component) %>% summarise(assignee_comp_success_rate = sum(final_resolution == "fixed") / n(), .groups="drop")
  final_data_full <- final_data_full %>%
    left_join(reporter_rates_final, by = "reporter_binned") %>%
    left_join(assignee_rates_final, by = "assignee_binned") %>%
    left_join(assignee_comp_rates_final, by = "assignee_component") %>%
    mutate(reporter_success_rate = ifelse(is.na(reporter_success_rate), global_rate_final, reporter_success_rate), 
           assignee_success_rate = ifelse(is.na(assignee_success_rate), global_rate_final, assignee_success_rate),
           assignee_comp_success_rate = ifelse(is.na(assignee_comp_success_rate), global_rate_final, assignee_comp_success_rate),
           reputation_interaction = reporter_success_rate * assignee_success_rate)
  
  it_final <- itoken(final_data_full$short_desc, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
  vocab_final <- create_vocabulary(it_final, stopwords = stopwords("en")) %>% prune_vocabulary(term_count_min = 10)
  vectorizer_final <- vocab_vectorizer(vocab_final)
  tfidf_final <- TfIdf$new()
  dtm_final <- create_dtm(it_final, vectorizer_final) %>% fit_transform(tfidf_final)
  feature_cols_structured_final <- setdiff(names(final_data_full), c("opening", "opening_date", "final_resolution", "target_numeric", "short_desc"))
  final_structured_data <- as.data.table(final_data_full)[, ..feature_cols_structured_final]
  for(col in names(final_structured_data)) { if(is.character(final_structured_data[[col]]) || is.factor(final_structured_data[[col]]) || is.logical(final_structured_data[[col]])) { set(final_structured_data, j = col, value = as.integer(as.factor(final_structured_data[[col]]))) } }
  final_combined_data <- cbind(as.data.frame(as.matrix(final_structured_data)), as.data.frame(as.matrix(dtm_final)))
  final_combined_data[is.na(final_combined_data)] <- 0
  final_combined_data$final_resolution <- final_data_full$final_resolution
  
  final_rf_model <- ranger(
    dependent.variable.name = "final_resolution",
    data = final_combined_data,
    num.trees = 500,
    probability = FALSE, # Set to FALSE for faster final prediction
    importance = 'impurity'
  )

  importance_df <- as.data.frame(importance(final_rf_model))
  importance_df$Feature <- rownames(importance_df)
  importance_df <- importance_df[order(-importance_df$importance), ]
  cat("\n--- Top 15 Feature Importance from Final Model (Impurity) ---\n")
  print(head(importance_df, 15))

  cat("\n--- Confusion Matrix for Final Model (on all data) ---\n")
  predicted_labels <- predict(final_rf_model, data = final_combined_data)$predictions
  actual_labels <- final_data_full$final_resolution
  conf_matrix <- confusionMatrix(data = predicted_labels, reference = actual_labels)
  print(conf_matrix)

} else {
  cat("\nWalk-forward validation did not produce any results. Please check data quality and fold sizes.\n")
}
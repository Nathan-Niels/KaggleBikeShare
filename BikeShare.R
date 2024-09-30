library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(earth)

## EDA
train_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/train.csv")
train_data

glimpse(train_data)
skim(train_data)
plot_intro(train_data)
plot_correlation(train_data)
plot_bar(train_data)
plot_histogram(train_data)

# Explore percent missing data
plot_missing(train_data)
# No missing data

# Plot of atemp and temp
plot1 <- ggplot(data = train_data,
                mapping = aes(x = atemp,
                              y = temp)) +
  geom_point() +
  geom_smooth(se = FALSE)

plot1

# Barplot of weather
plot2 <- ggplot(data = train_data,
                mapping = aes(x = weather)) +
  geom_bar()

plot2

# Plot of holiday and working day
plot3 <- ggplot(data = train_data,
                mapping = aes(x = holiday,
                              y = workingday)) +
  geom_point()
plot3

# Plot of datetime
plot4 <- ggplot(data = train_data,
                mapping = aes(x = datetime)) +
  geom_histogram(bins = 15)
plot4

# 4 panel ggplot of 4 key features
(plot1 + plot2) / (plot3 + plot4)

## Data Cleaning
test_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/test.csv")

clean_train_data <- train_data %>% 
  mutate(log_count = log(count)) %>% 
  select(-casual, -registered, -count)
view(clean_train_data)
clean_train_data

## Feature Engineering
bike_recipe <- recipe(log_count ~ ., data = clean_train_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = as.factor(weather)) %>% 
  step_time(datetime, features = "hour") %>% 
  step_mutate(datetime_hour = as.factor(datetime_hour)) %>% 
  step_mutate(season = as.factor(season)) %>%
  step_mutate(holiday = as.factor(holiday)) %>% 
  step_mutate(workingday = as.factor(workingday)) %>% 
  step_rm(datetime) %>% 
  step_rm(atemp) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

prepped_bike_recipe <- prep(bike_recipe)
bake(prepped_bike_recipe, new_data = clean_train_data)

complex_bike_recipe <- recipe(log_count ~ ., data = clean_train_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = as.factor(weather)) %>% 
  step_time(datetime, features = "hour") %>% 
  step_mutate(datetime_hour = as.factor(datetime_hour)) %>% 
  step_mutate(season = as.factor(season)) %>%
  step_mutate(holiday = as.factor(holiday)) %>% 
  step_mutate(workingday = as.factor(workingday)) %>% 
  step_rm(datetime) %>% 
  step_rm(atemp) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_interact(~ all_predictors() * all_predictors())
prepped_complex_bike_recipe <- prep(complex_bike_recipe) 
bake(prepped_complex_bike_recipe, new_data = clean_train_data)


bike_lm <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(bike_lm) %>% 
  fit(data = clean_train_data)

bike_preds <- exp(predict(bike_workflow, new_data = test_data))

# Prepare for submission
kaggle_submission <- bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/LinearPreds.csv", 
            delim = ",")

## Penalized regression model
preg_model <- linear_reg(penalty = .01, mixture = 0.1) %>% 
  set_engine("glmnet")

preg_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(preg_model) %>% 
  fit(data = clean_train_data)

preg_bike_preds <- exp(predict(preg_wf, new_data = test_data))

# Prepare for submission
kaggle_submission <- preg_bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/PregLinearPreds.csv", 
            delim = ",")

## Tuning Model

# Penalized linear regression model
tpreg_model <- linear_reg(penalty = tune(),
                          mixture = tune()) %>% # Setting these as tuning parameters
  set_engine("glmnet")

# Set workflow
tpreg_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(tpreg_model)

# Grid of values to tune over
tune_grid <- grid_regular(penalty(),
                          mixture(),
                          levels = 5) # 25 total tuning parameter possibilities

# Split data for CV
folds <- vfold_cv(clean_train_data, v = 5, repeats = 1)

# Run CV
CV_results <- tpreg_wf %>% 
  tune_grid(resamples = folds,
            grid = tune_grid,
            metrics = metric_set(rmse, mae, rsq))

# Plot Results
collect_metrics(CV_results) %>% 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., 
         aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()

# Find Best Tuning Parameters
best_tune <- CV_results %>% 
  select_best(metric = "rmse")
best_tune

# Finalize workflow and fit it
final_wf <- tpreg_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = clean_train_data)

# Predict
tpreg_bike_preds <- exp(predict(final_wf, new_data = test_data))

# Prepare for submission
kaggle_submission <- tpreg_bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/TPregLinearPreds.csv", 
            delim = ",")

## Regression Tree
rt_model <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

# Set workflow
rt_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(rt_model)

# Grid of values to tune over
rt_tune_grid <- grid_regular(tree_depth(),
                          cost_complexity(),
                          min_n(),
                          levels = 5) # 25 total tuning parameter possibilities

# Split data for CV
rt_folds <- vfold_cv(clean_train_data, v = 5, repeats = 1)

# Run CV
rt_CV_results <- rt_wf %>% 
  tune_grid(resamples = rt_folds,
            grid = rt_tune_grid,
            metrics = metric_set(rmse, mae, rsq))

# Plot Results
collect_metrics(rt_CV_results) %>% 
  filter(.metric == "rmse") %>% 
  ggplot(data = ., 
         aes(x = tree_depth, y = cost_complexity, color = factor(min_n))) +
  geom_line()

# Find Best Tuning Parameters
rt_best_tune <- rt_CV_results %>% 
  select_best(metric = "rmse")
rt_best_tune

# Finalize workflow and fit it
final_wf <- rt_wf %>% 
  finalize_workflow(rt_best_tune) %>% 
  fit(data = clean_train_data)

# Predict
rt_bike_preds <- exp(predict(final_wf, new_data = test_data))

# Prepare for submission
kaggle_submission <- rt_bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/RTLinearPreds.csv", 
            delim = ",")

## Random Forest

rf_model <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Create workflow with model & recipe
rf_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(rf_model)

# Set up grid of tuning values
rf_tune_grid <- grid_regular(mtry(range = c(1,33)),
                             min_n(),
                             levels = 5) 

# Set up K-fold CV
rf_folds <- vfold_cv(clean_train_data, v = 5, repeats = 1)

rf_CV_results <- rf_wf %>% 
  tune_grid(resamples = rf_folds,
            grid = rf_tune_grid,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
rf_best_tune <- rf_CV_results %>% 
  select_best(metric = "rmse")
rf_best_tune

# Finalize workflow and predict
rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_best_tune) %>% 
  fit(data = clean_train_data)

rf_bike_preds <- exp(predict(rf_final_wf, new_data = test_data))

# Prepare for kaggle submission
kaggle_submission <- rf_bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/RFLinearPreds.csv", 
            delim = ",")

## Stacking
# Split data for CV
folds <- vfold_cv(clean_train_data, v = 5, repeats = 1)

# Create a control grid
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Penalized regression model
spreg_model <- linear_reg(penalty = tune(),
                          mixture = tune()) %>% 
  set_engine("glmnet")

# Set Workflow
spreg_wf <- workflow() %>% 
  add_recipe(complex_bike_recipe) %>% 
  add_model(spreg_model)

# Grid of tuning parameter values
spreg_tuning_grid <- grid_regular(penalty(),
                                  mixture(),
                                  levels = 5)

# Run CV
spreg_models <- spreg_wf %>% 
  tune_grid(resamples = folds,
            grid = spreg_tuning_grid,
            metrics = metric_set(rmse, mae),
            control = untunedModel)

# Linear regression model
slin_reg <-
  linear_reg() %>% 
  set_engine("lm")

# Set workflow
slin_reg_wf <-
  workflow() %>% 
  add_model(slin_reg) %>% 
  add_recipe(bike_recipe)

# Finalize model
slin_reg_model <-
  fit_resamples(
    slin_reg_wf,
    resamples = folds,
    metrics = metric_set(rmse),
    control = tunedModel
  )

# Random Forest Models
srf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 50) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Create workflow with model & recipe
srf_wf <- workflow() %>% 
  add_recipe(complex_bike_recipe) %>% 
  add_model(srf_model)

# Set up grid of tuning values
srf_tune_grid <- grid_regular(mtry(range = c(1:33)),
                             min_n(),
                             levels = 5)

# Run Cv
srf_models <- srf_wf %>% 
  tune_grid(resamples = folds,
            grid = srf_tune_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

# Specify which models to include
my_stack <- stacks() %>% 
  add_candidates(slin_reg_model) %>% 
  add_candidates(spreg_models) %>% 
  add_candidates(srf_models)

# Fit the stacked model
stack_mod <- my_stack %>% 
  blend_predictions() %>% 
  fit_members()

# Use stacked data to get prediction
stack_preds <- stack_mod %>% 
                  predict(new_data = test_data)

# Prepare for kaggle submission
kaggle_submission <- stack_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/StackLinearPreds.csv", 
            delim = ",")

# MARS
mars_mod <- mars(num_terms = tune(),
                 prod_degree = tune(),
                 prune_method = "backward") %>% 
  set_mode("regression") %>% 
  set_engine("earth")

mars_wf <- workflow() %>% 
  add_recipe(complex_bike_recipe) %>% 
  add_model(mars_mod)

mars_tune_grid <- expand.grid(num_terms = c(1:10),
                              prod_degree = c(1,2))

mars_CV_results <- mars_wf %>% 
  tune_grid(resamples = folds,
            grid = mars_tune_grid,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
mars_best_tune <- mars_CV_results %>% 
  select_best(metric = "rmse")
mars_best_tune

# Finalize workflow and predict
mars_final_wf <- mars_wf %>% 
  finalize_workflow(mars_best_tune) %>% 
  fit(data = clean_train_data)

mars_bike_preds <- exp(predict(mars_final_wf, new_data = test_data))


# Prepare for kaggle submission
kaggle_submission <- mars_bike_preds %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/MarsLinearPreds.csv", 
            delim = ",")

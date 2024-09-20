library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(glmnet)

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

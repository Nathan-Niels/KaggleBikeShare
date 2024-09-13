library(tidymodels)
library(tidyverse)
library(vroom)

train_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/train.csv")
test_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/test.csv")

## Setup and Fit Linear Regression Model
my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula = count ~ season+holiday+workingday+weather+temp+humidity+windspeed,
      data = train_data)

## Generate Predictions Using LInear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test_data)
bike_predictions

## Format the Predictions for Kaggle Submission
kaggle_submission <- bike_predictions %>% 
  bind_cols(., test_data) %>% 
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(count = pmax(0, count)) %>% 
  mutate(datetime = as.character(format(datetime)))

## Write out the file
vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/LinearPreds.csv", 
            delim = ",")

## Poisson Regression
library(poissonreg)

# Write poisson model
my_pois_model <- poisson_reg() %>% 
  set_engine("glm") %>% # glm = generalized linear model
  set_mode("regression") %>% 
  fit(formula = count ~ season + holiday + workingday + weather +
        temp + humidity + windspeed, # omitted atemp
      data = train_data)

# Generating predictions using linear model
bike_predictions_pois <- predict(my_pois_model,
                                 new_data = test_data)
bike_predictions_pois

# Format predictions for Kaggle submission
pois_kaggle_submission <- bike_predictions_pois %>% 
  bind_cols(., test_data) %>% # bind predictions with test data
  select(datetime, .pred) %>% 
  rename(count = .pred) %>% 
  mutate(datetime = as.character(format(datetime)))

# Write out file
vroom_write(x = pois_kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/KaggleBikeShare/PoissonPreds.csv", 
            delim = ",")

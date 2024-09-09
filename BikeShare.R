library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)


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

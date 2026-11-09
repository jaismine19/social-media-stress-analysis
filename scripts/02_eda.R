# ==========================================
# PART 2: EXPLORATORY DATA ANALYSIS
# Social Media and Stress Patterns
# ==========================================

# Run from the social-media-stress-analysis repository folder.

# ==========================================
# 1. LOAD PACKAGES
# ==========================================

# Run this ONCE in the Console if these packages are missing:
# install.packages(c("tidyverse", "moments"))

library(tidyverse)
library(moments)
library(janitor)
library(corrplot)

# ==========================================
# 2. IMPORT DATASET
# ==========================================

# Find the dataset from either the project folder
# or the scripts folder.
data_path <- "data/processed/social_media_stress_clean.csv"

if (!file.exists(data_path)) {
  data_path <- "../data/processed/social_media_stress_clean.csv"
}

df <- read_csv(
  data_path,
  show_col_types = FALSE
)

# ==========================================
# 3. DATASET INSPECTION
# ==========================================

# Number of rows and columns
dim(df)

# Column names and data types
glimpse(df)

# General summary
summary(df)
dim(df)

# ==========================================
# 4. DATA QUALITY CHECKS
# ==========================================

# Missing values in each column
colSums(is.na(df))

# Completely duplicated rows
sum(duplicated(df))

# Duplicated participant IDs
sum(duplicated(df$user_id))

# ==========================================
# 5. CATEGORICAL VARIABLE DISTRIBUTIONS
# ==========================================

# Gender
gender_summary <- df %>%
  count(gender) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

print(gender_summary)

# Social media platform
platform_summary <- df %>%
  count(social_media_platform) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

print(platform_summary)

# Screen-time group
screen_time_summary <- df %>%
  count(screen_time_group) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

print(screen_time_summary)

# ==========================================
# 6. CATEGORICAL VARIABLE VISUALIZATIONS
# ==========================================

# Gender distribution - Bar Chart
gender_plot <- ggplot(df, aes(x = gender)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Gender Distribution",
    x = "Gender",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(gender_plot)

# Social media platform distribution
platform_plot <- ggplot(df, aes(x = social_media_platform)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Social Media Platform Distribution",
    x = "Social Media Platform",
    y = "Number of Participants"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(platform_plot)

# Screen-time group distribution
screen_time_group_plot <- ggplot(df, aes(x = screen_time_group)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Screen-Time Group Distribution",
    x = "Screen-Time Group",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(screen_time_group_plot)

# ==========================================
# 7. NUMERICAL VARIABLE SUMMARY
# ==========================================

# Exclude user_id because it identifies participants.
numeric_vars <- c(
  "age",
  "daily_screen_time_hours",
  "sleep_quality",
  "stress_level",
  "days_without_social_media",
  "exercise_frequency_per_week",
  "happiness_index"
)

eda_summary <- data.frame(
  Variable = numeric_vars,
  Mean = sapply(df[numeric_vars], mean, na.rm = TRUE),
  Median = sapply(df[numeric_vars], median, na.rm = TRUE),
  SD = sapply(df[numeric_vars], sd, na.rm = TRUE),
  Minimum = sapply(df[numeric_vars], min, na.rm = TRUE),
  Maximum = sapply(df[numeric_vars], max, na.rm = TRUE),
  Skewness = sapply(
    df[numeric_vars],
    moments::skewness,
    na.rm = TRUE
  ),
  row.names = NULL
)

# Round numerical columns only.
eda_summary[, -1] <- round(eda_summary[, -1], 2)

print(eda_summary)

# ==========================================
# 8. NUMERICAL VARIABLE HISTOGRAMS
# ==========================================

# Daily screen time
# Bins have edges at 0, 1, 2, 3, ... hours.
screen_time_plot <- ggplot(
  df,
  aes(x = daily_screen_time_hours)
) +
  geom_histogram(
    binwidth = 1,
    boundary = 0,
    fill = "steelblue",
    color = "white"
  ) +
  scale_x_continuous(
    breaks = seq(0, 12, by = 1)
  ) +
  labs(
    title = "Distribution of Daily Screen Time",
    x = "Daily Screen Time (Hours)",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(screen_time_plot)

# Age
age_plot <- ggplot(df, aes(x = age)) +
  geom_histogram(
    binwidth = 5,
    boundary = 0,
    fill = "steelblue",
    color = "white"
  ) +
  labs(
    title = "Age Distribution",
    x = "Age (Years)",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(age_plot)

# Sleep quality
sleep_plot <- ggplot(df, aes(x = sleep_quality)) +
  geom_histogram(
    binwidth = 1,
    boundary = 0.5,
    fill = "steelblue",
    color = "white"
  ) +
  scale_x_continuous(
    breaks = scales::breaks_width(1)
  ) +
  labs(
    title = "Sleep Quality Distribution",
    x = "Sleep Quality Score",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(sleep_plot)

# Stress level
stress_plot <- ggplot(df, aes(x = stress_level)) +
  geom_histogram(
    binwidth = 1,
    boundary = 0.5,
    fill = "steelblue",
    color = "white"
  ) +
  scale_x_continuous(
    breaks = scales::breaks_width(1)
  ) +
  labs(
    title = "Stress Level Distribution",
    x = "Stress Level Score",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(stress_plot)

# Days without social media
days_without_plot <- ggplot(
  df,
  aes(x = days_without_social_media)
) +
  geom_histogram(
    binwidth = 1,
    boundary = -0.5,
    fill = "steelblue",
    color = "white"
  ) +
  scale_x_continuous(
    breaks = scales::breaks_width(1)
  ) +
  labs(
    title = "Days Without Social Media Distribution",
    x = "Number of Days",
    y = "Number of Participants"
  ) +
  theme_minimal()

print(days_without_plot)

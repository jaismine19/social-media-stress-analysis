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

# ==========================================
# 9. COMPLETE EDA CHART SET
# ==========================================

library(tidyverse)
library(corrplot)

# Find the project folder for saving outputs.
project_dir <- if (dir.exists("data/processed")) {
  "."
} else if (dir.exists("../data/processed")) {
  ".."
} else {
  stop("Open the project folder or its scripts folder, then run again.")
}

figures_dir <- file.path(project_dir, "reports", "figures")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Readable labels
variable_labels <- c(
  age = "Age (Years)",
  daily_screen_time_hours = "Daily Screen Time (Hours)",
  sleep_quality = "Sleep Quality Score",
  stress_level = "Stress Level Score",
  days_without_social_media = "Days Without Social Media",
  exercise_frequency_per_week = "Exercise Frequency per Week",
  happiness_index = "Happiness Index",
  gender = "Gender",
  social_media_platform = "Social Media Platform",
  screen_time_group = "Screen-Time Group"
)

numeric_vars <- c(
  "age", "daily_screen_time_hours", "sleep_quality",
  "stress_level", "days_without_social_media",
  "exercise_frequency_per_week", "happiness_index"
)

# Hold charts and observations for later use.
eda_plots <- list()
eda_notes <- list()

# Common chart formatting
eda_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.caption = element_text(hjust = 0, size = 9),
    plot.title = element_text(face = "bold")
  )

# ==========================================
# 10. HISTOGRAM AND BOXPLOT: EVERY NUMERIC VARIABLE
# ==========================================

for (v in numeric_vars) {
  
  values <- df[[v]]
  valid <- values[is.finite(values)]
  
  note <- sprintf(
    "Mean: %.2f; median: %.2f. Observed values range from %.2f to %.2f.",
    mean(valid), median(valid), min(valid), max(valid)
  )
  
  width <- if (v == "age") 5 else 1
  edge <- if (v == "age") 0 else 0.5
  
  p_hist <- ggplot(df, aes(x = .data[[v]])) +
    geom_histogram(
      binwidth = width,
      boundary = edge,
      fill = "steelblue",
      color = "white",
      na.rm = TRUE
    ) +
    labs(
      title = paste("Distribution of", variable_labels[[v]]),
      x = variable_labels[[v]],
      y = "Number of Participants",
      caption = note
    ) +
    eda_theme
  
  p_box <- ggplot(df, aes(x = "", y = .data[[v]])) +
    geom_boxplot(fill = "steelblue", na.rm = TRUE) +
    labs(
      title = paste("Boxplot of", variable_labels[[v]]),
      x = NULL,
      y = variable_labels[[v]],
      caption = sprintf(
        "Median: %.2f. The middle 50%% of values lie between %.2f and %.2f.",
        median(valid),
        quantile(valid, 0.25),
        quantile(valid, 0.75)
      )
    ) +
    eda_theme
  
  eda_plots[[paste0(v, "_histogram")]] <- p_hist
  eda_plots[[paste0(v, "_boxplot")]] <- p_box
  eda_notes[[paste0(v, "_distribution")]] <- note
  
  print(p_hist)
  print(p_box)
}
print(eda_plots[["exercise_frequency_per_week_histogram"]])
print(eda_plots[["exercise_frequency_per_week_boxplot"]])
# ==========================================
# 11. CATEGORICAL CHARTS
# ==========================================

for (v in c("gender", "social_media_platform", "screen_time_group")) {
  
  p <- ggplot(df, aes(x = .data[[v]])) +
    geom_bar(fill = "steelblue") +
    labs(
      title = paste(variable_labels[[v]], "Distribution"),
      x = variable_labels[[v]],
      y = "Number of Participants"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  
  key <- paste0(v, "_bar")
  eda_plots[[key]] <- p
  
  print(p)
  
  ggsave(
    filename = file.path(figures_dir, paste0(key, ".png")),
    plot = p,
    width = 11,
    height = 7,
    dpi = 300,
    bg = "white"
  )
}
print(eda_plots[["gender_bar"]])
print(eda_plots[["social_media_platform_bar"]])
print(eda_plots[["screen_time_group_bar"]])




# ==========================================
# 12. NUMERIC RELATIONSHIPS
# ==========================================

# Seven scatterplots cover the numerical relationship questions.
scatter_pairs <- tribble(
  ~x,                           ~y,
  "daily_screen_time_hours",     "stress_level",
  "daily_screen_time_hours",     "sleep_quality",
  "exercise_frequency_per_week", "stress_level",
  "exercise_frequency_per_week", "happiness_index",
  "age",                        "daily_screen_time_hours",
  "age",                        "stress_level",
  "age",                        "happiness_index"
)

for (i in seq_len(nrow(scatter_pairs))) {
  
  x <- scatter_pairs$x[i]
  y <- scatter_pairs$y[i]
  
  r <- cor(df[[x]], df[[y]], use = "complete.obs")
  
  direction <- if (!is.finite(r)) {
    "not estimable"
  } else if (abs(r) < 0.10) {
    "close to zero"
  } else if (r > 0) {
    "positive"
  } else {
    "negative"
  }
  
  note <- sprintf(
    "Pearson correlation: r = %.2f (%s). This describes association, not causation.",
    r, direction
  )
  
  p <- ggplot(df, aes(x = .data[[x]], y = .data[[y]])) +
    geom_point(
      position = if (
        x == "exercise_frequency_per_week" &&
        y == "happiness_index"
      ) {
        position_jitter(width = 0.15, height = 0.15, seed = 123)
      } else {
        "identity"
      },
      color = "#003366",
      alpha = 0.45,
      size = 2,
      na.rm = TRUE
    ) +
    labs(
      title = paste(variable_labels[[y]], "vs", variable_labels[[x]]),
      x = variable_labels[[x]],
      y = variable_labels[[y]],
      caption = note
    ) +
    eda_theme
  
  key <- paste(x, y, sep = "_vs_")
  eda_plots[[key]] <- p
  eda_notes[[key]] <- note
  
  print(p)
}
eda_plots[["exercise_frequency_per_week_vs_happiness_index"]] <-
  ggplot(df, aes(
    x = exercise_frequency_per_week,
    y = happiness_index
  )) +
  geom_point(
    position = position_jitter(
      width = 0.2,
      height = 0.15,
      seed = 123
    ),
    color = "#003366",
    alpha = 0.45,
    size = 2
  ) +
  scale_x_continuous(breaks = 0:7) +
  scale_y_continuous(breaks = 4:10) +
  labs(
    title = "Exercise Frequency vs.\nHappiness Index",
    x = "Exercise Frequency per Week",
    y = "Happiness Index",
    caption = "Points shifted slightly to reveal overlapping participants."
  ) +
  theme_minimal()

print(eda_plots[["exercise_frequency_per_week_vs_happiness_index"]])

print(eda_plots[["daily_screen_time_hours_vs_sleep_quality"]])
print(eda_plots[["sleep_quality_by_social_media_platform"]])


# ==========================================
# 13. GROUP COMPARISONS
# ==========================================

group_pairs <- tribble(
  ~x,                      ~y,
  "social_media_platform",  "stress_level",
  "screen_time_group",      "stress_level",
  "social_media_platform",  "sleep_quality",
  "gender",                 "stress_level"
)

for (i in seq_len(nrow(group_pairs))) {
  
  x <- group_pairs$x[i]
  y <- group_pairs$y[i]
  
  medians <- df %>%
    group_by(.data[[x]]) %>%
    summarise(Median = median(.data[[y]], na.rm = TRUE),
              .groups = "drop")
  
  median_text <- paste(
    paste0(medians[[x]], ": ", round(medians$Median, 2)),
    collapse = "; "
  )
  
  note <- paste0(
    "Group medians — ", median_text,
    ". Compare the boxes and their overlap to assess visible differences."
  )
  
  p <- ggplot(df, aes(x = .data[[x]], y = .data[[y]])) +
    geom_boxplot(fill = "steelblue", na.rm = TRUE) +
    labs(
      title = paste(variable_labels[[y]], "by", variable_labels[[x]]),
      x = variable_labels[[x]],
      y = variable_labels[[y]],
      caption = stringr::str_wrap(note, width = 95)
    ) +
    eda_theme +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  key <- paste(y, "by", x, sep = "_")
  eda_plots[[key]] <- p
  eda_notes[[key]] <- note
  
  print(p)
}

# ==========================================
# 14. CORRELATION HEATMAP
# ==========================================

# Participant ID is deliberately excluded.
correlation_matrix <- cor(
  df[numeric_vars],
  use = "complete.obs"
)

# Shorter labels keep the heatmap readable.
heatmap_labels <- c(
  "Age", "Screen time", "Sleep quality", "Stress",
  "Days offline", "Exercise", "Happiness"
)

dimnames(correlation_matrix) <- list(
  heatmap_labels, heatmap_labels
)

corrplot(
  correlation_matrix,
  method = "color",
  type = "upper",
  diag = FALSE,
  addCoef.col = "black",
  number.cex = 0.75,
  tl.col = "black",
  tl.srt = 45,
  title = "Correlations Between Numerical Variables",
  mar = c(0, 0, 3, 0)
)

# Find the strongest observed correlation, excluding the diagonal.
pair_indices <- which(
  upper.tri(correlation_matrix),
  arr.ind = TRUE
)

pair_values <- correlation_matrix[pair_indices]
usable <- which(is.finite(pair_values))

if (length(usable) > 0) {
  strongest <- usable[which.max(abs(pair_values[usable]))]
  pair <- pair_indices[strongest, ]
  
  heatmap_note <- sprintf(
    paste0(
      "The strongest observed linear association is between %s and %s ",
      "(r = %.2f). Correlation does not establish causation."
    ),
    heatmap_labels[pair[1]],
    heatmap_labels[pair[2]],
    pair_values[strongest]
  )
} else {
  heatmap_note <- "Correlations could not be estimated."
}

eda_notes[["correlation_heatmap"]] <- heatmap_note
cat(heatmap_note, "\n")

# ==========================================
# 15. EXPORT FOUR SELECTED CHARTS
# ==========================================

selected_charts <- c(
  "daily_screen_time_hours_histogram",
  "daily_screen_time_hours_vs_stress_level",
  "stress_level_by_screen_time_group"
)

for (key in selected_charts) {
  ggsave(
    filename = file.path(figures_dir, paste0(key, ".png")),
    plot = eda_plots[[key]],
    width = 10,
    height = 7,
    dpi = 300,
    bg = "white"
  )
}

# The heatmap is the fourth exported chart.
png(
  file.path(figures_dir, "correlation_heatmap.png"),
  width = 2400,
  height = 2200,
  res = 250
)

corrplot(
  correlation_matrix,
  method = "color",
  type = "upper",
  diag = FALSE,
  addCoef.col = "black",
  number.cex = 0.75,
  tl.col = "black",
  tl.srt = 45,
  title = "Correlations Between Numerical Variables",
  mar = c(0, 0, 3, 0)
)

mtext(
  paste(strwrap(heatmap_note, width = 90), collapse = "\n"),
  side = 1,
  line = -1,
  cex = 0.65
)

dev.off()

# Save the calculated observations as a reference.
notes_text <- unlist(
  lapply(names(eda_notes), function(key) {
    c(paste0("## ", key), eda_notes[[key]], "")
  }),
  use.names = FALSE
)

writeLines(
  c("# EDA Chart Observations", "", notes_text),
  file.path(project_dir, "reports", "part2_chart_observations.md")
)

message("EDA charts completed. Four PNGs saved in: ", figures_dir)
# Counts and percentages for screen-time groups and platforms
print(screen_time_summary)
print(platform_summary)

# Stress distribution within each group
df %>%
  group_by(screen_time_group) %>%
  summarise(
    n = n(),
    Median = median(stress_level, na.rm = TRUE),
    Q1 = quantile(stress_level, 0.25, na.rm = TRUE),
    Q3 = quantile(stress_level, 0.75, na.rm = TRUE)
  )

# Relationships with age
df %>%
  summarise(
    Age_ScreenTime = cor(
      age, daily_screen_time_hours, use = "complete.obs"
    ),
    Age_Stress = cor(
      age, stress_level, use = "complete.obs"
    ),
    Age_Happiness = cor(
      age, happiness_index, use = "complete.obs"
    )
  ) %>%
  mutate(across(everything(), ~ round(.x, 2)))
    library(ggplot2)
    
    ggplot(df, aes(x = daily_screen_time_hours, y = sleep_quality)) +
      geom_point(color = "steelblue", alpha = 0.35) +
      labs(
        title = "Daily Screen Time vs. Sleep Quality",
        x = "Daily Screen Time (Hours)",
        y = "Sleep Quality Score"
      ) +
      theme_minimal()
    
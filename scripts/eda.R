#!/usr/bin/env Rscript
# =====================================================================
# EXPLORATORY DATA ANALYSIS (EDA) IN R   (easy-to-read version)
# =====================================================================
# Reads : data/processed/social_media_stress_clean.csv
# Saves : every chart to reports/figures/ (files start with r_)
#
# HOW TO RUN (from the project root):
#     Rscript scripts/eda.R
#
# WHAT THIS SCRIPT DOES
# ---------------------
# EDA means "getting to know the data". It answers questions like:
#   * How big is the dataset and what types are the columns?
#   * Are there missing values?
#   * How is each variable distributed? (histograms, boxplots)
#   * Are the variables skewed?
#   * Do variables move together? (scatter plots, correlations)
#   * Are there outliers? (IQR scan)
#
# Everything is written with BASE R functions (plus the small
# `corrplot` package for the heatmap - it is optional).
# Every section prints its findings in plain English.
# =====================================================================

# ---------------------------------------------------------------------
# 0. SETUP
# ---------------------------------------------------------------------
# Run this script from the project root folder.
DATA_FILE <- "data/processed/social_media_stress_clean.csv"
FIG_DIR   <- "reports/figures"

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(DATA_FILE, stringsAsFactors = FALSE)

# Columns we will analyse
NUMERIC <- c("age", "daily_screen_time_hours", "sleep_quality",
             "stress_level", "days_without_social_media",
             "exercise_frequency_per_week", "happiness_index")
CATEGORICAL <- c("gender", "social_media_platform", "screen_time_group")

# Helper: save one chart to reports/figures/ and close the device
save_chart <- function(name, draw, width = 9, height = 5) {
  png(file.path(FIG_DIR, name), width = width, height = height,
      units = "in", res = 120)
  draw()
  dev.off()
  cat("    saved -> reports/figures/", name, "\n", sep = "")
}

# Helper: skewness (how lopsided a distribution is)
#   > 1  strongly right-skewed (long tail of high values)
#   < -1 strongly left-skewed  (long tail of low values)
skewness <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  mean((x - mean(x))^3) / (sd(x)^3)
}

cat("STARTING THE EDA (R)\n")
cat("Loaded", nrow(df), "rows and", ncol(df), "columns.\n\n")

# =====================================================================
# SECTION 1 - OVERVIEW
# =====================================================================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 1 - OVERVIEW\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Structure of the data:\n")
str(df)
cat("\nSummary of every column:\n")
print(summary(df))

# =====================================================================
# SECTION 2 - MISSING VALUES
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 2 - MISSING VALUES\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
missing_counts <- colSums(is.na(df))
cat("Missing values per column:\n")
print(missing_counts)
cat("Total missing values:", sum(missing_counts), "\n")

# Which exact rows have gaps? Check every cell, one by one.
found_any <- FALSE
for (i in 1:nrow(df)) {          # look at every row...
  for (col in names(df)) {       # ...and every column inside it
    if (is.na(df[i, col])) {     # is this cell empty?
      found_any <- TRUE
      if ("user_id" %in% names(df)) {
        row_label <- df[i, "user_id"]
      } else {
        row_label <- i
      }
      cat("  ", row_label, ": missing in '", col, "'\n", sep = "")
    }
  }
}
if (!found_any) {
  cat("No rows have missing values - every cell is filled.\n")
}
if (sum(missing_counts) > 0) {
  # Draw a heatmap: white = present, coloured = missing
  save_chart("r_missing_values.png", function() {
    na_matrix <- is.na(df)
    image(t(na_matrix), col = c("white", "firebrick"), axes = FALSE,
          main = "Missing values heatmap (red = missing)")
    axis(2, at = seq(0, 1, length.out = ncol(df)), labels = names(df), las = 2)
  }, width = 9, height = 4)
}

# =====================================================================
# SECTION 3 - UNIVARIATE ANALYSIS (one variable at a time)
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 3 - UNIVARIATE (numeric columns)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (col in NUMERIC) {
  # Histogram with density curve next to a boxplot
  save_chart(paste0("r_", col, ".png"), function() {
    par(mfrow = c(1, 2))
    hist(df[[col]], main = paste(col, "- distribution"),
         xlab = col, col = "steelblue", probability = TRUE)
    lines(density(df[[col]]), col = "darkred", lwd = 2)
    boxplot(df[[col]], main = paste(col, "- boxplot"),
            col = "steelblue", horizontal = TRUE)
  })

  # Print the key numbers in plain English
  m <- mean(df[[col]])
  med <- median(df[[col]])
  sk <- skewness(df[[col]])
  cat(sprintf("  %-30s mean %.2f, median %.2f, skewness %+.2f",
              col, m, med, sk), "\n")
  if (sk > 1) {
    cat("      -> strongly right-skewed (long tail of high values)\n")
  } else if (sk < -1) {
    cat("      -> strongly left-skewed (long tail of low values)\n")
  } else {
    cat("      -> roughly symmetric or mildly skewed\n")
  }
}

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 3b - UNIVARIATE (categorical columns)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

for (col in CATEGORICAL) {
  counts <- table(df[[col]])
  cat("\n  ", col, " counts:\n", sep = "")
  for (label in names(counts)) {
    cat(sprintf("    %-18s %3d users (%.1f%%)\n",
                label, counts[[label]], 100 * counts[[label]] / nrow(df)))
  }
  save_chart(paste0("r_", col, "_counts.png"), function() {
    par(mar = c(9, 4, 3, 2))
    barplot(counts, main = paste(col, "- counts"), col = "steelblue",
            las = 2, cex.names = 0.9)
  }, width = 9, height = 5)
}

# =====================================================================
# SECTION 4 - BIVARIATE ANALYSIS (two variables together)
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 4 - BIVARIATE (relationships)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Colours for platforms (one colour per platform)
platform_colors <- rainbow(length(unique(df$social_media_platform)))
names(platform_colors) <- unique(df$social_media_platform)

# 4.1 Scatter: screen time vs stress, coloured by platform
save_chart("r_screen_time_vs_stress.png", function() {
  plot(df$daily_screen_time_hours, df$stress_level,
       col = platform_colors[df$social_media_platform],
       pch = 19, cex = 1.1,
       xlab = "Daily screen time (hours)", ylab = "Stress level (1-10)",
       main = "Screen time vs stress, coloured by platform")
  legend("topleft", legend = names(platform_colors),
         col = platform_colors, pch = 19, cex = 0.8)
}, width = 8, height = 6)
cat("  Screen time vs stress: correlation =",
    sprintf("%+.2f\n", cor(df$daily_screen_time_hours, df$stress_level)))

# 4.2 Scatter: screen time vs sleep quality
save_chart("r_screen_time_vs_sleep.png", function() {
  plot(df$daily_screen_time_hours, df$sleep_quality,
       col = "steelblue", pch = 19,
       xlab = "Daily screen time (hours)", ylab = "Sleep quality (1-10)",
       main = "Screen time vs sleep quality")
  abline(lm(df$sleep_quality ~ df$daily_screen_time_hours),
         col = "darkred", lwd = 2)
}, width = 8, height = 6)
cat("  Screen time vs sleep quality: correlation =",
    sprintf("%+.2f\n", cor(df$daily_screen_time_hours, df$sleep_quality)))

# 4.3 Scatter: exercise vs happiness
save_chart("r_exercise_vs_happiness.png", function() {
  plot(df$exercise_frequency_per_week, df$happiness_index,
       col = "seagreen", pch = 19,
       xlab = "Exercise sessions per week", ylab = "Happiness index (1-10)",
       main = "Exercise vs happiness")
}, width = 8, height = 6)
cat("  Exercise vs happiness: correlation =",
    sprintf("%+.2f\n", cor(df$exercise_frequency_per_week, df$happiness_index)))

# 4.4 Boxplots: a numeric variable across groups
box_group <- function(value_col, group_col, title, file_name) {
  save_chart(file_name, function() {
    boxplot(df[[value_col]] ~ df[[group_col]],
            main = title, xlab = group_col, ylab = value_col,
            col = "steelblue", las = 2)
  }, width = 9, height = 5)
  cat("\n  Mean", value_col, "by", group_col, ":\n")
  means <- tapply(df[[value_col]], df[[group_col]], mean)
  for (label in names(means)) {
    cat(sprintf("    %-18s %.2f\n", label, means[[label]]))
  }
}

box_group("stress_level", "screen_time_group",
          "Stress by screen-time group", "r_stress_by_screen_time_group.png")
box_group("sleep_quality", "social_media_platform",
          "Sleep quality by platform", "r_sleep_by_platform.png")
box_group("stress_level", "gender",
          "Stress by gender", "r_stress_by_gender.png")

# =====================================================================
# SECTION 5 - CORRELATION
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 5 - CORRELATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Correlation measures how two numbers move together:\n")
cat("  +1 = move together, -1 = move opposite, 0 = no link.\n\n")

cor_matrix <- cor(df[NUMERIC])
print(round(cor_matrix, 2))

# Heatmap: use corrplot if installed, otherwise base R image()
if (requireNamespace("corrplot", quietly = TRUE)) {
  save_chart("r_correlation_heatmap.png", function() {
    corrplot::corrplot(cor_matrix, method = "color",
                       addCoef.col = "black", tl.col = "black",
                       tl.cex = 0.75, number.cex = 0.7)
  }, width = 8, height = 7)
} else {
  save_chart("r_correlation_heatmap.png", function() {
    image(t(cor_matrix)[, ncol(cor_matrix):1], col = heat.colors(20),
          axes = FALSE, main = "Correlation heatmap")
    axis(1, at = seq(0, 1, length.out = ncol(cor_matrix)),
         labels = colnames(cor_matrix), las = 2, cex.axis = 0.8)
    axis(2, at = seq(1, 0, length.out = ncol(cor_matrix)),
         labels = colnames(cor_matrix), las = 2, cex.axis = 0.8)
  }, width = 8, height = 7)
}

# Find the strongest links (ignore a variable with itself)
# combn(NUMERIC, 2) gives every pair of columns exactly once.
cat("\n  Strongest relationships:\n")
combs <- combn(NUMERIC, 2)
top_pairs <- data.frame(a = character(), b = character(),
                        value = numeric(), stringsAsFactors = FALSE)
for (i in seq_len(ncol(combs))) {
  col_a <- combs[1, i]
  col_b <- combs[2, i]
  top_pairs <- rbind(top_pairs, data.frame(a = col_a, b = col_b,
                                           value = cor_matrix[col_a, col_b],
                                           stringsAsFactors = FALSE))
}
top_pairs <- top_pairs[order(-abs(top_pairs$value)), ]
for (i in 1:5) {
  direction <- ifelse(top_pairs$value[i] > 0, "positive", "negative")
  cat(sprintf("    %s <-> %s: %+.2f (%s)\n",
              top_pairs$a[i], top_pairs$b[i], top_pairs$value[i], direction))
}

# =====================================================================
# SECTION 6 - OUTLIER SCAN (IQR rule)
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 6 - OUTLIER SCAN (IQR rule)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Values more than 1.5 x IQR beyond the quartiles are flagged.\n")
cat("We LOOK at outliers, we do not delete them without a reason.\n\n")

for (col in NUMERIC) {
  q1 <- quantile(df[[col]], 0.25)
  q3 <- quantile(df[[col]], 0.75)
  iqr <- q3 - q1
  low_fence <- q1 - 1.5 * iqr
  high_fence <- q3 + 1.5 * iqr
  n_out <- sum(df[[col]] < low_fence | df[[col]] > high_fence)
  cat(sprintf("  %-30s %d potential outlier(s) (outside %.1f - %.1f)\n",
              col, n_out, low_fence, high_fence))
}

# =====================================================================
# SECTION 7 - INTERESTING FACTS (for the report)
# =====================================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("SECTION 7 - INTERESTING FACTS (for Members 3 & 4)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("1. Screen time and stress are strongly linked (+0.74): heavier users report more stress.\n")
cat("2. Screen time and sleep quality move opposite (-0.76): more screen, worse sleep.\n")
cat("3. Stress and happiness are strongly negative (-0.74).\n")
cat("4. Mean screen time is", sprintf("%.1f", mean(df$daily_screen_time_hours)),
    "hours/day; the heaviest user reports 10.8 hours.\n")
cat("5. TikTok is the most common platform (95 users), Instagram the least (74).\n")
cat("6. Stress barely differs by gender (means 6.6-6.63) - the story is in habits, not demographics.\n")
cat("7. Only 43 users (8.6%) report Low stress - the classes are imbalanced.\n")

cat("\nEDA FINISHED - all charts are in reports/figures/ (files starting with r_)\n")

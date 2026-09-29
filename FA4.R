# ==============================================================================
# APM1205 - Formative Assessment 4
# Dummy-Variable Regression Using the diamonds Dataset
# Complete R Code
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load packages and data
# ------------------------------------------------------------------------------

library(ggplot2)

data(diamonds, package = "ggplot2")


# ------------------------------------------------------------------------------
# PART A - Data Exploration and Preparation
# ------------------------------------------------------------------------------

# View the first observations
head(diamonds)

# Inspect the structure
str(diamonds)

# Summary statistics
summary(diamonds)

# Number of observations and variables
n_observations <- nrow(diamonds)
n_variables <- ncol(diamonds)

cat("Number of observations:", n_observations, "\n")
cat("Number of variables:", n_variables, "\n")

# Examine the categories of cut
table(diamonds$cut)

# Set Ideal as the reference category
diamonds$cut <- relevel(diamonds$cut, ref = "Ideal")


# ------------------------------------------------------------------------------
# PART B - Dummy Variables
# ------------------------------------------------------------------------------

# Create four dummy variables.
# Ideal is the reference category, so it is represented by all zeros.

diamonds$D1_Fair <- as.integer(diamonds$cut == "Fair")
diamonds$D2_Good <- as.integer(diamonds$cut == "Good")
diamonds$D3_VeryGood <- as.integer(diamonds$cut == "Very Good")
diamonds$D4_Premium <- as.integer(diamonds$cut == "Premium")

# Display the dummy-variable coding table
dummy_table <- data.frame(
  Cut = c(
    "Fair",
    "Good",
    "Very Good",
    "Premium",
    "Ideal"
  ),
  D1_Fair = c(1, 0, 0, 0, 0),
  D2_Good = c(0, 1, 0, 0, 0),
  D3_VeryGood = c(0, 0, 1, 0, 0),
  D4_Premium = c(0, 0, 0, 1, 0)
)

print(dummy_table)


# ------------------------------------------------------------------------------
# PART C - Additive Dummy-Variable Regression
# ------------------------------------------------------------------------------

# Model:
# price = beta0 + beta1(carat) + dummy-variable effects + error

model1 <- lm(
  price ~ carat + cut,
  data = diamonds
)

# Full regression output
summary(model1)

# Regression coefficients
coef(model1)

# Store coefficients
b1 <- coef(model1)

cat("\nAdditive Model Coefficients:\n")
print(round(b1, 3))

# R-squared
r_squared_model1 <- summary(model1)$r.squared

cat(
  "\nAdditive Model R-squared:",
  round(r_squared_model1, 3),
  "\n"
)


# ------------------------------------------------------------------------------
# PART D - Interaction Model
# ------------------------------------------------------------------------------

# Interaction model:
# price = beta0 + beta1(carat) + cut effects
#         + interaction effects + error

model2 <- lm(
  price ~ carat * cut,
  data = diamonds
)

# Full regression output
summary(model2)

# Regression coefficients
coef(model2)

# R-squared
r_squared_model2 <- summary(model2)$r.squared

cat(
  "\nInteraction Model R-squared:",
  round(r_squared_model2, 3),
  "\n"
)


# ------------------------------------------------------------------------------
# Interaction Coefficients and P-values
# ------------------------------------------------------------------------------

interaction_names <- c(
  "carat:cutPremium",
  "carat:cutVery Good",
  "carat:cutGood",
  "carat:cutFair"
)

interaction_results <- summary(model2)$coefficients[
  interaction_names,
  c("Estimate", "Pr(>|t|)")
]

cat("\nInteraction Coefficients and P-values:\n")
print(round(interaction_results, 6))


# ------------------------------------------------------------------------------
# Incremental F-test / ANOVA
# ------------------------------------------------------------------------------

model_comparison <- anova(
  model1,
  model2
)

cat("\nIncremental F-test / ANOVA:\n")
print(model_comparison)

# Extract F-statistic and p-value
F_value <- model_comparison$F[2]
p_value <- model_comparison$Pr[2]

cat(
  "\nF-statistic =",
  round(F_value, 3),
  "\n"
)

cat(
  "p-value =",
  format.pval(p_value, digits = 3),
  "\n"
)


# ------------------------------------------------------------------------------
# PART E - Visualization
# ------------------------------------------------------------------------------

# Create figures folder if it does not already exist
dir.create(
  "figures",
  showWarnings = FALSE
)

# Price versus carat, with separate regression lines by cut
p <- ggplot(
  diamonds,
  aes(
    x = carat,
    y = price,
    color = cut
  )
) +
  geom_point(
    alpha = 0.30
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE
  ) +
  labs(
    title = "Diamond Price versus Carat by Cut",
    x = "Carat",
    y = "Price (US Dollars)",
    color = "Cut"
  ) +
  theme_minimal()

# Display graph
print(p)

# Save graph
ggsave(
  filename = "figures/diamond_price_vs_carat_by_cut.png",
  plot = p,
  width = 9,
  height = 6,
  dpi = 300
)

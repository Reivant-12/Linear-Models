
# ================================================================
# APM1205 — SUMMATIVE ASSESSMENT 1: COMPREHENSIVE LINEAR MODELING
# Dataset: Seoul Bike Sharing Demand
# Environment: Google Colab / Python
#
# Generates:
#   data/SeoulBikeData.csv
#   code/SA1_Linear_Models.py
#   output/figures/
#   output/tables/
#   report/SA1_Linear_Models_Report.pdf
#   README.md
#   APM1205-SA1.zip
# ================================================================

import os
import sys
import re
import html
import shutil
import zipfile
import urllib.request
import subprocess
import warnings
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

# -------------------- 1. INSTALL LIBRARIES ----------------------

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pandas", "numpy", "scipy", "statsmodels",
    "matplotlib", "seaborn", "reportlab", "patsy"
])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.diagnostic import (
    het_breuschpagan, linear_reset
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm
from patsy import dmatrix
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
    Table, TableStyle, PageBreak, KeepTogether
)

SEED = 1205
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# -------------------- 2. CREATE REPOSITORY STRUCTURE ------------

ROOT = Path("/content/APM1205-SA1")
DATA = ROOT / "data"
CODE = ROOT / "code"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"
REPORT = ROOT / "report"

for folder in [DATA, CODE, FIGURES, TABLES, REPORT]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA / "SeoulBikeData.csv"
PDF_FILE = REPORT / "SA1_Linear_Models_Report.pdf"

# -------------------- 3. LOAD THE DATASET -----------------------

def acquire_dataset():
    """Use an existing dataset, download the UCI archive, or upload CSV."""
    if DATA_FILE.exists():
        return

    # Also accept a CSV already placed in the Colab working directory.
    candidates = [
        Path("/content/SeoulBikeData.csv"),
        Path("/content/SeoulBikeData (1).csv"),
        Path("SeoulBikeData.csv")
    ]

    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, DATA_FILE)
            return

    # Attempt to download the UCI dataset.
    url = (
        "https://archive.ics.uci.edu/static/public/560/"
        "seoul+bike+sharing+demand.zip"
    )
    archive = ROOT / "seoul_bike_dataset.zip"

    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            csv_members = [
                n for n in zf.namelist()
                if n.lower().endswith(".csv")
                and "seoul" in Path(n).name.lower()
            ]
            if not csv_members:
                csv_members = [
                    n for n in zf.namelist()
                    if n.lower().endswith(".csv")
                ]
            if not csv_members:
                raise FileNotFoundError("No CSV found in UCI archive.")

            member = csv_members[0]
            with zf.open(member) as src, open(DATA_FILE, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print("Dataset downloaded from UCI.")
        return

    except Exception as exc:
        print("Automatic download failed:", str(exc))

    # Fallback for Colab.
    from google.colab import files
    print("Upload your SeoulBikeData.csv file.")
    uploaded = files.upload()

    csv_names = [
        name for name in uploaded
        if name.lower().endswith(".csv")
    ]
    if not csv_names:
        raise FileNotFoundError(
            "No CSV uploaded. Upload SeoulBikeData.csv and run again."
        )

    shutil.copy2(csv_names[0], DATA_FILE)


acquire_dataset()

def read_seoul_csv(path):
    """Read the commonly distributed semicolon-separated UCI CSV."""
    errors = []
    for encoding in ["utf-8-sig", "unicode_escape", "latin1"]:
        for sep in [";", ","]:
            try:
                candidate = pd.read_csv(
                    path, sep=sep, encoding=encoding
                )
                if candidate.shape[1] > 3:
                    candidate.columns = [
                        str(c).strip().replace("\ufeff", "")
                        for c in candidate.columns
                    ]
                    return candidate
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError("Unable to read dataset. " + "; ".join(errors[-3:]))


raw = read_seoul_csv(DATA_FILE)

# Normalize source headers into predictable Python names.
def normalize_name(name):
    name = str(name).strip().lower()
    name = name.replace("°", "")
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

df = raw.copy()
df.columns = [normalize_name(c) for c in df.columns]

# Map known source-column variations.
ALIASES = {
    "rented_bike_count": [
        "rented_bike_count", "rented_bike", "bike_count",
        "rental_count", "rented_bike_count_count"
    ],
    "temperature": [
        "temperature", "temperature_c", "temperature_celsius"
    ],
    "humidity": ["humidity", "humidity_percent"],
    "wind_speed": ["wind_speed", "wind_speed_m_s"],
    "visibility": ["visibility", "visibility_10m"],
    "dew_point_temperature": [
        "dew_point_temperature", "dew_point_temperature_c"
    ],
    "solar_radiation": [
        "solar_radiation", "solar_radiation_mj_m2"
    ],
    "rainfall": ["rainfall", "rainfall_mm"],
    "snowfall": ["snowfall", "snowfall_cm"],
    "seasons": ["seasons", "season"],
    "holiday": ["holiday"],
    "functioning_day": ["functioning_day", "functioningday"],
    "weather": ["weather"],
    "hour": ["hour"],
    "date": ["date"]
}

rename_map = {}
for standard, options in ALIASES.items():
    for option in options:
        if option in df.columns:
            rename_map[option] = standard
            break

df = df.rename(columns=rename_map)

if "rented_bike_count" not in df.columns:
    raise ValueError(
        "Could not identify the rental-count column. "
        f"Columns found: {list(raw.columns)}"
    )

# Ensure essential columns are numeric where applicable.
numeric_candidates = [
    "rented_bike_count", "temperature", "humidity", "wind_speed",
    "visibility", "dew_point_temperature", "solar_radiation",
    "rainfall", "snowfall", "hour"
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Preserve a clean copy of the original data for the report.
original_rows = len(df)
original_columns = len(df.columns)
missing_before = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

# Remove rows without the response and remove exact duplicates.
df = df.drop_duplicates().copy()
df = df.dropna(subset=["rented_bike_count"]).copy()

# Use categorical data types for categorical predictors.
for col in ["seasons", "holiday", "functioning_day", "weather"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

# Add a date-order index for a time-order diagnostic.
if "date" in df.columns and "hour" in df.columns:
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

if len(df) < 30:
    raise ValueError("Too few usable observations for this analysis.")

df.to_csv(DATA_FILE, index=False)

print("Dataset loaded successfully.")
print("Rows before cleaning:", original_rows)
print("Rows after cleaning:", len(df))
print("Columns:", list(df.columns))
print("Missing cells before cleaning:", missing_before)
print("Duplicate rows found:", duplicate_rows)

# -------------------- 4. DEFINE THE RESEARCH PROBLEM ------------

RESEARCH_QUESTION = (
    "How are weather conditions, time of day, season, holidays, "
    "and bike-service availability associated with hourly bike rental "
    "demand in Seoul?"
)

HYPOTHESES = [
    "H1: Mean rental demand differs with temperature.",
    "H2: At least one selected quantitative predictor is associated "
    "with rental demand after adjusting for other predictors.",
    "H3: Mean rental demand differs across seasons.",
    "H4: The association between season and rental demand depends "
    "on whether the bike service is functioning."
]

# -------------------- 5. SELECT AVAILABLE VARIABLES --------------

quantitative = [
    c for c in [
        "temperature", "humidity", "wind_speed", "visibility",
        "dew_point_temperature", "solar_radiation", "rainfall",
        "snowfall", "hour"
    ]
    if c in df.columns and df[c].nunique(dropna=True) > 1
]

categorical = [
    c for c in ["seasons", "holiday", "functioning_day", "weather"]
    if c in df.columns and df[c].nunique(dropna=True) > 1
]

if not quantitative:
    raise ValueError("No usable quantitative predictors were identified.")

# Use available variables, retaining at least three predictors where possible.
main_quant = [
    c for c in [
        "temperature", "humidity", "wind_speed",
        "solar_radiation", "rainfall", "snowfall", "hour",
        "visibility", "dew_point_temperature"
    ]
    if c in quantitative
]

main_cat = [
    c for c in ["seasons", "holiday", "functioning_day", "weather"]
    if c in categorical
]

# Complete cases for the modeling variables.
model_variables = ["rented_bike_count"] + main_quant + main_cat
model_df = df[model_variables].dropna().copy()

if len(model_df) < 30:
    raise ValueError(
        "Too few complete observations for the selected model. "
        "Inspect missing values and column types."
    )

# -------------------- 6. SAVE DATA PREPARATION TABLES ------------

data_info = pd.DataFrame({
    "Variable": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Count": [int(df[c].isna().sum()) for c in df.columns],
    "Missing_Percent": [
        round(100 * df[c].isna().mean(), 3) for c in df.columns
    ],
    "Unique_Values": [int(df[c].nunique(dropna=True)) for c in df.columns]
})
data_info.to_csv(TABLES / "data_quality_summary.csv", index=False)

cleaning_summary = pd.DataFrame({
    "Measure": [
        "Original rows", "Rows after duplicate removal and response check",
        "Duplicate rows detected", "Missing cells before cleaning",
        "Rows used in the main complete-case model"
    ],
    "Value": [
        original_rows, len(df), duplicate_rows, missing_before,
        len(model_df)
    ]
})
cleaning_summary.to_csv(TABLES / "cleaning_summary.csv", index=False)

# -------------------- 7. DESCRIPTIVE STATISTICS ------------------

desc_cols = ["rented_bike_count"] + main_quant
descriptive = model_df[desc_cols].describe().T
descriptive["missing_in_cleaned_data"] = [
    int(df[c].isna().sum()) for c in desc_cols
]
descriptive.to_csv(TABLES / "descriptive_statistics.csv")

categorical_tables = {}
for col in main_cat:
    counts = (
        model_df[col].value_counts(dropna=False)
        .rename_axis(col).reset_index(name="Count")
    )
    counts["Percent"] = 100 * counts["Count"] / counts["Count"].sum()
    categorical_tables[col] = counts
    counts.to_csv(TABLES / f"frequency_{col}.csv", index=False)

# -------------------- 8. EXPLORATORY DATA ANALYSIS ---------------

# Figure 1: response distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["rented_bike_count"], kde=True, bins=40)
plt.title("Distribution of Hourly Bike Rentals")
plt.xlabel("Rented Bike Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIGURES / "01_rental_distribution.png", dpi=180)
plt.close()

# Figure 2: rental count against temperature
if "temperature" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, x="temperature", y="rented_bike_count",
        scatter_kws={"alpha": 0.18, "s": 12},
        line_kws={"color": "red"}
    )
    plt.title("Hourly Rentals vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Rented Bike Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "02_rentals_vs_temperature.png", dpi=180)
    plt.close()

# Figure 3: correlation heatmap
corr_cols = [
    c for c in desc_cols
    if pd.api.types.is_numeric_dtype(df[c])
]
corr = df[corr_cols].corr(numeric_only=True)
corr.to_csv(TABLES / "correlation_matrix.csv")

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Quantitative Variables")
plt.tight_layout()
plt.savefig(FIGURES / "03_correlation_heatmap.png", dpi=180)
plt.close()

# Figure 4: rental count by season
if "seasons" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="seasons", y="rented_bike_count")
    plt.title("Bike Rentals by Season")
    plt.xlabel("Season")
    plt.ylabel("Rented Bike Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "04_rentals_by_season.png", dpi=180)
    plt.close()

# Figure 5: rental count by hour
if "hour" in df.columns:
    hourly = df.groupby("hour", observed=False)["rented_bike_count"].mean()
    hourly.to_csv(TABLES / "mean_rentals_by_hour.csv")
    plt.figure(figsize=(8, 5))
    hourly.plot(marker="o")
    plt.title("Mean Bike Rentals by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Rented Bike Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "05_rentals_by_hour.png", dpi=180)
    plt.close()

# Figure 6: rentals by season and functioning day
if "seasons" in df.columns and "functioning_day" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=df, x="seasons", y="rented_bike_count",
        hue="functioning_day", errorbar=("ci", 95), dodge=0.15
    )
    plt.title("Season and Functioning-Day Interaction")
    plt.xlabel("Season")
    plt.ylabel("Mean Rented Bike Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "06_season_functioning_interaction.png", dpi=180)
    plt.close()

# -------------------- 9. SIMPLE LINEAR REGRESSION ----------------

simple_predictor = (
    "temperature" if "temperature" in quantitative else quantitative[0]
)
simple_df = df[["rented_bike_count", simple_predictor]].dropna().copy()

simple_formula = f"rented_bike_count ~ {simple_predictor}"
simple_model = smf.ols(simple_formula, data=simple_df).fit()

with open(TABLES / "simple_regression_summary.txt", "w") as f:
    f.write(simple_model.summary().as_text())

simple_coeff = pd.DataFrame({
    "Term": simple_model.params.index,
    "Estimate": simple_model.params.values,
    "Std_Error": simple_model.bse.values,
    "t_value": simple_model.tvalues.values,
    "p_value": simple_model.pvalues.values,
    "CI_Lower": simple_model.conf_int()[0].values,
    "CI_Upper": simple_model.conf_int()[1].values
})
simple_coeff.to_csv(TABLES / "simple_regression_coefficients.csv", index=False)

plt.figure(figsize=(8, 5))
sns.regplot(
    data=simple_df, x=simple_predictor, y="rented_bike_count",
    scatter_kws={"alpha": 0.2, "s": 12},
    line_kws={"color": "red"}
)
plt.title(f"Simple Linear Regression: Rentals vs {simple_predictor}")
plt.xlabel(simple_predictor.replace("_", " ").title())
plt.ylabel("Rented Bike Count")
plt.tight_layout()
plt.savefig(FIGURES / "07_simple_regression.png", dpi=180)
plt.close()

# -------------------- 10. MULTIPLE LINEAR REGRESSION --------------

def formula_for(response, quantitative_vars, categorical_vars):
    terms = list(quantitative_vars)
    terms += [f"C({c})" for c in categorical_vars]
    if not terms:
        raise ValueError("No explanatory variables available.")
    return response + " ~ " + " + ".join(terms)

multiple_formula = formula_for(
    "rented_bike_count", main_quant, main_cat
)
multiple_model = smf.ols(multiple_formula, data=model_df).fit()

with open(TABLES / "multiple_regression_summary.txt", "w") as f:
    f.write(multiple_model.summary().as_text())

def coefficient_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        "Term": model.params.index,
        "Estimate": model.params.values,
        "Std_Error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "CI_Lower": ci[0].values,
        "CI_Upper": ci[1].values
    })

multiple_coeff = coefficient_table(multiple_model)
multiple_coeff.to_csv(TABLES / "multiple_regression_coefficients.csv", index=False)

fit_stats = pd.DataFrame([{
    "Model": "Multiple linear regression",
    "N": int(multiple_model.nobs),
    "R_squared": multiple_model.rsquared,
    "Adjusted_R_squared": multiple_model.rsquared_adj,
    "F_statistic": multiple_model.fvalue,
    "F_p_value": multiple_model.f_pvalue,
    "AIC": multiple_model.aic,
    "BIC": multiple_model.bic
}])
fit_stats.to_csv(TABLES / "model_fit_statistics.csv", index=False)

# -------------------- 11. HYPOTHESIS TESTS AND CONFIDENCE INTERVALS

# The coefficient table contains individual t-tests and 95% confidence intervals.
# Overall model F-test is recorded in model_fit_statistics.csv.

# Nested-model test for the categorical season predictor, if available.
season_test_result = None
if "seasons" in main_cat:
    reduced_cats = [c for c in main_cat if c != "seasons"]
    reduced_formula = formula_for(
        "rented_bike_count", main_quant, reduced_cats
    )
    reduced_model = smf.ols(reduced_formula, data=model_df).fit()
    season_test_result = multiple_model.compare_f_test(reduced_model)

    pd.DataFrame([{
        "Test": "Joint F-test for adding Season",
        "F_statistic": season_test_result[0],
        "p_value": season_test_result[1],
        "df_difference": season_test_result[2]
    }]).to_csv(TABLES / "season_partial_f_test.csv", index=False)

# Practical scale: compare coefficient magnitude to the response SD.
response_sd = float(model_df["rented_bike_count"].std())
practical = multiple_coeff.copy()
practical["Absolute_Effect_in_Response_SD"] = (
    practical["Estimate"].abs() / response_sd
)
practical.to_csv(TABLES / "coefficient_practical_scale.csv", index=False)

# -------------------- 12. MATRIX FORM AND LEAST SQUARES ------------

# Demonstrate y = X beta + epsilon on a manageable sample.
matrix_vars = [
    c for c in ["temperature", "humidity", "hour"]
    if c in main_quant
]
matrix_cats = [
    c for c in ["seasons", "functioning_day"]
    if c in main_cat
]
matrix_vars = matrix_vars + matrix_cats
matrix_data = model_df[["rented_bike_count"] + matrix_vars].dropna().copy()
matrix_data = matrix_data.sample(
    n=min(300, len(matrix_data)), random_state=SEED
)

matrix_formula = formula_for(
    "rented_bike_count",
    [c for c in matrix_vars if c in main_quant],
    [c for c in matrix_vars if c in main_cat]
)
y = matrix_data["rented_bike_count"].to_numpy(dtype=float)
X_df = dmatrix(
    matrix_formula.split("~", 1)[1],
    data=matrix_data,
    return_type="dataframe"
)
X = np.asarray(X_df, dtype=float)

# Explicit normal-equation demonstration; pseudoinverse fallback handles
# a potentially singular matrix safely.
XtX = X.T @ X
Xty = X.T @ y
try:
    beta_hat = np.linalg.solve(XtX, Xty)
    matrix_method = "Inverse/solve normal equations"
except np.linalg.LinAlgError:
    beta_hat = np.linalg.pinv(XtX) @ Xty
    matrix_method = "Pseudoinverse fallback"

beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)

matrix_results = pd.DataFrame({
    "Term": list(X_df.columns),
    "Beta_Normal_Equation": beta_hat,
    "Beta_Numpy_Lstsq": beta_lstsq
})
matrix_results.to_csv(TABLES / "matrix_least_squares_coefficients.csv", index=False)

with open(TABLES / "matrix_demonstration.txt", "w") as f:
    f.write("Matrix model: y = X beta + epsilon\n")
    f.write(f"Number of observations: {X.shape[0]}\n")
    f.write(f"Number of design-matrix columns: {X.shape[1]}\n")
    f.write(f"Method: {matrix_method}\n")
    f.write("Normal-equation formula: beta_hat = (X'X)^(-1)X'y\n")
    f.write("If X'X is singular, a pseudoinverse was used.\n")
    f.write(f"Maximum difference from numpy.linalg.lstsq: "
            f"{np.max(np.abs(beta_hat - beta_lstsq)):.12g}\n")

# -------------------- 13. DUMMY VARIABLES AND ANOVA ----------------

if "seasons" in main_cat:
    season_anova = smf.ols(
        "rented_bike_count ~ C(seasons)", data=model_df
    ).fit()
    season_anova_table = anova_lm(season_anova, typ=2)
    season_anova_table.to_csv(TABLES / "season_one_way_anova.csv")

    season_means = (
        model_df.groupby("seasons", observed=False)["rented_bike_count"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    season_means.to_csv(TABLES / "season_means.csv", index=False)

    # Reference category and dummy-variable coefficients.
    season_dummy_model = smf.ols(
        "rented_bike_count ~ C(seasons, Treatment(reference=0))",
        data=model_df
    ).fit()
    with open(TABLES / "season_dummy_variable_summary.txt", "w") as f:
        f.write(season_dummy_model.summary().as_text())

# -------------------- 14. TWO-WAY ANOVA AND INTERACTION ------------

interaction_model = None
interaction_test = None
interaction_formula = None

if "seasons" in main_cat and "functioning_day" in main_cat:
    interaction_formula = (
        "rented_bike_count ~ C(seasons) * C(functioning_day)"
    )
    additive_formula = (
        "rented_bike_count ~ C(seasons) + C(functioning_day)"
    )

    additive_model = smf.ols(additive_formula, data=model_df).fit()
    interaction_model = smf.ols(
        interaction_formula, data=model_df
    ).fit()

    interaction_test = interaction_model.compare_f_test(additive_model)

    pd.DataFrame([{
        "Test": "Season x Functioning Day interaction",
        "F_statistic": interaction_test[0],
        "p_value": interaction_test[1],
        "df_difference": interaction_test[2]
    }]).to_csv(TABLES / "interaction_partial_f_test.csv", index=False)

    anova_lm(interaction_model, typ=2).to_csv(
        TABLES / "two_way_anova_type2.csv"
    )

# -------------------- 15. DIAGNOSTICS AND VALIDATION ---------------

diagnostic_model = multiple_model
fitted = diagnostic_model.fittedvalues
residuals = diagnostic_model.resid
influence = diagnostic_model.get_influence()
standardized_resid = influence.resid_studentized_internal
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

# Residuals vs fitted
plt.figure(figsize=(8, 5))
plt.scatter(fitted, residuals, alpha=0.25, s=12)
plt.axhline(0, linestyle="--")
plt.title("Diagnostic Plot: Residuals vs Fitted")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig(FIGURES / "08_residuals_vs_fitted.png", dpi=180)
plt.close()

# Normal Q-Q plot
plt.figure(figsize=(7, 5))
sm.qqplot(residuals, line="45", fit=True)
plt.title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(FIGURES / "09_residual_qq_plot.png", dpi=180)
plt.close()

# Scale-location plot
plt.figure(figsize=(8, 5))
plt.scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.25, s=12)
plt.title("Scale-Location Plot")
plt.xlabel("Fitted Values")
plt.ylabel("Square Root of Absolute Standardized Residual")
plt.tight_layout()
plt.savefig(FIGURES / "10_scale_location.png", dpi=180)
plt.close()

# Cook's distance
plt.figure(figsize=(8, 5))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
plt.axhline(4 / len(cooks_d), linestyle="--")
plt.title("Cook's Distance by Observation")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.tight_layout()
plt.savefig(FIGURES / "11_cooks_distance.png", dpi=180)
plt.close()

# Statistical diagnostic tests
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
    residuals, diagnostic_model.model.exog
)

shapiro_sample = residuals
if len(shapiro_sample) > 5000:
    shapiro_sample = shapiro_sample.sample(
        5000, random_state=SEED
    )

shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)

try:
    reset_result = linear_reset(
        diagnostic_model, power=2, use_f=True
    )
    reset_stat = float(np.asarray(reset_result.fvalue).squeeze())
    reset_p = float(np.asarray(reset_result.pvalue).squeeze())
except Exception:
    reset_stat, reset_p = np.nan, np.nan

dw_stat = durbin_watson(residuals)

diagnostic_results = pd.DataFrame([
    {
        "Test": "Breusch-Pagan test for heteroscedasticity",
        "Statistic": bp_lm,
        "p_value": bp_lm_p,
        "Interpretation_rule": "Small p-value suggests nonconstant variance"
    },
    {
        "Test": "Shapiro-Wilk residual normality test",
        "Statistic": shapiro_stat,
        "p_value": shapiro_p,
        "Interpretation_rule": "Small p-value suggests non-normal residuals"
    },
    {
        "Test": "Ramsey RESET specification test",
        "Statistic": reset_stat,
        "p_value": reset_p,
        "Interpretation_rule": "Small p-value may indicate misspecification"
    },
    {
        "Test": "Durbin-Watson statistic",
        "Statistic": dw_stat,
        "p_value": np.nan,
        "Interpretation_rule": "Near 2 suggests limited first-order residual autocorrelation"
    }
])
diagnostic_results.to_csv(TABLES / "diagnostic_tests.csv", index=False)

influence_table = pd.DataFrame({
    "Observation_Index": np.arange(len(model_df)),
    "Fitted_Value": fitted.to_numpy(),
    "Residual": residuals.to_numpy(),
    "Studentized_Residual": standardized_resid,
    "Leverage": leverage,
    "Cooks_Distance": cooks_d
})
influence_table["Potential_Influential"] = (
    (influence_table["Cooks_Distance"] > 4 / len(model_df))
    | (influence_table["Studentized_Residual"].abs() > 3)
)
influence_table.sort_values(
    "Cooks_Distance", ascending=False
).head(30).to_csv(TABLES / "top_influential_observations.csv", index=False)

# -------------------- 16. LOG RESPONSE SENSITIVITY MODEL ------------

# Count outcomes are nonnegative; log1p is a sensitivity analysis,
# not an automatic replacement for the original model.
log_df = model_df.copy()
log_df["log_rented_bike_count"] = np.log1p(
    log_df["rented_bike_count"]
)
log_formula = formula_for(
    "log_rented_bike_count", main_quant, main_cat
)
log_model = smf.ols(log_formula, data=log_df).fit()

with open(TABLES / "log_response_sensitivity_summary.txt", "w") as f:
    f.write(log_model.summary().as_text())

pd.DataFrame([{
    "Model": "OLS on log1p(rented_bike_count)",
    "N": int(log_model.nobs),
    "R_squared_on_log_scale": log_model.rsquared,
    "Adjusted_R_squared_on_log_scale": log_model.rsquared_adj,
    "AIC_on_log_scale": log_model.aic,
    "Note": "Fit statistics are on the transformed response scale."
}]).to_csv(TABLES / "log_response_sensitivity_fit.csv", index=False)

# -------------------- 17. AUTOMATED INTERPRETATIONS ----------------

def fmt_num(x, digits=3):
    try:
        if pd.isna(x):
            return "not available"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def p_text(p):
    if pd.isna(p):
        return "not available"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

simple_slope = float(simple_model.params.iloc[1])
simple_p = float(simple_model.pvalues.iloc[1])
simple_r2 = float(simple_model.rsquared)

if simple_slope > 0:
    simple_direction = "positive"
elif simple_slope < 0:
    simple_direction = "negative"
else:
    simple_direction = "approximately zero"

simple_interpretation = (
    f"In the simple model, a one-unit increase in {simple_predictor} "
    f"is associated with an estimated change of {simple_slope:.3f} "
    f"rental units in the mean hourly rental count. The estimated "
    f"association is {simple_direction}; its coefficient test gives "
    f"{p_text(simple_p)}. The model R-squared is {simple_r2:.3f}, "
    f"meaning the model accounts for about {100*simple_r2:.1f}% of "
    f"the observed variation in rental counts in this sample."
)

significant_terms = multiple_coeff[
    (multiple_coeff["Term"] != "Intercept")
    & (multiple_coeff["p_value"] < 0.05)
]
if len(significant_terms):
    significant_text = "; ".join(
        f"{row.Term} (estimate {row.Estimate:.3f}, "
        f"{p_text(row.p_value)})"
        for row in significant_terms.head(8).itertuples()
    )
else:
    significant_text = (
        "No non-intercept coefficient met the 0.05 significance "
        "threshold in this fitted model."
    )

fit_interpretation = (
    f"The multiple regression uses {int(multiple_model.nobs)} "
    f"complete observations. R-squared = {multiple_model.rsquared:.3f}; "
    f"adjusted R-squared = {multiple_model.rsquared_adj:.3f}; "
    f"overall F = {multiple_model.fvalue:.3f}; "
    f"overall-model {p_text(multiple_model.f_pvalue)}. "
    f"These statistics describe in-sample fit and do not establish "
    f"causation or guarantee out-of-sample predictive accuracy."
)

if season_test_result is not None:
    season_interpretation = (
        f"After controlling for the selected predictors, adding the "
        f"season factor gives F = {season_test_result[0]:.3f}, "
        f"{p_text(season_test_result[1])}. "
        + (
            "There is evidence that season contributes to the model."
            if season_test_result[1] < 0.05
            else "The test does not provide strong evidence that season "
                 "adds explanatory value at the 0.05 level."
        )
    )
else:
    season_interpretation = (
        "A season-specific test was not run because a usable season "
        "column was unavailable."
    )

if interaction_test is not None:
    interaction_interpretation = (
        f"The season-by-functioning-day interaction test gives "
        f"F = {interaction_test[0]:.3f}, "
        f"{p_text(interaction_test[1])}. "
        + (
            "This supports different seasonal patterns by functioning-day "
            "status in this model."
            if interaction_test[1] < 0.05
            else "The test does not provide strong evidence of an interaction "
                 "at the 0.05 level."
        )
    )
else:
    interaction_interpretation = (
        "The interaction analysis was not available because both season "
        "and functioning-day categorical variables were not present."
    )

bp_interpretation = (
    "The Breusch-Pagan test suggests possible nonconstant residual variance."
    if bp_lm_p < 0.05
    else "The Breusch-Pagan test does not show strong evidence of "
         "nonconstant residual variance."
)
shapiro_interpretation = (
    "The Shapiro-Wilk test suggests residuals depart from normality."
    if shapiro_p < 0.05
    else "The Shapiro-Wilk test does not show strong evidence against "
         "normal residuals."
)
reset_interpretation = (
    "The RESET test suggests possible model specification issues."
    if pd.notna(reset_p) and reset_p < 0.05
    else "The RESET test does not show strong evidence of functional-form "
         "misspecification."
    if pd.notna(reset_p)
    else "The RESET test could not be evaluated."
)

# -------------------- 18. BUILD THE PDF REPORT ---------------------

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=14
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=12, spaceAfter=8
))
styles.add(ParagraphStyle(
    name="SubHeading", parent=styles["Heading2"],
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"],
    fontSize=8.5, leading=12, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="TinyBody", parent=styles["BodyText"],
    fontSize=7, leading=9
))

def P(text, style="SmallBody"):
    return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])

def add_table(story, dataframe, title, max_rows=18):
    story.append(P(title, "SubHeading"))
    if dataframe is None or dataframe.empty:
        story.append(P("No results available for this table."))
        return

    shown = dataframe.head(max_rows).copy()
    headers = [str(c) for c in shown.columns]
    data = [headers]
    for _, row in shown.iterrows():
        data.append([
            "" if pd.isna(v) else (
                f"{v:.4g}" if isinstance(v, (float, np.floating))
                else str(v)
            )
            for v in row.values
        ])

    usable_width = A4[0] - 1.2 * inch
    col_width = usable_width / max(1, len(headers))
    t = Table(
        data, repeatRows=1,
        colWidths=[col_width] * len(headers),
        hAlign="LEFT"
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("LEADING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    if len(dataframe) > max_rows:
        story.append(P(
            f"Only the first {max_rows} rows are shown here; "
            "see the complete CSV table in output/tables/."
        ))

def add_figure(story, filename, caption, width=6.4*inch):
    path = FIGURES / filename
    if path.exists():
        story.append(P(caption, "SubHeading"))
        img = PDFImage(str(path))
        scale = min(width / img.imageWidth, 4.5*inch / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)
        story.append(Spacer(1, 8))

story = []

story.append(P(
    "APM1205 — Summative Assessment 1",
    "ReportTitle"
))
story.append(P(
    "Comprehensive Linear Modeling Analysis",
    "ReportTitle"
))
story.append(P(f"Date generated: {date.today().isoformat()}"))
story.append(P("Student name: ______________________________"))
story.append(P("Course and section: _________________________"))
story.append(Spacer(1, 12))

story.append(P("Executive Summary", "SectionHeading"))
story.append(P(
    "This report investigates hourly Seoul bike-sharing demand using "
    "simple and multiple linear regression, statistical inference, "
    "matrix least squares, categorical predictors, ANOVA, interaction "
    "analysis, and model diagnostics. The findings describe associations "
    "in the observed dataset and should not be interpreted as causal effects."
))
story.append(P("Research question: " + RESEARCH_QUESTION))

story.append(P("A. Research Question and Data Preparation", "SectionHeading"))
story.append(P("Real-world problem"))
story.append(P(
    "Bike-sharing operators need to understand how rental demand varies "
    "with weather, time, season, holidays, and service availability. "
    "This analysis estimates those relationships to support descriptive "
    "planning and demand-management decisions."
))
story.append(P("Research hypotheses"))
for h in HYPOTHESES:
    story.append(P("• " + h))
story.append(P("Dataset and measurement scales"))
story.append(P(
    f"The data file contains {original_rows} rows before cleaning and "
    f"{original_columns} columns. The analysis uses hourly rental count "
    "as the quantitative response. Temperature, humidity, wind speed, "
    "visibility, dew-point temperature, solar radiation, rainfall, "
    "snowfall, and hour are quantitative predictors when available. "
    "Season, holiday, functioning day, and weather are treated as "
    "categorical predictors when present. Rental count is a quantitative "
    "count variable; hour is numeric but represents a cyclical time-of-day "
    "measure; categorical labels are nominal factors."
))
story.append(P(
    f"Cleaning removed {duplicate_rows} duplicate rows and excluded rows "
    "with missing rental counts. Missing values in predictors were handled "
    "by complete-case analysis for the relevant model. The complete-case "
    f"multiple model used {len(model_df)} observations. See the data-quality "
    "and cleaning CSV tables for details."
))
add_table(story, cleaning_summary, "Table 1. Data cleaning summary")
add_table(story, data_info, "Table 2. Variable types and missing values", 18)

story.append(P("B. Exploratory Data Analysis", "SectionHeading"))
story.append(P(
    "Descriptive statistics summarize the response and quantitative "
    "predictors. Graphs are used to inspect skewness, potential outliers, "
    "relationships, and differences between groups. A linear model may be "
    "sensitive to skewed counts, nonlinear relationships, and unequal "
    "variance; these issues are revisited in the diagnostics section."
))
add_table(story, descriptive.reset_index().rename(columns={"index": "Variable"}),
          "Table 3. Descriptive statistics")
add_figure(
    story, "01_rental_distribution.png",
    "Figure 1. Distribution of hourly rental counts."
)
add_figure(
    story, "02_rentals_vs_temperature.png",
    "Figure 2. Rental counts versus temperature with a fitted straight line."
)
add_figure(
    story, "03_correlation_heatmap.png",
    "Figure 3. Correlations among quantitative variables."
)
add_figure(
    story, "04_rentals_by_season.png",
    "Figure 4. Rental-count distributions across seasons."
)
add_figure(
    story, "05_rentals_by_hour.png",
    "Figure 5. Mean rentals by hour, where hour is available."
)

story.append(P("C. Simple Linear Regression", "SectionHeading"))
story.append(P(
    f"The fitted model is: rental count = intercept + slope × "
    f"{simple_predictor} + error. The estimated slope is "
    f"{simple_slope:.3f}, with {p_text(simple_p)}. "
    + simple_interpretation
))
story.append(P(
    "The intercept is the fitted mean rental count when the predictor is "
    "zero; this may not be practically meaningful if zero is outside or "
    "near the edge of the observed predictor range. R-squared describes "
    "the proportion of in-sample response variation accounted for by this "
    "one-predictor model, not a causal effect."
))
add_table(story, simple_coeff, "Table 4. Simple regression coefficients")
add_figure(
    story, "07_simple_regression.png",
    "Figure 6. Simple linear regression fit."
)

story.append(P("D. Multiple Linear Regression", "SectionHeading"))
story.append(P(
    "The multiple regression includes the available quantitative "
    "predictors and categorical factors listed in the model formula. "
    "Weather, time, and seasonal variables are included because they are "
    "plausibly related to rental demand and are explicitly suggested in "
    "the assessment. Coefficients are adjusted associations conditional "
    "on the other terms in the model."
))
story.append(P(f"Model formula: {multiple_formula}"))
story.append(P(fit_interpretation))
story.append(P(
    "Selected non-intercept terms with p-values below 0.05: "
    + significant_text
))
story.append(P(
    "Statistical significance is not the same as practical importance. "
    "A small coefficient can be statistically significant in a large "
    "dataset, while a larger estimate can be uncertain. Interpret "
    "coefficient size in rental-count units and in the context of the "
    "predictor's measurement scale."
))
add_table(story, fit_stats, "Table 5. Multiple-model fit statistics")
add_table(
    story,
    multiple_coeff.sort_values("p_value"),
    "Table 6. Multiple regression coefficients and 95% confidence intervals",
    25
)

story.append(P("E. Statistical Inference", "SectionHeading"))
story.append(P(
    "Individual coefficient tests use H0: beta_j = 0 against a two-sided "
    "alternative. The coefficient table reports estimates, standard errors, "
    "t-statistics, p-values, and 95% confidence intervals. A confidence "
    "interval gives a range of parameter values compatible with the fitted "
    "model and its assumptions; it does not mean that the fixed parameter "
    "has a 95% probability of lying in this particular interval."
))
story.append(P(
    "The overall F-test evaluates whether the explanatory terms jointly "
    "improve on an intercept-only model under the classical linear-model "
    "assumptions. The p-value alone does not indicate whether an effect "
    "is large enough to matter operationally."
))
add_table(
    story,
    multiple_coeff.sort_values("p_value"),
    "Table 7. Inference for regression coefficients",
    25
)

story.append(P("F. Matrix Form and Least-Squares Theory", "SectionHeading"))
story.append(P("The selected matrix model is y = Xβ + ε."))
story.append(P(
    "Here, y is the vector of observed rental counts; X is the design "
    "matrix containing a column of ones for the intercept, quantitative "
    "predictors, and dummy-variable columns for categorical levels; β is "
    "the vector of unknown regression coefficients; and ε is the vector "
    "of unobserved errors."
))
story.append(P(
    "Least-squares estimation minimizes the sum of squared residuals. "
    "When X has full column rank, the normal-equation solution is "
    "β̂ = (XᵀX)⁻¹Xᵀy. The code computes this on a reproducible sample and "
    "checks it against numpy.linalg.lstsq. If XᵀX is singular, it uses a "
    "pseudoinverse fallback and records that choice."
))
add_table(
    story, matrix_results,
    "Table 8. Matrix least-squares coefficient demonstration", 18
)
story.append(P(
    "The matrix solution and statsmodels regression use the same least-"
    "squares principle. Differences can arise when the sample, formula, "
    "reference categories, or design-matrix columns differ."
))

story.append(P("G. Dummy Variables and ANOVA", "SectionHeading"))
story.append(P(
    "Categorical predictors are represented through indicator (dummy) "
    "columns. One level is used as the reference category; each reported "
    "factor coefficient represents the fitted difference from that "
    "reference level, holding other model terms fixed. The default "
    "treatment coding in statsmodels uses a reference level determined "
    "by the factor's category ordering unless explicitly set."
))
story.append(P(season_interpretation))
if "seasons" in main_cat:
    story.append(P(
        "The one-way ANOVA tests whether all seasonal mean rental counts "
        "are equal. In a linear-model framework, ANOVA and dummy-variable "
        "regression are equivalent ways to represent group mean differences; "
        "the overall factor F-test tests the joint contribution of the "
        "season indicators."
    ))
    add_table(
        story, season_means,
        "Table 9. Rental count summaries by season"
    )
    add_table(
        story, season_anova_table.reset_index(),
        "Table 10. One-way ANOVA for season"
    )

story.append(P("H. Two-Way ANOVA and Interaction", "SectionHeading"))
story.append(P(
    "The interaction model includes season, functioning-day status, and "
    "their product term. The interaction F-test compares this model with "
    "an additive model. A significant interaction indicates that the "
    "seasonal difference is not constant across functioning-day groups "
    "on the model's rental-count scale. A nonsignificant test is not proof "
    "that all group effects are identical."
))
story.append(P(interaction_interpretation))
add_figure(
    story, "06_season_functioning_interaction.png",
    "Figure 7. Mean rental counts by season and functioning-day status."
)
if interaction_test is not None:
    add_table(
        story,
        pd.DataFrame([{
            "F_statistic": interaction_test[0],
            "p_value": interaction_test[1],
            "df_difference": interaction_test[2]
        }]),
        "Table 11. Interaction partial F-test"
    )

story.append(P("I. Model Diagnostics and Validation", "SectionHeading"))
story.append(P(
    "The residuals-versus-fitted plot is used to inspect nonlinearity and "
    "changing variance; the Q-Q plot assesses residual normality; the "
    "scale-location plot examines residual spread; and Cook's distance "
    "helps flag potentially influential observations. These plots should "
    "be interpreted together rather than as automatic pass/fail rules."
))
story.append(P(bp_interpretation))
story.append(P(shapiro_interpretation))
story.append(P(reset_interpretation))
story.append(P(
    f"The Durbin-Watson statistic is {dw_stat:.3f}. Values near 2 are "
    "consistent with little first-order residual autocorrelation, but "
    "the statistic is not a complete test of independence. Because these "
    "data are hourly and time-ordered, residual dependence is a meaningful "
    "concern. Ordinary regression p-values may be unreliable if temporal "
    "dependence is substantial."
))
story.append(P(
    f"{int(influence_table['Potential_Influential'].sum())} observations "
    "were flagged by the exploratory rule of Cook's distance greater than "
    "4/n or absolute internally studentized residual greater than 3. "
    "Flagged cases should be investigated rather than automatically deleted."
))
add_table(story, diagnostic_results, "Table 12. Diagnostic tests")
add_figure(
    story, "08_residuals_vs_fitted.png",
    "Figure 8. Residuals versus fitted values."
)
add_figure(
    story, "09_residual_qq_plot.png",
    "Figure 9. Normal Q-Q plot."
)
add_figure(
    story, "10_scale_location.png",
    "Figure 10. Scale-location plot."
)
add_figure(
    story, "11_cooks_distance.png",
    "Figure 11. Cook's distance."
)

story.append(P("Transformation and alternative specification"))
story.append(P(
    "Rental counts can be skewed and may show nonconstant variance. "
    "The log1p response model is provided as a sensitivity analysis, "
    "not as an automatic final model. Its R-squared and AIC are on a "
    "different response scale and should not be directly compared with "
    "the original-response model as if they were identical criteria. "
    "If prediction is the main goal, a Poisson or negative-binomial "
    "model and time-aware validation may be useful extensions, although "
    "they are outside the main linear-model assessment."
))

story.append(P("J. Final Model and Statistical Communication", "SectionHeading"))
story.append(P(
    "The selected primary model is the multiple linear regression using "
    "the available weather, time, and categorical predictors. It is chosen "
    "because it directly addresses the research question, adjusts for "
    "multiple factors, and supports coefficient inference, confidence "
    "intervals, dummy-variable interpretation, and model diagnostics. "
    "Its limitations are the linear functional form, possible count-data "
    "skewness, potential temporal dependence, and possible heteroscedasticity. "
    "The diagnostic findings should be considered before using its "
    "p-values for strong decisions."
))
story.append(P(
    "In nontechnical terms, the analysis estimates how bike rentals tend "
    "to change alongside weather and operating conditions. The regression "
    "results identify which associations are most evident in this dataset, "
    "while the confidence intervals and diagnostics indicate how much "
    "uncertainty remains. These observational data can inform planning, "
    "but they do not prove that changing a weather or operating condition "
    "would cause a particular change in rentals."
))
story.append(P(
    "Conclusion: The research question is addressed by estimating the "
    "simple and multiple regression relationships and testing seasonal "
    "and functioning-day differences. The actual direction, size, and "
    "strength of each association should be reported using the generated "
    "tables, confidence intervals, and diagnostic results rather than "
    "assuming any particular result in advance."
))

story.append(P("References", "SectionHeading"))
story.append(P(
    "UCI Machine Learning Repository. Seoul Bike Sharing Demand Dataset. "
    "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand"
))
story.append(P(
    "Seoul Bike Sharing Demand dataset distributed as SeoulBikeData.csv. "
    "Python libraries used: pandas, NumPy, SciPy, statsmodels, patsy, "
    "Matplotlib, seaborn, and ReportLab."
))

doc = SimpleDocTemplate(
    str(PDF_FILE),
    pagesize=A4,
    rightMargin=0.6*inch,
    leftMargin=0.6*inch,
    topMargin=0.6*inch,
    bottomMargin=0.6*inch,
    title="APM1205 SA1 Linear Models Report",
    author="APM1205 Student"
)
doc.build(story)

# -------------------- 19. CREATE README ---------------------------

readme = f"""# APM1205-SA1: Comprehensive Linear Modeling Analysis

## Student information
- Student name: [Enter your name]
- Course and section: [Enter your section]
- Assessment: APM1205 Summative Assessment 1

## Dataset
Seoul Bike Sharing Demand Dataset, from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

The response variable is Rented Bike Count. Available weather, time,
seasonal, holiday, and functioning-day variables are used as predictors.

## Research question
{RESEARCH_QUESTION}

## Software
Python, intended for Google Colab. The script uses pandas, NumPy, SciPy,
statsmodels, patsy, Matplotlib, seaborn, and ReportLab.

## Analysis included
- Data cleaning, missing-value and variable-type checks
- Descriptive statistics and exploratory data analysis
- Simple and multiple linear regression
- Coefficient tests, confidence intervals, R-squared, adjusted R-squared,
  and overall F-test
- Matrix least-squares demonstration
- Dummy variables and one-way ANOVA
- Two-way ANOVA and season-by-functioning-day interaction
- Residual diagnostics, influence checks, and log-response sensitivity model
- PDF report with tables, figures, interpretations, and references

## Repository structure
```text
APM1205-SA1/
├── README.md
├── data/
│   └── SeoulBikeData.csv
├── code/
│   └── SA1_Linear_Models.py
├── output/
│   ├── figures/
│   └── tables/
└── report/
    └── SA1_Linear_Models_Report.pdf
```

## Reproduce the analysis
1. Download or clone this repository.
2. Install Python 3.10+ and the required packages:
   `pip install pandas numpy scipy statsmodels patsy matplotlib seaborn reportlab`
3. Run `python code/SA1_Linear_Models.py` from the repository root.
4. The script regenerates the tables, figures, and PDF report.
5. In Colab, the same workflow can be run in a notebook cell. If the dataset
   is missing, the script attempts to download the UCI archive and then
   offers a CSV upload fallback.

## Reproducibility notes
- Random seed: {SEED}
- Models use complete cases for their respective variables.
- The matrix demonstration uses a fixed random seed.
- Diagnostic tests and plots are exploratory and should be interpreted
  alongside the regression assumptions and the time-ordered nature of data.
- Fill in student name and section before submission.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

# Save the running Colab cell as the reproducible Python script.
# In IPython/Colab, the most recent input normally contains this cell's code.
try:
    ip = get_ipython()
    cell_source = ip.history_manager.input_hist_raw[-1]
    if cell_source and len(cell_source) > 1000:
        (CODE / "SA1_Linear_Models.py").write_text(
            cell_source, encoding="utf-8"
        )
    else:
        raise ValueError("Current cell source unavailable")
except Exception:
    (CODE / "SA1_Linear_Models.py").write_text(
        "# See the Google Colab notebook cell used to generate this project.\n"
        "# Paste the complete analysis cell here to create a standalone script.\n",
        encoding="utf-8"
    )

# -------------------- 20. PACKAGE THE REPOSITORY ------------------

zip_path = Path("/content/APM1205-SA1.zip")
if zip_path.exists():
    zip_path.unlink()

shutil.make_archive(
    str(zip_path.with_suffix("")),
    "zip",
    root_dir=ROOT.parent,
    base_dir=ROOT.name
)

print("\n" + "="*65)
print("ANALYSIS COMPLETE")
print("="*65)
print("Dataset:", DATA_FILE)
print("PDF report:", PDF_FILE)
print("README:", ROOT / "README.md")
print("Python script:", CODE / "SA1_Linear_Models.py")
print("Project ZIP:", zip_path)
print("\nModel formula:")
print(multiple_formula)
print("\nModel fit:")
print(fit_stats.to_string(index=False))
print("\nSimple regression:")
print(simple_interpretation)
print("\nMultiple regression:")
print(fit_interpretation)
print("\nDiagnostics:")
print(bp_interpretation)
print(shapiro_interpretation)
print(reset_interpretation)

# Download the packaged project from Colab.
from google.colab import files
files.download(str(zip_path))
# APM1205-SA1: Comprehensive Linear Modeling Analysis

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
How are weather conditions, time of day, season, holidays, and bike-service availability associated with hourly bike rental demand in Seoul?

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
- Random seed: 1205
- Models use complete cases for their respective variables.
- The matrix demonstration uses a fixed random seed.
- Diagnostic tests and plots are exploratory and should be interpreted
  alongside the regression assumptions and the time-ordered nature of data.
- Fill in student name and section before submission.

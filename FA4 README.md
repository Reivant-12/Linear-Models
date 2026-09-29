# APM1205 Formative Assessment 4 – Dummy-Variable Regression

This folder contains a complete, reproducible solution structure for APM1205 Formative Assessment 4.

## Files

- `FA4_Dummy_Regression.Rmd` – R Markdown report containing Parts A–E, calculations, interpretations, tables, regression equations, ANOVA/incremental F-test, and the required visualization.
- `FA4_Dummy_Regression.R` – standalone R script containing the complete analysis code.
- `FA4_Dummy_Regression.pdf` – completed assessment report.
- `figures/diamond_price_vs_carat_by_cut.png` – required Price-versus-Carat visualization by cut.

## Dataset

The analysis uses the actual `diamonds` dataset supplied with `ggplot2`. No simulated observations are generated.

## Reference category

`Ideal` is used as the reference category, following the example in the assessment. The four dummy variables are:

- `D1_Fair`
- `D2_Good`
- `D3_VeryGood`
- `D4_Premium`

A diamond in the Ideal category has all four dummy variables equal to zero.

## Models

### Additive model

```r
model1 <- lm(price ~ carat + cut, data = diamonds)
```

This model permits different intercepts by cut but uses a common carat slope.

### Interaction model

```r
model2 <- lm(price ~ carat * cut, data = diamonds)
```

This model allows the carat slope to vary by cut.

### Incremental F-test

```r
anova(model1, model2)
```

The model comparison evaluates whether adding the four carat-by-cut interaction terms significantly improves the model.

## Reproducing the report

1. Open `FA4_Dummy_Regression.Rmd` in RStudio.
2. Install `ggplot2`, `knitr`, and a LaTeX distribution such as TinyTeX if they are not already installed.
3. Set the working directory to this folder, or open the `.Rmd` directly from this folder.
4. Knit the R Markdown document to PDF.
5. The code automatically creates/updates the `figures/` directory and saves the required figure.

The report follows the assessment requirements: Parts A–E, R commands, relevant output, dummy-variable coding table, regression equations, ANOVA/incremental F-test, visualization, and statistical interpretation.

# Airbnb Price Prediction (Chicago)

Predicting nightly prices of Airbnb listings in Chicago with regression models in R.

This university project was built for a Kaggle regression challenge in the course **Machine Learning and Artificial Intelligence** (2023). The goal is to predict listing prices accurately and to understand which listing characteristics drive price, so the platform can plan its strategy in Chicago.

The project compares eight regression approaches, from penalized linear models to tree ensembles, all scored by **RMSE** on the Kaggle leaderboard (lower is better). The best single model was a **tuned random forest (RMSE 249.13)**. The best overall submission was a **weighted ensemble of random forest, lasso, ridge, and a regression tree (RMSE 246.81)**.

All work is in one R Markdown file: [`Group_07_code.Rmd`](Group_07_code.Rmd).

---

## Table of contents

- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key findings](#key-findings)
- [How to run](#how-to-run)
- [Repository structure](#repository-structure)
- [Known limitations and possible improvements](#known-limitations-and-possible-improvements)
- [License](#license)

---

## Dataset

The data comes from the course's Kaggle competition and is **not included in this repository**. It contains `train.csv` (listings with prices) and `test.csv` (listings without prices, used for the submission).

| Column | Description |
|---|---|
| `ID` | Listing identifier |
| `name`, `host_id`, `host_name` | Listing and host details (dropped before modeling) |
| `neighbourhood` | Chicago community area |
| `latitude`, `longitude` | Listing coordinates |
| `room_type` | Entire home/apt, private room, shared room, or hotel room |
| `price` | **Target**: nightly price |
| `minimum_nights` | Minimum stay length |
| `number_of_reviews` | Total number of reviews |
| `last_review` | Date of the most recent review (dropped) |
| `reviews_per_month` | Average reviews per month |
| `calculated_host_listings_count` | Number of listings the host has |
| `availability_365` | Days per year the listing is available |

---

## Approach

```
train.csv ──► Cleaning ──► EDA & maps ──► Linear regression screening ──► Dummy encoding ──► Lasso-based feature selection ──► 8 tuned models ──► Weighted ensemble ──► Kaggle submission
```

### 1. Cleaning
- Missing `reviews_per_month` values (listings with no reviews) are set to 0.
- `last_review` is dropped rather than filled in, so listings without reviews (often new ones) can stay in the data.
- `price` is **log-transformed**, since its distribution is heavily right-skewed. Predictions are converted back with `exp()` before submission.

### 2. Exploratory analysis
- Distributions of all numeric variables and a `ggpairs` overview
- A **choropleth map** of log price over Chicago's community areas, plus a coordinate scatter plot colored by room type
- Price by room type (box plots and ridgeline densities)
- A correlation heatmap with significance testing. Every numeric variable except `minimum_nights` is significantly correlated with price.

### 3. Feature screening and selection
- OLS regressions with and without the non-significant variables, including a `room_type × calculated_host_listings_count` interaction
- Train and test are combined before one-hot encoding `neighbourhood` and `room_type`, so both have the same dummy columns even when some neighborhoods appear in only one of them.
- A first lasso / ridge pass on all features. **Lasso variable importance** was then used to keep only the strongest predictors. This left 13 features:
  - `latitude`, `longitude`, `reviews_per_month`, `availability_365`, `calculated_host_listings_count`, `minimum_nights`
  - Neighborhood dummies: South Shore, Loop, West Town, Near North Side
  - Room type dummies: entire home/apt, private room, shared room

### 4. Models
Built with **tidymodels** (recipes + workflows). Hyperparameters were tuned with 10-fold cross-validation.

| Model | Engine | Tuned parameters |
|---|---|---|
| Polynomial regression | `lm` | polynomial degree (1–10, one-standard-error rule) |
| Regression splines | `lm` + `step_bs` | fixed knots on 5 numeric predictors |
| Ridge regression | `glmnet` | penalty (log-scale grid, 50 levels) |
| Lasso regression | `glmnet` | penalty (log-scale grid, 50 levels) |
| Regression tree | `rpart` | cost complexity |
| Bagging | `randomForest` (`mtry` = all predictors) | — |
| Random forest | `randomForest`, 500 trees | `mtry`, `min_n` (random grid, then a refined regular grid) |
| Boosting | `xgboost`, 500 trees | depth, `min_n`, loss reduction, sample size, `mtry`, learning rate (Latin hypercube, 30 candidates) |

### 5. Ensemble
The final submission is a **weighted average** of the predictions, giving the random forest four times the weight of each other model:

```
price = (lasso + ridge + regression_tree + 4 × random_forest) / 7
```

---

## Results

Kaggle RMSE on the test set (lower is better), as recorded in the report:

| Model | Kaggle RMSE |
|---|---|
| **Ensemble (lasso + ridge + tree + 4× random forest)** | **246.81** |
| Random forest (tuned) | 249.13 |
| Lasso (selected features) | 252.59 |
| Ridge (selected features) | 252.76 |
| Polynomial regression | 253.16 |
| XGBoost (tuned) | 253.41 |
| Regression splines | 254.4 |
| Bagging | 260.46 |
| Regression tree | 278.82 |

For comparison, the first ridge and lasso runs on **all** features scored 253.16 and 253.13. Feature selection gave both a small improvement.

> The report notes that these scores came from the original submissions. Re-running the code can give slightly different results.

---

## Key findings

- **Location matters most.** Higher prices cluster in Chicago's main high-traffic areas, and the only neighborhood indicators that lasso kept were the Loop, Near North Side, West Town, and South Shore. Latitude and longitude relate to price in a non-linear way, which is why linear regression was skipped in favor of polynomial, spline, and tree models.
- **Room type is a strong driver.** Entire homes/apartments and hotel rooms are priced higher than private rooms, and shared rooms are cheapest. Hotel rooms show the widest price range.
- **Review counts tell you little about price.** `number_of_reviews` had essentially no effect and was dropped, and `minimum_nights` was not significantly correlated with price.
- **Ensembles help.** Averaging models with different error patterns beat every individual model. Tree ensembles (random forest) did best on their own.

---

## How to run

1. Get `train.csv` and `test.csv` from the course's Kaggle competition and place them next to the `.Rmd` file.
2. For the neighborhood map, download the **Chicago community areas boundary shapefile** (City of Chicago data portal) into a folder named `./chicago/`. The code reads the layer `geo_export_36a240a2-354e-42ad-8e86-a38f0515bf66`, so update the layer name if yours is different.
3. Install the R packages:

```r
install.packages(c(
  "tidymodels", "ISLR", "GGally", "broom", "dotwhisker", "performance",
  "funModeling", "sjPlot", "tidyverse", "vip", "sf", "plotly", "corrplot",
  "ggcorrplot", "ggridges", "xgboost", "glmnet", "randomForest", "rpart",
  "doParallel"
))
```

4. Change the `write.csv(...)` output paths, which are hard-coded absolute paths, to a local `submission/` folder.
5. Knit the file in RStudio, or run:

```bash
Rscript -e 'rmarkdown::render("Group_07_code.Rmd")'
```

> The random forest and XGBoost grid searches use `doParallel` and can take a while.

The report also embeds an image (`img/IMG_2968.jpg`, a reference map of Chicago) that is not in this repository. Remove that line or add your own image.

---

## Repository structure

```
Price-Prediction/
├── Group_07_code.Rmd   # Full report: cleaning, EDA, models, ensemble, results
├── README.md
└── LICENSE             # GNU GPL v3
```

---

## Known limitations and possible improvements

- **Leftover placeholders.** Sections 4.1 and 5 still say "the XX model" where the chosen model should be named. It should read *random forest*, or the weighted ensemble.
- **In-sample RMSE comparison.** Section 4.1 compares models by RMSE on the training data. That favors flexible models such as random forest, which can nearly memorize the data. It is also on the log scale, so it can't be compared with the Kaggle scores. Cross-validated RMSE (already computed during tuning) would be a fairer basis for the comparison.
- **Fragile train/test split.** After combining the two sets for dummy encoding, rows are split again with `price != 0`. Because price is already log-transformed, a training listing priced at exactly 1 would have log price 0 and land in the test set. An explicit `is_train` flag column would be safer.
- **Choropleth averages.** The map joins every listing to its community polygon without aggregating, so each area is drawn many times and its color shows the last listing drawn. Computing the mean or median price per area before the join would show the real pattern.
- **Back-transformation bias.** `exp()` of a log-scale prediction estimates the median rather than the mean price, which tends to under-predict when RMSE is measured on the raw scale. A smearing correction could reduce RMSE.
- **Small inconsistency.** The splines section mentions 253.4, while the results section lists 254.4.
- **Hard-coded paths** in the `write.csv` calls and the missing shapefile and image make the report hard to knit on another machine.
- **Unused or duplicated libraries** (e.g. `ISLR`, `plotly`, repeated `tidyr` / `funModeling`) could be removed.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for details.

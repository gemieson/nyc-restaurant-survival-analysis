# NYC Restaurant Shutdown Risk: Survival Analysis

Modeling the time until a restaurant is shut down by NYC health inspectors, using survival analysis on roughly 296K public DOHMH inspection records. The project estimates when and for whom an enforcement shutdown is most likely, benchmarks a classical model against a machine learning one, validates its predictions, and serves the result as a risk scoring API.

> **Scope:** This models enforcement risk among currently active restaurants, not business failure. NYC removes permanently closed restaurants from the data, so the model cannot and does not predict restaurants going out of business.

## Why survival analysis

The outcome is a time to event with heavy censoring. Only about 3 percent of restaurants are ever shut down, and the rest are still operating when the data ends. A yes or no classifier would mislabel every censored restaurant as a clean negative and throw away the timing. Survival analysis uses "survived at least this long" correctly and models when a shutdown happens, not just whether it does.

## Data and reshaping

Source: [DOHMH NYC Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) (NYC Open Data). The raw data is one row per violation. Reshaping collapses it to one survival record per restaurant with time_zero (first operating inspection), duration_days, event (ever closed by DOHMH), and leakage safe baseline features (score, health and admin violations, cuisine, borough).

Final sample: **26,534 restaurants, 845 shutdown events (3.2 percent)**.

## Analysis and findings

The analysis moves through a deliberate progression: describe the raw patterns, model them, validate the model, benchmark it, and validate its predictions.

### 1. Descriptive: Kaplan-Meier curves

![Overall survival curve](figures/km_overall.png)

The overall curve starts at 1.0 and drops only about 4 percentage points before flattening. That small drop is the roughly 3 percent event rate, and the flat top is the 96 percent of restaurants that are never shut down. Almost all of the decline happens in the first 1,000 days. The flat tail past that point is an artifact of the 3 year data window, not evidence that older restaurants are safe.

![Survival by borough](figures/km_borough.png)

Splitting by borough shows clear separation: Brooklyn restaurants are shut down fastest, Manhattan and Staten Island slowest. A pairwise log rank test confirms the differences are statistically real (p below 0.001 for several pairs). This is unadjusted, so it does not yet tell us whether borough itself matters or whether it is standing in for something else.

![Survival by cuisine](figures/km_cuisine.png)

Splitting by cuisine produces the sharpest separation in the project. Caribbean and Chinese restaurants drop well below Italian and Donuts. These raw curves motivate the modeling but are still unadjusted, so the Cox model is needed to isolate the cuisine effect from confounds like borough and inspection score.

### 2. Inferential: Cox proportional hazards

Fitting all features together isolates each effect. Cuisine is the dominant predictor, significant even after controlling for inspection score and borough.

| Feature | Hazard ratio | Significance |
|---|---|---|
| Cuisine: Caribbean | 5.2 | p<0.005 |
| Cuisine: Chinese | 4.0 | p<0.005 |
| Cuisine: Bakery and Desserts | 3.2 | p<0.005 |
| Cuisine: Latin American | 2.5 | p<0.005 |
| Cuisine: Italian | 1.0 | not significant |
| Initial health violations (per violation) | 1.17 | p<0.005 |
| Initial score (per point) | 1.02 | p<0.005 |
| Initial admin violations (per violation) | 1.09 | not significant |
| Borough: Brooklyn | 1.33 | p=0.03 |

Two findings stand out beyond "bad inspections predict shutdowns":

**Borough effects are absorbed by cuisine.** Boroughs differed sharply in the raw curves, but once cuisine enters the model most borough terms lose significance. The geographic pattern was largely explained by which cuisines cluster in which boroughs. This is a confounding insight, not a null result.

**Administrative violations do not predict shutdown, but food safety violations do.** This validates the decision to split violations into two features rather than lumping them together.

Model concordance is about 0.75.

### 3. Rigor: proportional hazards assumption check

![Schoenfeld residuals for initial score](figures/schoenfeld_init_score.png)

A Schoenfeld residual test flagged initial score as violating proportional hazards. The residual plot shows a clear downward trend over time, meaning the effect of a bad opening score is strong early and fades for restaurants that survive longer. This was fixed by binning score at the official A, B, C grade thresholds and stratifying on it, which lets each grade band have its own baseline risk trajectory. All borough and cuisine terms passed the assumption test. Two borderline flags (health violations and Caribbean) showed flat residuals and were retained, with their hazard ratios read as time averaged effects.

### 4. Benchmark: Random Survival Forest

A Random Survival Forest makes no proportional hazards assumption and can capture non-linear effects. Compared fairly on a held out test set, Cox scored 0.737 and the forest scored 0.738. The flexible model found no meaningful non-linear structure that the interpretable Cox model missed, so Cox was kept. Knowing when not to reach for the more complex model is the point of the comparison.

### 5. Validation: calibration

![Calibration curve](figures/calibration.png)

Calibration checks whether predicted probabilities match reality: when the model says 20 percent, does it happen about 20 percent of the time. The smoothed calibration curve tracks the diagonal closely (ICI 0.005, E50 0.004), with only mild overestimation at the high risk tail where data is sparse. The model is well calibrated, so its outputs can be used as absolute risk estimates, not just relative rankings. This is what justifies serving real probabilities from the API.

## Deployment

The final model is served as a FastAPI risk scoring app. A POST request to /predict takes a restaurant's features and returns a risk score plus shutdown probabilities at 6 month, 1 year, and 2 year horizons.

## Limitations

**Survivorship bias.** Permanently closed restaurants are removed by NYC, so observed closures skew toward temporary shutdowns. Results describe enforcement risk among active restaurants.

**Three year window.** time_zero is the first observed inspection, not the true opening date (left truncation), and durations are bounded by the observation window.

**Association, not causation.** Cuisine hazard ratios likely proxy for unmeasured factors such as restaurant size, chain status, neighborhood, and enforcement patterns.

**Stack:** Python · pandas · lifelines · scikit-survival · matplotlib · FastAPI

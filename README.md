# Sleep Score Prediction Model

## Overview
This project aims to develop a machine learning regression model to predict a sleep score, ranging from 0 to 100, using data collected from the Garmin platform. The sleep score is a key indicator of sleep quality, and the goal of this model is to provide accurate predictions based on the available data.

## Data Collection
The data for this project were gathered via the Garmin website, which provides detailed information about various sleep-related metrics. The dataset includes features such as:

- Sleep duration
- Bedtime
- Heart rate variability

## Model Construction
The model is built using various regression techniques and evaluated through K-Fold Cross Validation. This ensures that the model's performance is robust and generalizable to unseen data. The metrics reported (such as RMSE, MAE, and R²) are averaged across the K folds to provide a reliable assessment of model performance.     
The model presented in the notebook is the most robust, with the least multicollinearity between variables, and the Random Forest and Gradient Boosting parameters entered are those from a random grid search CV associated with this model.

## Key Features
- K-Fold Cross Validation: Each model is evaluated using this technique, ensuring the metrics represent a mean across the folds, reducing variance in performance evaluations.
- Multiple Regression Models: Various machine learning algorithms are tested and compared to determine the most accurate model for predicting sleep scores.
- Comprehensive Evaluation: Performance metrics are reported to provide insights into the accuracy and robustness of the model.

## Results
The final model provides a reliable prediction of the sleep score, with performance metrics averaged across all K folds. The model can be further by incorporating more features from the Garmin dataset.

## Limitations and Future Work
The biggest limitation of this model is the amount of data available (400 rows, one per day), and consequently the problems this creates for learning.
- Exploring other sources of sleep-related data to enhance model accuracy.

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests to enhance the model or improve the codebase.

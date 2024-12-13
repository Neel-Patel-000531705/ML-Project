# ML-Project
# Project: The prediction of the flight ticket prices

## Overview
This work mainly involves providing statistical and machine learning tools for analyzing and making predictions for flight ticket prices. It consists of variable including price, duration, days till the trip, and date originated variables like, day of the week, quarter. 

The project employs multiple models, including:
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Linear Regression

## Dataset
The dataset used is stored in a CSV file: `flight_ticket_dataset.csv`. It contains the following features:

- **price**: Ticket price
- **duration**: Flight duration
- **days_left**: Days left until departure
- **Date**: The date of the flight
- Derived features:
  - **Year**: Extracted from the Date
  - **Month**: pulled out from the Date (which is the target variable for classification)
  - **Day_of_Week**: Day of the week (0-6)
  - **Quarter**: Fiscal quarter
  - **price_duration**: Moderating; Price by duration
  - **price_days_left**: The interaction term of price and number of days left

## Project Structure
1. **Data Preprocessing**:
   Missing value management – removing rows where key input variables and the output contain NaN.
   - Extracting features for the dates for the model for deeper analysis.
   - Feature selection: These are `price`, `duration`, `days_left`, `price_duration`, `price_days_left`, `Day_of_Week`, and `Quarter`.

2. **Model Training**:
   - **Random Forest Classifier**: tuned with the hyperparameters as `n_estimators”, ‘max_depth”, ‘min_samples_split”.
   - **K-Nearest Neighbors**:
     - Standardized features.
     Hyperparameter tuning we sign up GridSearchCV to search and discover the suitable `k’ from among many potential values of this parameter.
   - **Linear Regression**: Assessed in R-squared, mean absolute error, and mean squared error.

3. **Model Evaluation**:
   Accuracy values for the models of classification – Random Forest, K nearest neighbors.
   Linear Regression Original Data Set RÂ² MAE MSE Synthetic data set1 0.996 0.204 0.41554 Synthetic data set2 0.991 0.196 0.42164
- **RÂ² Scores**: Training -86% ~ | Test -63% ~

### K-Nearest Neighbors:
- **Best k**: 13~
- **Training Accuracy**: 49% ~
- **Test Accuracy**: 39% ~

### Linear Regression:
- **RÂ² Scores**: Training -94% ~ | Test -94% ~
- **MAE**: Training -67% ~ | Test -67% ~
- **MSE**: Training -67% ~ | Test -67% ~

## Installation and Setup
1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the root directory as `flight_ticket_dataset.csv`.
4. Run the notebook `Neel_Patel_ML_Project.ipynb` in a Jupyter Notebook environment.

## Usage
- Modify the dataset path in the notebook if required.
- Execute the cells to preprocess data, train models, and evaluate results.
- Tune hyperparameters and compare results.

## Future Improvements
- Integrate additional features for richer predictions.
- Use advanced models like XGBoost or Neural Networks.
- Implement cross-validation for more robust evaluation.



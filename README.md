# Car Price Prediction using Multiple Linear Regression

## Overview

This project builds a Multiple Linear Regression model to predict car prices based on features such as mileage, engine volume, year, brand, and other characteristics.

The workflow covers data cleaning, outlier removal, feature engineering, multicollinearity handling, model training, and evaluation.

---

## Objective

* Predict car prices accurately using structured data
* Apply end-to-end data preprocessing techniques
* Improve model performance through transformations and feature selection

---

## Dataset

The dataset contains information about used cars, including:

* Price (target variable)
* Mileage
* Engine Volume
* Year
* Brand
* Body Type
* Engine Type
* Registration status

---

## Project Workflow

### 1. Data Cleaning

* Removed missing values using:

  ```python
  data.dropna(axis=0)
  ```
* Ensured only complete rows are used for modeling

---

### 2. Outlier Removal

* Removed extreme values using quantiles:

  * Price → top 1% removed
  * Mileage → top 1% removed
  * Engine Volume → capped at 6.5
  * Year → removed very old cars (bottom 1%)

Result: More reliable and realistic dataset

---

### 3. Data Transformation

* Applied log transformation to price:

  ```python
  log_price = np.log(Price)
  ```
* Reason:

  * Reduces skewness
  * Improves linear relationship

---

### 4. Feature Engineering

* Converted categorical variables using one-hot encoding:

  ```python
  pd.get_dummies(..., drop_first=True)
  ```
* Avoided dummy variable trap using `drop_first=True`

---

### 5. Multicollinearity Check

* Used Variance Inflation Factor (VIF) to detect correlated features
* Removed highly correlated variable (`Year`)

---

### 6. Feature Scaling

* Standardized features using:

  ```python
  StandardScaler()
  ```
* Ensures all variables are on the same scale

---

### 7. Train-Test Split

* Training set: 80%
* Testing set: 20%

  ```python
  train_test_split(test_size=0.2)
  ```

---

### 8. Model Training

* Used Linear Regression:

  ```python
  LinearRegression().fit(X_train, y_train)
  ```

---

### 9. Model Evaluation

#### Training Performance

* Scatter plot: Actual vs Predicted
* Residual distribution (should be normal)

#### Metrics

* R² Score measures model accuracy

---

### 10. Predictions

* Predictions made on test data:

  ```python
  y_hat_test = model.predict(X_test)
  ```

* Converted back from log scale:

  ```python
  np.exp(predictions)
  ```

---

### 11. Error Analysis

* Created performance table with:

  * Prediction
  * Actual value
  * Residual
  * Percentage difference

Helps identify:

* Best predictions
* Worst predictions

---

## Key Insights

* Log transformation significantly improved model performance
* Removing outliers made relationships clearer
* Multicollinearity negatively affected the model and was resolved using VIF
* Feature scaling ensured stable training

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels

---

## How to Run

1. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```

2. Run the notebook/script

3. Observe:

   * Data cleaning steps
   * Visualizations
   * Model training
   * Predictions

---

## Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Perform cross-validation
* Hyperparameter tuning
* Feature importance analysis

---

## Conclusion

This project demonstrates a complete data science workflow:

* From raw data to cleaned dataset
* From preprocessing to model training
* From predictions to evaluation

It highlights how proper data handling and transformations can significantly improve model accuracy.

---

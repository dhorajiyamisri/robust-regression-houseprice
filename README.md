<div align="center">

# 🏠 Robust Regression
### **House Price Prediction using Machine Learning**

<p>
  <strong>📊 Regression Analysis • 🔁 Cross-Validation • 🌲 Ensemble Learning • 📐 Model Evaluation</strong>
</p>

<br>

<img 
  src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=22&duration=2800&pause=900&color=10B981&center=true&vCenter=true&width=850&lines=House+Price+Prediction;Regression+Model+Comparison;Ridge+%7C+Lasso+%7C+Elastic+Net;Decision+Tree+%7C+Random+Forest;SVR+%7C+Cross-Validation+%7C+Evaluation" 
  alt="Typing Animation"
/>

<br>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=000000&color=3776AB">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=000000&color=150458">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white&labelColor=000000&color=013243">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=000000&color=F7931E">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=000000&color=F37626">
</p>

<br>

<a href="https://github.com/dhorajiyamisri/robust-regression-houseprice">
  <img src="https://img.shields.io/badge/VIEW%20PROJECT-10B981?style=for-the-badge&logo=github&logoColor=white">
</a>
&nbsp;
<a href="https://github.com/dhorajiyamisri/robust-regression-houseprice/blob/main/Robust_Regression_Engine.ipynb">
  <img src="https://img.shields.io/badge/OPEN%20NOTEBOOK-111827?style=for-the-badge&logo=jupyter&logoColor=white">
</a>

<br><br>

<p>
  <sub>✨ Built with Python • Scikit-Learn • Jupyter Notebook ✨</sub>
</p>

</div>

---

---

# 📌 Project Overview

**Robust Regression — House Price Prediction** is a Machine Learning project focused on predicting house prices using multiple regression techniques.

The project follows a complete regression workflow:

> **Dataset → Data Preparation → Feature Engineering → Regression Models → Cross-Validation → Model Evaluation → Model Comparison → Final Analysis**

Different model families are implemented and compared, including:

* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression
* Support Vector Regression (SVR)

The models are evaluated using **MAE, MSE, RMSE, and R² Score** to determine which approach provides the strongest predictive performance.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Understand the structure of a house price dataset.
* Identify input features and the target variable.
* Handle missing values and perform basic preprocessing.
* Convert date information into useful numerical features.
* Remove unnecessary identifier columns.
* Apply feature scaling where required.
* Implement Ridge and Lasso Regression.
* Understand the impact of regularization.
* Tune regularization parameters using Cross-Validation.
* Compare different Cross-Validation strategies.
* Implement Decision Tree Regression.
* Implement Random Forest Regression.
* Implement Linear and RBF Support Vector Regression.
* Tune SVR hyperparameters using Grid Search.
* Evaluate all models using standard regression metrics.
* Analyze overfitting and generalization.
* Select the best-performing regression model.

---

# 🏠 Problem Statement

House prices are influenced by several property-related factors such as:

* Property area
* Number of bedrooms
* Number of bathrooms
* Location quality
* Property age
* Distance from the city
* Nearby schools
* Nearby metro stations
* Crime rate
* Sale year
* Sale month

The objective of this project is to use these features to predict the **house price in INR** and determine which regression algorithm performs best on the given dataset.

Rather than relying on a single algorithm, multiple regression approaches are evaluated under the same dataset and testing setup.

---

# 📊 Dataset Information

The project uses a house-price dataset containing:

| Property        | Details           |
| --------------- | ----------------- |
| Total Records   | **3,800**         |
| Total Columns   | **12**            |
| Input Features  | **11 initially**  |
| Target Variable | `house_price_inr` |
| Train Samples   | **3,040**         |
| Test Samples    | **760**           |
| Test Size       | **20%**           |
| Random State    | **42**            |

The original dataset contains 12 columns. During preprocessing, `property_id` is removed because it is only an identifier, while `sale_date` is converted into `sale_year` and `sale_month`.

---

# 📋 Dataset Features

| Feature            | Description                         |
| ------------------ | ----------------------------------- |
| `property_id`      | Unique property identifier          |
| `sale_date`        | Property sale date                  |
| `area_sqft`        | Area of the property in square feet |
| `bedrooms`         | Number of bedrooms                  |
| `bathrooms`        | Number of bathrooms                 |
| `location_score`   | Location quality score              |
| `property_age`     | Age of the property                 |
| `distance_city_km` | Distance from city                  |
| `near_school`      | School proximity indicator          |
| `near_metro`       | Metro proximity indicator           |
| `crime_rate_index` | Crime-rate indicator                |
| `house_price_inr`  | 🎯 Target house price               |

---

# 🎯 Target Variable

The target variable is:

```python
house_price_inr
```

It represents the price of the property in Indian Rupees.

The notebook separates this column as `y`, while the remaining columns are initially used as input features `X`.

---

# 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Check

Missing values are checked before modelling.

Numeric columns are handled using the **median**, while non-numeric columns are handled using the **mode** where required.

### 2. Date Conversion

The `sale_date` column is converted into datetime format.

```python
X["sale_date"] = pd.to_datetime(X["sale_date"])
```

### 3. Feature Extraction

Two useful features are extracted:

```python
X["sale_year"] = X["sale_date"].dt.year
X["sale_month"] = X["sale_date"].dt.month
```

### 4. Remove Original Date

The original date column is removed after extracting useful information.

### 5. Remove Identifier

`property_id` is removed because it is an identifier rather than a predictive feature.

### 6. Feature Scaling

`StandardScaler` is used for continuous numerical features, particularly for the SVR workflow.

These preprocessing steps are implemented directly in the notebook.

---

# 🔄 Machine Learning Workflow

```text
                         Dataset
                            │
                            ▼
                  Data Understanding
                            │
                            ▼
                   Missing Value Check
                            │
                            ▼
                  Date Feature Extraction
                            │
                            ▼
                 Remove Unnecessary Columns
                            │
                            ▼
                    Train-Test Split
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      Regularized Models          Tree-Based Models
              │                           │
       ┌──────┴──────┐             ┌──────┴──────┐
       ▼             ▼             ▼             ▼
     Ridge         Lasso      Decision Tree  Random Forest
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                            │
                            ▼
                         SVR
                    Linear + RBF
                            │
                            ▼
                  Cross-Validation
                            │
                            ▼
                    Model Evaluation
                            │
                            ▼
                    Model Comparison
                            │
                            ▼
                    Overfitting Analysis
                            │
                            ▼
                    Final Model Selection
```

---

# 🧠 Machine Learning Models

## 1. Ridge Regression

Ridge Regression is a linear regression technique that uses **L2 regularization**.

In this project:

```python
Ridge(alpha=1.0)
```

is used initially.

Ridge regularization helps control large model coefficients and can reduce model complexity.

### Result

| Metric |           Score |
| ------ | --------------: |
| MAE    |       1,945,453 |
| MSE    | 6.449981 × 10¹² |
| RMSE   |       2,539,681 |
| R²     |    **0.919911** |

---

## 2. Lasso Regression

Lasso Regression uses **L1 regularization**.

It can shrink some feature coefficients toward zero, making it useful for controlling model complexity and potentially performing feature selection.

The notebook uses:

```python
Lasso(
    alpha=1.0,
    max_iter=10000
)
```

### Result

| Metric |           Score |
| ------ | --------------: |
| MAE    |       1,945,520 |
| MSE    | 6.450121 × 10¹² |
| RMSE   |       2,539,709 |
| R²     |    **0.919909** |

Ridge and Lasso produced almost identical performance on the test set.

---

# ⚙️ Regularization Parameter Tuning

Grid Search with 5-Fold Cross-Validation is used to test different Ridge `alpha` values:

```python
[0.01, 0.1, 1, 10, 100, 1000]
```

The best value obtained was:

```text
Best Alpha = 10
```

The corresponding reported cross-validation score was:

```text
-6221951335644.521
```

The negative value is expected because Scikit-Learn's `neg_mean_squared_error` scoring represents MSE as a negative score so that higher scoring remains better.

---

# 🔁 Cross-Validation Strategies

The project compares multiple validation strategies.

## K-Fold Cross-Validation

**Mean R²:**

```text
0.917447
```

## Stratified K-Fold

The continuous target is first divided into bins and then stratified.

**Mean R²:**

```text
0.917513
```

## Leave-One-Out Cross-Validation

The notebook evaluates LOOCV using MSE/RMSE because R² cannot be calculated with a single test sample in each iteration.

**RMSE:**

```text
2,501,086.87
```

## Time Series Split

**Mean R²:**

```text
0.917037
```

### Cross-Validation Comparison

| CV Strategy       |             Mean R² | Observation                  |
| ----------------- | ------------------: | ---------------------------- |
| K-Fold            |            0.917447 | Stable                       |
| Stratified K-Fold |            0.917513 | Slightly highest             |
| Time Series Split |            0.917037 | Similar performance          |
| Leave-One-Out     | RMSE = 2,501,086.87 | Evaluated using error metric |

The three R²-based strategies produced very similar results, suggesting relatively stable model performance across the tested splits.

---

# 🌳 Tree-Based Regression

## Decision Tree Regression

A controlled Decision Tree model is used to study nonlinear relationships while limiting model complexity.

### Result

| Metric |           Score |
| ------ | --------------: |
| MAE    |       2,155,391 |
| MSE    | 8.725474 × 10¹² |
| RMSE   |       2,953,891 |
| R²     |    **0.891656** |

---

# 🌲 Random Forest Regression

Random Forest combines multiple decision trees to create an ensemble model.

The project uses:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

### Result

| Metric |               Score |
| ------ | ------------------: |
| MAE    |       **1,757,755** |
| MSE    | **5.785958 × 10¹²** |
| RMSE   |       **2,405,402** |
| R²     |        **0.928156** |

Random Forest achieved the highest R² and the lowest MAE and RMSE among the evaluated models.

---

# 📐 Support Vector Regression

The project implements two SVR variants:

### Linear SVR

```python
SVR(
    kernel="linear",
    C=100
)
```

### RBF SVR

```python
SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)
```

The features are standardized before applying SVR.

---

# 🔍 SVR Hyperparameter Tuning

Grid Search with 5-Fold Cross-Validation is used for:

* Kernel
* C
* Gamma
* Epsilon

The best parameters found were:

```text
C = 100
epsilon = 0.1
gamma = scale
kernel = linear
```

The reported best cross-validation R² score was:

```text
0.075874
```

The final test-set results show that SVR performed considerably worse than Ridge, Lasso, Decision Tree, and Random Forest on this dataset.

---

# 📏 Evaluation Metrics

The project uses four major regression metrics.

| Metric       | Meaning                                       | Better Value |
| ------------ | --------------------------------------------- | ------------ |
| **MAE**      | Average absolute prediction error             | Lower        |
| **MSE**      | Average squared prediction error              | Lower        |
| **RMSE**     | Square root of MSE                            | Lower        |
| **R² Score** | Proportion of variance explained by the model | Higher       |

### Why Multiple Metrics?

A single metric may not provide a complete picture of model performance.

Therefore, this project compares models using:

```text
MAE + MSE + RMSE + R²
```

---

# 📊 Final Model Comparison

The notebook evaluates six model configurations on the same test set.

| Model             |              MSE |              MAE |             RMSE |     R² Score |
| ----------------- | ---------------: | ---------------: | ---------------: | -----------: |
| Ridge Regression  |     6.449981e+12 |     1.945453e+06 |     2.539681e+06 | **0.919911** |
| Lasso Regression  |     6.450121e+12 |     1.945520e+06 |     2.539709e+06 | **0.919909** |
| Decision Tree     |     8.725474e+12 |     2.155391e+06 |     2.953891e+06 | **0.891656** |
| **Random Forest** | **5.785958e+12** | **1.757755e+06** | **2.405402e+06** | **0.928156** |
| SVR Linear        |     7.261499e+13 |     6.620502e+06 |     8.521443e+06 |     0.098344 |
| SVR RBF           |     8.058569e+13 |     6.985495e+06 |     8.976954e+06 |    -0.000627 |

---

# 🏆 Best Performing Model

## Random Forest Regression

Based on the final test-set comparison, **Random Forest Regression** is the best-performing model in this project.

### Performance

```text
R² Score : 0.928156
MAE      : 1,757,755
RMSE     : 2,405,402
MSE      : 5.785958 × 10¹²
```

It achieved:

* Highest R² Score
* Lowest MAE
* Lowest RMSE
* Lowest MSE

among the models evaluated in the final comparison.

### Interpretation

The result suggests that the Random Forest model was better able to capture the relationships between the property features and house prices than the other tested approaches.

---

# 🔍 Overfitting & Generalization Analysis

The project compares training and testing R² scores.

| Model            | Training R² | Testing R² | Difference |
| ---------------- | ----------: | ---------: | ---------: |
| Ridge Regression |    0.917402 |   0.919911 |  -0.002509 |
| Lasso Regression |    0.917402 |   0.919909 |  -0.002507 |
| Decision Tree    |    0.962133 |   0.891656 |   0.070476 |
| Random Forest    |    0.989346 |   0.928156 |   0.061190 |

### Observation

The Decision Tree has a noticeable train-test gap of approximately **0.0705**, indicating overfitting.

Random Forest also has a train-test gap, but its testing R² is considerably higher than the Decision Tree's testing R².

Therefore, among the tree-based models, Random Forest provides better predictive performance and generalization.

---

# 📈 Project Results

The project demonstrates:

* Successful dataset preparation
* Feature and target identification
* Date-based feature extraction
* Feature scaling
* Regularized linear regression
* Multiple cross-validation strategies
* Tree-based regression
* Ensemble learning
* Support Vector Regression
* Hyperparameter tuning
* Model evaluation
* Overfitting analysis
* Final model selection

---

# 💻 Quick Code Preview

A simplified example of the Random Forest modelling approach used in the project:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv(
    "Advanced_Regression_HousePrice_Dataset_3800(Sheet1).csv"
)

# Prepare features and target
X = df.drop(columns=["house_price_inr"])
y = df["house_price_inr"]

# Convert date
X["sale_date"] = pd.to_datetime(X["sale_date"])
X["sale_year"] = X["sale_date"].dt.year
X["sale_month"] = X["sale_date"].dt.month

# Remove unnecessary columns
X = X.drop(columns=["sale_date", "property_id"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
```

---

# 📂 Repository Structure

```text
robust-regression-houseprice/
│
├── Advanced_Regression_HousePrice_Dataset_3800(Sheet1).csv
│
├── Robust_Regression_Engine.ipynb
│
├── theory_concepts.ipynb
│
├── README.md
│
└── docs/
    └── images/
        ├── dataset-overview.png
        ├── data-preprocessing.png
        ├── ridge-lasso-results.png
        ├── cross-validation-results.png
        ├── random-forest-results.png
        ├── model-comparison.png
        └── overfitting-analysis.png
```

> `docs/images/` is the recommended folder for the screenshots that should be added to the repository.

---

# 🛠️ Tech Stack

| Category                | Technologies                                                |
| ----------------------- | ----------------------------------------------------------- |
| Programming Language    | Python                                                      |
| Data Manipulation       | Pandas, NumPy                                               |
| Machine Learning        | Scikit-Learn                                                |
| Regression              | Ridge, Lasso, Decision Tree, Random Forest, SVR             |
| Validation              | K-Fold, Stratified K-Fold, Leave-One-Out, Time Series Split |
| Hyperparameter Tuning   | GridSearchCV                                                |
| Development Environment | Jupyter Notebook                                            |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/dhorajiyamisri/robust-regression-houseprice.git
```

## 2. Navigate to the Project

```bash
cd robust-regression-houseprice
```

## 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Robust_Regression_Engine.ipynb
```

---

# ▶️ How to Use

1. Clone the repository.
2. Install the required Python libraries.
3. Launch Jupyter Notebook.
4. Open `Robust_Regression_Engine.ipynb`.
5. Run the notebook cells sequentially.
6. Review preprocessing results.
7. Explore the implemented regression models.
8. Compare cross-validation strategies.
9. Review the final model comparison.
10. Analyze the overfitting results.

---

# 📘 Theory Documentation

The repository also contains a dedicated theory notebook:

**`theory_concepts.ipynb`**

It is intended to support the practical implementation with the underlying Machine Learning concepts and definitions.

> 📌 **Theory PDF:** Add the exported PDF version of the theory notebook to the repository before final submission if required by the assignment.

---

# 🎥 Project Demonstration

A 5–10 minute recorded demonstration can cover:

* Project introduction
* Dataset understanding
* Data preprocessing
* Ridge & Lasso Regression
* Cross-Validation
* Decision Tree
* Random Forest
* SVR
* Evaluation metrics
* Model comparison
* Overfitting analysis
* Final model selection

### 🎬 Video
Video Link: https://drive.google.com/file/d/10Ba9xd4vReKZPwmeXDCl_CXqYv4h0Zm2/view?usp=drive_link

> Replace the placeholder with the final recorded-video link before submission.

---

# 📝 Assignment Coverage

| Component                       | Project Coverage         |
| ------------------------------- | ------------------------ |
| Dataset Understanding           | ✅                        |
| Data Preprocessing              | ✅                        |
| Feature & Target Identification | ✅                        |
| Ridge Regression                | ✅                        |
| Lasso Regression                | ✅                        |
| Regularization                  | ✅                        |
| Cross-Validation                | ✅                        |
| Decision Tree Regression        | ✅                        |
| Random Forest Regression        | ✅                        |
| Support Vector Regression       | ✅                        |
| Hyperparameter Tuning           | ✅                        |
| Regression Metrics              | ✅                        |
| Model Comparison                | ✅                        |
| Overfitting Analysis            | ✅                        |
| Final Analysis                  | ✅                        |
| Theory Documentation            | ✅ Notebook               |
| Screenshots                     | 📌 Add to `docs/images/` |
| Theory PDF                      | 📌 To be added           |
| 5–10 Minute Video               | 📌 To be added           |

---

# 💡 Key Findings

### 1. Random Forest performed best

Random Forest achieved an R² of **0.928156**, the highest among the tested models.

### 2. Ridge and Lasso performed very similarly

Ridge achieved an R² of **0.919911**, while Lasso achieved **0.919909**.

### 3. Cross-validation results were stable

K-Fold, Stratified K-Fold, and Time Series Split produced R² values around **0.917**, indicating similar performance across these validation strategies.

### 4. Decision Tree showed overfitting

The Decision Tree had a training R² of **0.9621** and testing R² of **0.8917**, showing a noticeable train-test gap.

### 5. Random Forest generalized better than the single Decision Tree

Random Forest achieved a testing R² of **0.9282**, considerably higher than the Decision Tree's **0.8917**.

### 6. SVR performed poorly on this dataset

Both Linear and RBF SVR produced substantially lower R² scores than the other evaluated models.

---

# 🚀 Future Improvements

Possible future improvements include:

* More extensive feature engineering
* Additional hyperparameter optimization
* Advanced ensemble models
* Feature importance analysis
* Residual analysis
* External validation on unseen datasets
* Model serialization
* Deployment as a prediction application

These are **future enhancements** and are not part of the current implementation.

---

# 📌 Conclusion

This project demonstrates a complete Machine Learning regression workflow for house price prediction.

Starting from raw property data, the project performs preprocessing and feature preparation before applying multiple regression approaches.

The models are evaluated using **MAE, MSE, RMSE, and R² Score**, while Cross-Validation and train-test comparisons are used to study model stability and generalization.

Among the tested models, **Random Forest Regression achieved the strongest overall test performance**, with an R² Score of **0.928156**, MAE of approximately **1.76 million INR**, and RMSE of approximately **2.41 million INR**.

The project therefore demonstrates how systematic model comparison can be used to select an effective regression approach for a house price prediction problem.

---

# 👩‍💻 Author

## Misari Dhorajiyami

🎓 Diploma in Information Technology
📘 Data Science with AI & Machine Learning

**Skills:** Python | SQL | Power BI | Machine Learning

### GitHub

[![GitHub](https://img.shields.io/badge/GitHub-dhorajiyamisri-181717?style=for-the-badge\&logo=github)](https://github.com/dhorajiyamisri)

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Made with Python, Scikit-Learn & Jupyter Notebook**

</div>

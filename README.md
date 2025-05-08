# Wine Quality Prediction (Regression)

This project predicts wine quality scores based on physicochemical properties using both linear and nonlinear machine learning models.  
It combines red and white wine datasets from the UCI Machine Learning Repository and performs full data preprocessing, feature selection, model training, and evaluation.

## Group Member
- Annie Luo
- Daniel Li
- Xinyue(Fay) Yan
- Philip Wang
  
---

##  Dataset
- **Source**: [Wine Quality Data Set - UCI Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Two datasets: **Red wine** and **White wine**.
- Target: **Wine quality score** (continuous integer from 0 to 10).

---

##  Data Preprocessing
- Combined red and white datasets into a single dataset.
- Added a categorical column `'color'` and applied **one-hot encoding** (`color_white`).
- Scaled numerical features using **StandardScaler**.
- Performed an **80/20 train-test split** for model evaluation.

---

##  Models and Methods
### Linear Models:
- **Linear Regression** (Baseline)
- **Ridge Regression** (Tuned using alpha)
- **Lasso Regression** (Tuned using alpha)
- **Feature Selection**:
  - Forward Selection
  - Backward Elimination
  - Best Subset Selection

### Nonlinear Models:
- **Decision Tree Regressor** (tuned via GridSearchCV)
- **Random Forest Regressor** (with Out-of-Bag validation and hyperparameter tuning)
- **Gradient Boosting Regressor** (fine-tuned learning rate, max depth, estimators)

---

##  Hyperparameter Tuning
- **GridSearchCV** used for tuning decision tree, random forest, and gradient boosting models.

---

##  Evaluation Metrics
- **Mean Squared Error (MSE)** for both training and test sets.
- **R² Score** for training, test, and OOB (for Random Forest).
- **Feature Importance Analysis** for Random Forest and Gradient Boosting.
- Compared model performance across different feature subset sizes.

---

##  Key Results
- **Lasso Regression** achieved the best performance among linear models.
- **Random Forest** and **Gradient Boosting** models performed better overall, with Random Forest showing the best generalization (OOB R² ≈ Test R²).
- Full feature models generally performed better than reduced feature models after evaluation.

---

##  Computing Environment
- Python 3 Google Compute Engine backend
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Executed using Jupyter Notebook environment.

---

##  Project Files
- `wine_quality_regression_modeling.ipynb`: Full notebook with all preprocessing, modeling, tuning, evaluation, and visualizations.

---

##  Citation
> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547-553.

---

##  Future Work
- **Expand the data foundation**
- Ingest additional vintages, vineyards, and geographic regions to improve generalization.
- Web-scrape or try to buy tasting‑note datasets, then add it to the lab measurements so we can have a row of sensory details in our dataset.
- **Richer feature engineering**
- Create interaction and polynomial terms (e.g., pH × alcohol, (sulphates)^2) to capture nonlinear quality relationships among the variables.
- Derive domain‑specific ratios such as sugar‑to‑acid and free‑to‑total‑sulfur‑dioxide.
- **Broaden the model lineup**
- Prototype lightweight neural networks to test deep‑learning performance on our data.

---

# Credit Risk & Defaulter Prediction

## Project Overview

This project focuses on building a machine learning solution to predict loan defaults, helping financial institutions minimize risk. The core objective is to identify high-risk applicants (Class 1) while maintaining a balanced approval rate for safe borrowers.

In the context of credit risk, a "False Negative" (missing a defaulter) is significantly more expensive than a "False Positive" (rejecting a good customer). Therefore, this project prioritizes Recall (Sensitivity) over standard Accuracy.

## The Challenge: Handling Imbalance

The dataset presents a moderate class imbalance, with 78% of applicants repaying loans and only 22% defaulting.

If we optimized for standard accuracy, a model could achieve 78% simply by predicting "Safe" for every applicant. To address this, I implemented:

- **Stratified K-Fold Cross-Validation** to ensure representative training batches.
- **Class Weight Balancing** to penalize the model more heavily for missing a defaulter.
- **Recall & ROC-AUC** as the primary evaluation metrics.

## Key Insights & Findings

After training a Random Forest model and analyzing feature importance, three key drivers of risk emerged:

1.  **Debt-to-Income Ratio:** The strongest predictor of default. Applicants borrowing more than 30% of their annual income significantly increased risk exposure.
2.  **Home Ownership:** Renters were approximately 2x more likely to default compared to homeowners, suggesting homeownership acts as a proxy for financial stability or collateral.
3.  **Interest Rates:** Higher interest rates correlated strongly with default, confirming that the institution is correctly pricing riskier loans.

## Methodology

The workflow moved from a baseline model to a tuned ensemble classifier:

1.  **Data Preparation:** \* Removed data entry errors (e.g., Age > 100).
    - Imputed missing values for employment length and interest rates using median strategies.
2.  **Model Selection:**
    - **Logistic Regression:** Provided high Recall (78%) but low Precision (41%), leading to too many false alarms.
    - **Random Forest Classifier:** Captured non-linear relationships better.
3.  **Optimization:**
    - Used **GridSearchCV** to tune hyperparameters (`n_estimators`, `max_depth`).
    - Achieved a final **Recall of 76%** with an **ROC-AUC score of 0.93**, striking the best balance between risk mitigation and business opportunity.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (RandomForest, LogisticRegression, GridSearchCV)

## How to Run

To replicate this analysis:

1.  Clone the repository:

    ```bash
    git clone [https://github.com/RohanDey11/Credit-Risk-Prediction.git](https://github.com/RohanDey11/Credit-Risk-Prediction.git)
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the notebook:
    ```bash
    jupyter notebook notebooks/Credit_Risk_Defaulter.ipynb
    ```

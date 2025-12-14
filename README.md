# Heart Disease Prediction Using Machine Learning

## Project Overview

This project implements a **machine learning pipeline** to predict the presence of heart disease using patient clinical data. It covers the full workflow from data loading, preprocessing, exploratory data analysis (EDA), model training and comparison, hyperparameter tuning, to final evaluation and feature importance analysis.

The goal is to provide a **reproducible and interpretable ML solution** that can help in early detection of heart disease.

---

## Dataset

The dataset `heart_stat.xlsx` contains anonymized patient clinical data with the following columns:

| Column                  | Description |
|-------------------------|-------------|
| age                     | Patient age in years |
| sex                     | 1 = male, 0 = female |
| chest pain type          | Type of chest pain (1–4) |
| resting bp s             | Resting blood pressure (mm Hg) |
| cholesterol             | Serum cholesterol (mg/dl) |
| fasting blood sugar      | 1 if >120 mg/dl, else 0 |
| resting ecg             | Resting electrocardiographic results (0–2) |
| max heart rate           | Maximum heart rate achieved |
| exercise angina          | 1 = yes, 0 = no |
| oldpeak                 | ST depression induced by exercise relative to rest |
| ST slope                 | Slope of peak exercise ST segment (1–3) |
| target                  | 1 = presence of heart disease, 0 = absence |

> **Note:** Raw dataset is included in `data/raw/heart_stat.xlsx`. Preprocessing is done automatically in the scripts.

---

## Repository Structure
```bash
heart-disease-ml/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Optional: processed data (gitignored)
│
├── src/
│ ├── data_loader.py # Load and preprocess data
│ ├── model_training.py # Model definitions and training
│ └── evaluation.py # Evaluation and feature importance
│
├── requirements.txt # Python dependencies
├── main.py # Run full pipeline
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files
```


---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd heart-disease-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```



## Usage
Run full pipeline

```bash
python main.py
```

This will execute the following steps:

1. Load and inspect the dataset
2. Perform exploratory data analysis (EDA)
3. Preprocess features (scaling and train/test split)
4. Train multiple models (Logistic Regression, Random Forest, SVM, Gradient Boosting)
5. Compare baseline performance
6. Hyperparameter tuning for Random Forest
7. Evaluate the best model and show feature importance

Results

Random Forest was the best performer:
- Accuracy: ~0.93
- ROC-AUC: ~0.97

Feature importance highlights key predictors such as chest pain type, ST slope, exercise angina, oldpeak, cholesterol, max heart rate, and age.

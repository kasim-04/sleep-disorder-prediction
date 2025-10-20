# Sleep Disorder Prediction

> **Note:** This project is developed solely for educational purposes.

## Project Overview
This project focuses on predicting sleep disorders based on health and lifestyle features.  
It includes thorough exploratory data analysis, simple feature engineering, and a custom-built multiclass logistic regression model implemented from scratch.  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kasim-04/sleep-disorder-prediction/blob/main/Sleep_Disorder_Prediction.ipynb)

## Dataset
The dataset is publicly available on Kaggle: [Disorder Dataset](https://www.kaggle.com/datasets/varishabatool/disorder)  

The dataset contains multiple health-related features such as `Age`, `Gender`, `Sleep Duration`, `Blood Pressure`, `Physical Activity Level`, `Stress Level`.  
The target variable is `Sleep Disorder`.

## Results
Classification metrics on the test set:

### Per-class metrics
|               | Precision | Recall | F1-score |
|---------------|----------|-------|----------|
| Insomnia      | 0.84     | 0.91  | 0.88     |
| No disorder   | 0.98     | 0.97  | 0.98     |
| Sleep Apnea   | 0.87     | 0.83  | 0.85     |

### Overall metrics
|               | Value |
|---------------|-------|
| Accuracy      | 0.93  |
| Macro avg F1  | 0.90  |
| Weighted avg F1 | 0.93 |
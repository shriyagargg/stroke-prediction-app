# Stroke Risk Prediction using Machine Learning and Streamlit

## Overview

This project is a **Stroke Risk Prediction web application** built using **machine learning classification techniques** and deployed with **Streamlit**.

The application allows users to enter patient health information and instantly receive a **predicted probability of stroke risk** along with a clear risk indication.

The project demonstrates the **complete workflow of a real-world ML system**, including data preprocessing, feature engineering, model training, prediction, visualization, and deployment.

---

## Features

* Interactive patient health input form
* Real-time stroke risk probability prediction
* Clean and user-friendly medical-style interface
* Handles missing values and categorical encoding
* Uses trained machine learning model for inference
* Deployed as an interactive **Streamlit web application**

---

## Tech Stack

* **Programming Language:** Python

* **Libraries & Frameworks:**

  * Streamlit
  * Pandas
  * NumPy
  * Scikit-learn
  * Joblib
  * Matplotlib
  * Seaborn

* **Deployment:** Streamlit Community Cloud

---

## Dataset Information

The model uses a healthcare dataset containing patient attributes such as:

* Age
* Hypertension
* Heart disease
* Average glucose level
* BMI
* Smoking status
* Work type
* Residence type
* Marital status

**Target Variable:**

* `stroke → 0 (No Stroke), 1 (Stroke)`

---

## Machine Learning Model

* **Model Used:** Random Forest Classifier
* Handles **non-linear relationships** in medical data
* Uses **class weighting** to address dataset imbalance
* Trained on preprocessed and scaled clinical features
* Outputs **stroke probability** for real-time prediction

---

## Data Pipeline

1. Load healthcare dataset
2. Remove irrelevant ID column
3. Handle missing BMI values using median imputation
4. Encode categorical variables using one-hot encoding
5. Scale numerical features using StandardScaler
6. Train Random Forest classification model
7. Save model, scaler, and feature columns
8. Load trained model in Streamlit for prediction

---

## Visualizations

* Stroke risk probability indicator
* Prediction result display (High Risk / Low Risk)
* Clean medical-style UI for better interpretation

---

## Project Structure

```
stroke-risk-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── models/
│   ├── stroke_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
└── assets/
    └── screenshot.png
```

---

## Future Enhancements

* Add ROC curve and confusion matrix visualization
* Compare multiple ML models (Logistic Regression, XGBoost)
* Apply SMOTE for improved imbalance handling
* Generate downloadable medical risk report (PDF)
* Add authentication and patient history tracking

---

## Disclaimer

This project is created for **educational and learning purposes only**.
It **must not be used for real medical diagnosis or treatment decisions**.

---

## Author

**Shriya Garg**
B.Tech Computer Science Engineering Student

---

## Contact

**Email:** *Shriyagarg170@gmail.com*
**LinkedIn:** *https://www.linkedin.com/in/shriyagargg/*

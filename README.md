# 🧠 Stroke Risk Prediction Web App

A **machine learning–powered healthcare web application** that predicts the probability of stroke based on patient clinical information.
Built using **Python, Scikit-learn, and Streamlit**, with an interactive and user-friendly interface for real-time risk estimation.

---

## 🚀 Live Demo

🔗 *Add your deployed Streamlit link here*
Example: https://your-username-stroke-risk-prediction.streamlit.app

---

## 📌 Problem Statement

Stroke is one of the leading causes of **death and long-term disability** worldwide.
Early identification of high-risk individuals enables:

* Preventive healthcare
* Lifestyle intervention
* Timely medical support

This project applies **supervised machine learning** to estimate stroke risk from patient health attributes.

---

## 🧾 Dataset Information

The model uses a healthcare dataset containing:

* Age
* Hypertension
* Heart disease
* Average glucose level
* BMI
* Smoking status
* Work type
* Residence type
* Marital status

**Target variable:**
`stroke → 0 (No), 1 (Yes)`

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

* Removed irrelevant **ID column**
* Handled missing **BMI values** using median imputation
* Applied **one-hot encoding** to categorical variables
* Scaled numerical features using **StandardScaler**

### 2. Handling Class Imbalance

* Used **class-weighted Random Forest**
* Improved detection of minority stroke cases

### 3. Model Training & Evaluation

* Stratified **train-test split**
* Evaluated using:

  * Accuracy
  * Precision & Recall
  * **ROC-AUC score**
* Generated:

  * ROC Curve
  * Confusion Matrix
  * Feature Importance

### 4. Deployment

* Saved:

  * Trained model
  * Scaler
  * Feature column order
* Integrated into a **Streamlit web application** for real-time prediction.

---

## 🖥️ Streamlit App Features

* Interactive medical input form
* Real-time **stroke probability prediction**
* Custom healthcare **risk threshold**
* ROC curve & confusion matrix visualization
* Clean, responsive healthcare-style UI
* Consistent preprocessing between training and inference

---

## 🛠 Tech Stack

**Languages & Libraries**

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Joblib
* Matplotlib & Seaborn

---

## 📂 Project Structure

```
stroke-risk-prediction/
│
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

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/stroke-risk-prediction.git

# Navigate to project folder
cd stroke-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📈 Results

* Achieved strong **ROC-AUC performance** for stroke prediction
* Improved minority-class detection using **class weighting**
* Delivered an **end-to-end ML deployment pipeline** from data → model → web app

---

## 🔮 Future Improvements

* SMOTE-based imbalance handling
* Model comparison (Logistic Regression, XGBoost)
* Downloadable **PDF medical risk report**
* User authentication & patient history tracking
* Cloud deployment with monitoring

---

## ⚠️ Disclaimer

This project is created for **educational and research purposes only**
and **must not be used for real medical diagnosis or treatment decisions**.

---

## 👩‍💻 Author

**Shriya Garg**
B.Tech Computer Science Engineering Student

---

## ⭐ If you found this project useful

Consider giving it a **star ⭐ on GitHub**.

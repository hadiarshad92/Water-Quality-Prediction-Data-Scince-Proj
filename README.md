# 💧 Water Potability Prediction – Data Science Project

This repository contains my **Data Science course project**, where I built and evaluated machine learning models to predict **drinking water potability** based on physicochemical properties of water.

---

## 📌 Project Overview
The goal of this project is to predict whether water is **potable (safe to drink)** or **not potable**, using 9 measured attributes such as pH, Hardness, Solids, and Turbidity.  

- **Dataset size:** 3,276 samples  
- **Features:** 9 continuous variables  
- **Target:** `Potability` (0 = not potable, 1 = potable)  
- **Key challenge:** Class imbalance (~39% potable vs 61% not potable)  

---

## 🗂 Dataset Features
- `pH`  
- `Hardness`  
- `Solids`  
- `Chloramines`  
- `Sulfate`  
- `Conductivity`  
- `Organic_carbon`  
- `Trihalomethanes`  
- `Turbidity`  
- `Potability` (target variable)

---

## 🔍 Project Steps
1. **Data Cleaning & Preprocessing**
   - Handled missing values with mean imputation  
   - Standardized all features with `StandardScaler`  
   - Train-test split (70/30)  

2. **Exploratory Data Analysis (EDA)**
   - Histograms, scatter plots, and correlation heatmaps  
   - Feature distribution comparisons for potable vs non-potable samples  

3. **Models Implemented**
   - K-Nearest Neighbors (KNN)  
   - Decision Tree Classifier  
   - Gaussian Naïve Bayes  
   - Logistic Regression  
   - Voting Classifier (ensemble of the above)  

4. **Evaluation Metrics**
   - Accuracy  
   - Precision, Recall, F1-score  
   - Confusion Matrix  
   - *(MSE/RMSE were also tested, though less useful for classification)*  

---

## 📊 Results (Test Set Accuracy)
- **Decision Tree:** 63.58% ✅ (best model)  
- Voting Classifier: 63.28%  
- KNN: 61.95%  
- Gaussian NB: 61.95%  
- Logistic Regression: 59.51%  

👉 **Observation:** Models predict *non-potable* water better than *potable*. Recall for potable samples is relatively low.

---

## ⚙️ Technologies Used
- Python 3  
- Jupyter Notebook  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  

---

## 📁 Repository Structure

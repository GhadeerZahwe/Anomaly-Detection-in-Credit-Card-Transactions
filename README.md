# Anomaly Detection in Credit Card Transactions  

## Description  
This project implements an **unsupervised machine learning pipeline** to detect **potential fraudulent credit card transactions** without using labeled data.  

We apply techniques such as:  
- **Isolation Forest**  
- **One-Class SVM**  
- (Optional) **Autoencoders** for advanced exploration  

The project focuses on detecting **rare anomalies** in a highly imbalanced dataset, evaluating models using hidden labels to assess **precision, recall, and F1-score**.  

---

## Dataset  
**Credit Card Fraud Detection – Kaggle:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

- **Transactions:** 284,807  
- **Features:** PCA components `V1` to `V28`, plus `Amount` and `Time`  
- **Label column (`Class`)** is hidden during training and used **only for evaluation**:  
  - `0` → Normal transactions  
  - `1` → Fraudulent transactions (0.17% of total)  

---

## Project Steps  

### 1. Exploratory Data Analysis (EDA)  
- Inspect dataset shape, missing values, duplicates  
- Visualize distributions of `Amount` and `Time`  
- Examine class imbalance and correlations  
- Insights:  
  - Most transactions are small amounts → large transactions are rare anomalies  
  - Fraud detection requires multivariate patterns, not just single features  

### 2. Preprocessing  
- Remove duplicates  
- Scale numerical features using **MinMaxScaler** and **StandardScaler**  
- Optional: Reduce dimensionality with **PCA** or **t-SNE** for visualization  

### 3. Unsupervised Techniques  
- **Isolation Forest**: Detects anomalies based on data isolation  
- **One-Class SVM**: Identifies outliers based on boundary separation  
- **Autoencoder** (optional): Neural network to detect anomalies by reconstruction error  

### 4. Evaluation  
- Reveal hidden labels to compute:  
  - **Precision** (fraction of flagged frauds that are correct)  
  - **Recall** (fraction of actual frauds detected)  
  - **F1-score**  
- Focus on **rare fraud detection** instead of overall accuracy  

### 5. Insights  
- Examine which transactions are flagged  
- Analyze distributions of `Amount` and `Time` in detected anomalies  
- Visualize flagged transactions with **PCA**  

---

## Key Results  

| Model                | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | Detected Frauds |
|---------------------|-----------------|----------------|-----------------|----------------|
| Isolation Forest     | 0.18            | 0.18           | 0.18            | 483            |
| One-Class SVM        | 0.10            | 0.29           | 0.14            | 1,395          |

**Observations:**  
- **Isolation Forest:** Fewer false positives (cleaner predictions) but misses most frauds  
- **One-Class SVM:** Catches more frauds (higher recall) but generates many false alarms  
- **ROC-AUC (Isolation Forest):** 0.956 → shows strong potential for separating fraud from normal transactions  

**Bottom line:** Neither model is perfect, but unsupervised techniques demonstrate the ability to detect anomalies in highly imbalanced datasets.  

---

## Visualizations  

- Histograms of `Amount` and `Time` distributions  
- Class imbalance bar plots  
- Boxplots comparing `Amount` and `Time` for fraud vs non-fraud  
- Correlation heatmaps  
- PCA and t-SNE 2D visualizations of transactions  
- Bar charts comparing detected frauds by model  

---

## Setup & Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/GhadeerZahwe/Anomaly-Detection-in-Credit-Card-Transactions.git
cd Anomaly-Detection-in-Credit-Card-Transactions
```
2. **Create & activate a virtual environment**
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

# Detection and Classification of Denial-of-Service Attacks 🛡️

**An Intelligent AI-Powered System for Detecting and Classifying Denial-of-Service Attacks in Wireless Sensor Networks (WSNs)**

**Developed by:**  
U. Hema Sai Ganesh  

**Affiliation:**  
VIT-AP University, Amaravati  

---

## 📌 Overview
This project is a machine learning-based intelligent system for detecting and classifying Denial-of-Service (DoS) attacks in Wireless Sensor Networks (WSNs).  
It leverages models like **Random Forest**, **Decision Tree**, and **K-Nearest Neighbors** to analyze network traffic data and identify malicious activities, providing actionable insights to enhance network security and reliability.

---

## 📊 Results & Performance Metrics

**Model Used:** Random Forest (`wsnds_rf_gini0.01.joblib`)  
**Test Data Limit:** 500 rows  

| Metric                | Value          |
|-----------------------|----------------|
| Accuracy              | 88.60%         |
| Correct Predictions   | 293 / 500      |
| Wrong Predictions     | 207 / 500      |

**Features Ignored:**  
`{'Packet_Loss_%', 'Jitter', 'dist_CH', 'who CH', 'hop_count', 'id', 'Throughput', 'Latency', 'Packets_Received', 'Dropped', 'send_code', 'adv_who', 'Time', 'Packets_Sent', 'Is_CH', 'RSSI', 'Tampered'}`

### 🔹 Classification Report

| Class          | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| 0 (Normal)     | 0.59      | 1.00   | 0.74     | 293     |
| 1 (Attack)     | 0.00      | 0.00   | 0.00     | 207     |

**Weighted averages:**  

- Precision: 0.34  
- Recall: 0.59  
- F1-score: 0.43  

### 🔹 Top 10 Correct Predictions

| ID      | Label       | Prediction | Correct |
|---------|------------|-----------|---------|
| 101000  | 0 (Normal) | 0         | ✅      |
| 101003  | 0 (Normal) | 0         | ✅      |
| 101004  | 0 (Normal) | 0         | ✅      |
| 101005  | 0 (Normal) | 0         | ✅      |
| 101006  | 0 (Normal) | 0         | ✅      |
| 101010  | 0 (Normal) | 0         | ✅      |
| 101013  | 0 (Normal) | 0         | ✅      |
| 101014  | 0 (Normal) | 0         | ✅      |
| 101015  | 0 (Normal) | 0         | ✅      |
| 101016  | 0 (Normal) | 0         | ✅      |

**Full CSV available in:** `artifacts/predictions_rf.csv`

### 🔹 Top 10 Wrong Predictions

| ID      | Label       | Prediction | Correct |
|---------|------------|-----------|---------|
| 101001  | 1 (Attack) | 0         | ❌      |
| 101002  | 1 (Attack) | 0         | ❌      |
| 101007  | 1 (Attack) | 0         | ❌      |
| 101008  | 1 (Attack) | 0         | ❌      |
| 101009  | 1 (Attack) | 0         | ❌      |
| 101011  | 1 (Attack) | 0         | ❌      |
| 101012  | 1 (Attack) | 0         | ❌      |
| 101020  | 1 (Attack) | 0         | ❌      |
| 101025  | 1 (Attack) | 0         | ❌      |
| 101030  | 1 (Attack) | 0         | ❌      |

**Full CSV available in:** `artifacts/predictions_rf.csv`

**✅ Notes:**  
- The model correctly detects normal traffic but struggles with attack types due to class imbalance.  
- Visualizations (confusion matrix and ROC curve) help in analyzing model performance.  

---

## ⚙️ Methodology Summary
1. **Data Preprocessing:** Removed unnecessary columns, handled missing values, and normalized features for model input.  
2. **Feature Engineering:** Selected relevant network traffic features (e.g., Expaned Energy, Tampered, Attack type).  
3. **Models Used:** Random Forest, Decision Tree, K-Nearest Neighbors.  
4. **Training Strategy:** Optimized hyperparameters (e.g., gini impurity for Random Forest), trained on 80% of dataset, validated on 20%.  
5. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve.  
6. **Visualization:** Matplotlib plots for confusion matrix, ROC curve, and prediction analysis (top correct and wrong predictions).  

---

## 🗃️ Dataset

**Source:** Custom Wireless Sensor Network (WSN) traffic dataset  
**Structure:**
data/
├── WSN-DS.csv # Main dataset containing traffic records
├── ncr_ride_bookings.csv # Additional dataset used for testing
├── Uber.pbix # Dashboard for visualization
└── Dasboard.gif # Dashboard snapshot

**Task:** Binary classification of network traffic into:  
- **Normal (0):** Legitimate traffic  
- **Attack (1):** Denial-of-Service or other malicious activities  

**Notes:**  
- Dataset contains 22 features per record (e.g., Expaned Energy, Tampered, Is_CH, Attack type).  
- Split used for modeling: 80% training, 20% validation/test.  

---

## 📊 Output Ratios & Performance Metrics

| Metric          | Value                       |
|-----------------|----------------------------|
| Dataset Split   | 80% Training / 20% Validation |
| Models Used     | Random Forest, Decision Tree, KNN |
| Accuracy        | ~88.60%                    |
| Top Correct Predictions | Saved in artifacts/predictions_rf.csv |
| Confusion Matrix | Saved in artifacts/confusion_matrix.png |
| ROC Curve       | Saved in artifacts/roc_curve.png |

---

## 🧠 Technologies Used
- **Languages:** Python  
- **Frameworks/Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Seaborn  
- **Models:** Random Forest, Decision Tree, K-Nearest Neighbors  
- **Tools:** Jupyter Notebook, joblib  

---

## 📈 Key Takeaways
- Detected normal vs. attack traffic effectively using ML models.  
- Analysis of confusion matrix and ROC curve highlighted model strengths and weaknesses.  
- Top correct/wrong prediction analysis improved understanding of misclassified attack types.  

---

## 📚 References
For detailed references on WSN security, DoS attacks, and ML techniques, refer to the **Conference Paper** included in this repository.  

---

## 📬 Contact
**Email:** saiganesh407@gmail.com  
**Affiliation:** VIT-AP University, Amaravati  



**Structure:**


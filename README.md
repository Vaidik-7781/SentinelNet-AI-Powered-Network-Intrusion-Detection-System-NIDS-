# SentinelNet – AI-Powered Network Intrusion Detection System (NIDS)

## Project Overview
SentinelNet-AI is an intelligent Network Intrusion Detection System (NIDS) that detects and classifies network traffic as normal or malicious using machine learning techniques.  
The system processes network flow data, extracts relevant features, trains multiple models, and generates real-time alerts, logs, and reports for identified intrusions.

---

## Milestone Summary

| Milestone | Focus Area | Techniques Used | Key Outcome |
|------------|-------------|-----------------|--------------|
| 1 | Data Cleaning, Encoding & Scaling | LabelEncoder, StandardScaler | Prepared clean and encoded dataset for modeling |
| 2 | Feature Engineering & Baseline Models | PCA, Random Forest Feature Importance | Built initial RF/SVM/LR models; baseline accuracy ≈ 93% |
| 3 | Model Tuning & Anomaly Detection | Isolation Forest, K-Means, RandomizedSearchCV | Compared tuned RF/LR/SVM models; analyzed performance |
| 4 | Real-Time Prediction, Alert Logging, Final Optimization | Streaming Simulation, Logging, SMOTE, PCA, Ensemble | Completed IDS pipeline, added live simulation and alerting |

---

# Milestone 1  
## Weeks 1 – 2 : Dataset Cleaning, Encoding and Preprocessing

### 1. Dataset Acquisition and Inspection
- Downloaded the Wednesday-workingHours dataset from the CICIDS 2017 dataset.
- Inspected dataset using `.head()`, `.info()`, and `.describe()` to understand data structure and balance.
- Verified distribution of normal and attack flows.

### 2. Data Cleaning and Validation
- Removed duplicate and irrelevant records.
- Handled missing values by imputation or removal.
- Ensured data consistency and valid numeric types.

### 3. Encoding Categorical Features
- Applied Label Encoding and One-Hot Encoding to convert categorical features.
- Saved encoded dataset for model training:  
  `Encoded_wednesday.csv`

### 4. Scaling and Normalization
- Standardized features using `StandardScaler` to normalize feature ranges.

### 5. Train/Test Split
- Used stratified sampling in `train_test_split()` to preserve class ratios.

### 6. Exploratory Data Analysis (EDA)
- Performed visual and statistical exploration of features.
- Generated boxplots, histograms, and correlation heatmaps to identify relationships.
- Saved final datasets:  
  `Cleaned_wednesday.csv`, `Encoded_wednesday.csv`

---

# Milestone 2  
## Weeks 3 – 4 : Feature Engineering and Baseline Model Training

### 1. Feature Engineering and Dataset Preparation
- Loaded preprocessed data from Milestone 1.
- Split dataset into features (X) and labels (y).
- Encoded and scaled remaining variables.

### 2. Dimensionality Reduction – PCA
- Applied Principal Component Analysis (PCA) to reduce redundancy.
- Selected components explaining maximum variance.

### 3. Feature Importance and Selection
- Used Random Forest to rank features by importance.
- Chose the Top 10 features contributing most to detection accuracy.

### 4. Correlation and Redundancy Analysis
- Computed correlation matrix to identify multicollinearity.
- Cross-checked feature overlap using PCA results.

### 5. Baseline Model Training
- Trained Random Forest, SVM, and Logistic Regression models.
- Evaluated accuracy, precision, recall, and F1-score.

**Model Performance (Before Tuning)**

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Random Forest | 0.929 | 0.942 | 0.929 | 0.930 |
| SVM | 0.865 | 0.853 | 0.865 | 0.846 |
| Logistic Regression | 0.868 | 0.871 | 0.868 | 0.853 |

- Random Forest achieved the best baseline performance with balanced precision and recall.

### 6. Feature Visualization and Refinement
- Visualized top-ranked features using bar charts.
- Saved reduced feature dataset for subsequent milestones.

---

# Milestone 3  
## Weeks 5 – 6 : Model Tuning and Anomaly Detection

### 1. Anomaly Detection (Unsupervised)
- Implemented K-Means and Isolation Forest algorithms for anomaly detection.
- Preprocessed and scaled the dataset before clustering.
- Isolation Forest outperformed K-Means in detecting rare intrusions.

### 2. Supervised Model Evaluation
- Evaluated Random Forest, SVM, and Logistic Regression models on labeled data.
- Measured classification metrics for performance comparison.

### 3. Hyperparameter Tuning
- Used RandomizedSearchCV to optimize Random Forest and Logistic Regression.
- Improved generalization while avoiding overfitting.
- Skipped SVM tuning due to computational cost and used the pre-trained version.

### 4. Model Evaluation (After Tuning)

**Model Performance (After Tuning)**

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Random Forest (Tuned) | 0.929 | 0.964 | 0.722 | 0.806 |
| Logistic Regression (Tuned) | 0.868 | 0.709 | 0.400 | 0.455 |
| SVM (Pre-trained) | 0.865 | 0.470 | 0.288 | 0.298 |

- Random Forest remained the best model even after tuning.
- Slight drop in recall was observed, possibly due to data imbalance and stricter model boundaries.

### 5. Confusion Matrix and ROC Analysis
- Plotted confusion matrices and ROC curves for model comparison.
- Calculated AUC values to assess discrimination performance.

### 6. Summary
- Unsupervised anomaly detection completed successfully.
- Supervised models evaluated and tuned.
- Tuned Random Forest selected as the final model for real-time simulation.

---

# Milestone 4  
## Weeks 7 – 8 : Real-Time Prediction, Alert Logging, and Final Optimization

### 1. Real-Time Prediction Simulation
- Simulated live traffic classification using the tuned Random Forest model.
- Used a 1,500-packet subset to imitate real-time packet flow.
- Processed data in mini-batches for smooth simulation.
- Saved results as:
  - `week7_predictions.csv` (sample subset)
  - `week7_predictions_full.csv` (complete dataset results).

### 2. Alert Generation and Logging
- Generated alerts for packets classified as intrusions.
- Logged detection events into `alert_logs.txt` with timestamps.
- Created summary reports showing total packets, detected intrusions, and percentage of attacks:
  - `week7_alert_summary.csv`
  - `week7_alert_summary.txt`

### 3. Readable Reports and Visualization
- Compiled a readable report `week7_readable_report.txt` summarizing predictions and statistics.
- Visualized:
  - Bar chart of predicted traffic types.
  - Pie chart showing benign vs intrusion ratios.
- Combined outputs into `week7_final_results.csv`.

### 4. Reason for Using 1,500 Packets Instead of Full Dataset
| Reason | Explanation |
|--------|--------------|
| Efficiency | The full dataset (≈682k packets) would take hours to process. |
| Balance | The 1,500-sample contains both benign and malicious packets, giving a realistic mix. |
| Resource Limitations | Prevents memory overuse and runtime issues in Google Colab. |
| Deployment Readiness | A smaller dataset allows smooth web simulation performance. |

---

## Final Outcomes
- Completed end-to-end intrusion detection and alert generation workflow.  
- Achieved approximately 93 % accuracy with the tuned Random Forest model.  
- Real-time alert and logging system implemented successfully.  
- Results saved in CSV and text formats for readable analysis.  
- Model is ready to be exported for website deployment in future work.

---

## Key Project Files

| File / Folder | Description |
|----------------|-------------|
| Cleaned_wednesday.csv | Cleaned dataset |
| Encoded_wednesday.csv | Encoded dataset |
| Wednesday_top10_features.csv | Top 10 selected features |
| rf_model_tuned.pkl | Tuned Random Forest (used for deployment) |
| week7_predictions.csv | Real-time simulation results |
| alert_logs.txt | Generated intrusion logs |
| week7_alert_summary.csv | Summary of detected intrusions |
| week7_final_results.csv | Consolidated result file |

---

## Conclusion
SentinelNet demonstrates how AI and machine learning can enhance network security by detecting malicious activity in real-time.  
The project evolved through four milestones — data preprocessing, feature engineering, model tuning, and alert generation — achieving high accuracy and practical performance.  
The tuned Random Forest model powers the live website for responsive and accurate network intrusion detection.

---

Developed by:  
Vaidik Gupta
2025  
AI-Powered Intrusion Detection System – SentinelNet Project

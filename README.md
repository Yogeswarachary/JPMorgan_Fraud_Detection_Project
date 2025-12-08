# JP Morgan Chase Fraud Detection Project
### **Overview**
This project detects fraudulent transactions using historical data from JPMorgan Chase, addressing the challenge of identifying fraud amid millions of daily online payments. Banks rely on advanced software to flag suspicious activities, as even minor fraud causes significant losses. The goal focuses on predicting future fraud using a dataset of 6.36 million rows across 11 columns from a 480 MB CSV file.​

### Dataset Description
The dataset includes transaction details with these key columns:

**Column_Name	  Datatype	    Description**
Step	          Integer	      Time step or day of transaction.
Type	          Object	      Transaction type (e.g., PAYMENT, TRANSFER, CASHOUT).
Amount	        Float	        Transaction amount.
NameOrig	      Object	      Sender account ID.
OldBalanceOrig	Float	        Sender's balance before transaction.
NewBalanceOrig	Float	        Sender's balance after transaction.
NameDest	      Object	      Receiver account ID.
OldBalanceDest	Float	        Receiver's balance before transaction.
NewBalanceDest	Float	        Receiver's balance after transaction.
IsFraud	Integer	Fraud         flag (1 = fraud, 0 = normal).
IsFlaggedFraud	Integer	      Bank-flagged suspicious transaction (0 or 1)​.

> **Out of 6.36 million transactions, the bank flagged 16 as fraud via IsFlaggedFraud, while customer feedback identified 8,213 actual frauds in IsFraud.​**

## Project Versions
### Version 1
- Performed EDA, converted CSV to Parquet (480 MB to 240 MB), cleaned data (no nulls/duplicates), analyzed outliers via boxplots (e.g., 102,688 in Step), applied log transformations, engineered 31 features (e.g., balance diffs, zero flags), used one-hot encoding, SMOTE on training data (5M to 10M rows), and tested models like Random Forest, XGBoost, Isolation Forest, Logistic Regression, and hybrids.
- Models showed trade-offs in false positives/negatives, with high compute times (up to 40+ minutes) due to SMOTE.​

### Version 2
- Mirrored EDA/feature engineering but skipped SMOTE, using ScalePosWeight (773) instead for supervised models.
- Tested Logistic Regression, Decision Tree, Naive Bayes, LightGBM, CatBoost (best at 76-80 precision, 2-3 false negatives, 401-506 false positives), XGBoost, Isolation Forest, K-Means;
- hybrid CatBoost achieved 0 false negatives and 357 false positives in 15-20 minutes. Created pipeline, saved PKL models, and generated predictions.​

### Version 3
- Optimised for CPU efficiency with vectorisation for fast EDA visualisations (seconds vs. minutes).
- Retained top Version 2 models (Decision Tree, CatBoost, XGBoost, Isolation Forest, K-Means) for hybrid CatBoost, matching Version 2 results (0 false negatives, 357 false positives) with lower resource use. Built hybrid pipeline class and saved outputs.​

### Deployment
Used 1,000-row sample (800 non-fraud, 200 fraud; 9 columns, excluding targets). Developed Streamlit app (app.py) with frontend UI, hybridpipeline.py, and PKL models in a folder. Install CatBoost/Streamlit via pip, run streamlit run app.py for localhost UI: enter transaction data, click predict for hybrid pipeline output.

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
OldBalanceDest	Float	        Receiver's balance before the transaction.
NewBalanceDest	Float	        Receiver's balance after transaction.
IsFraud	Integer	Fraud         flag (1 = fraud, 0 = normal).
IsFlaggedFraud	Integer	      Bank-flagged suspicious transaction (0 or 1)​.

> **Out of 6.36 million transactions, the bank flagged 16 as fraud via IsFlaggedFraud, while customer feedback identified 8,213 actual frauds in IsFraud.​**

#### Version 1: Initial Approach (SMOTE)
- **Methodology**: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
- **Challenge**: The dataset tripled in size, leading to significant computational overhead. Model execution took 40–60 minutes per model, with the full notebook requiring 6–7 hours.

#### Version 2: Algorithmic Optimization
- **Methodology**: Replaced SMOTE with Scale_Pos_Weight for CatBoost and LightGBM.
- **Results**: Execution time dropped to 4–5 hours. Achieved a milestone of near-zero False Negatives using a Hybrid CatBoost model.

#### Version 3: Streamlining for Deployment 
- **Methodology**: Retained Version 2's EDA but filtered for only the top-performing models.
- **Results**: Execution time reduced to under 3.5 hours. Exported models and EDA steps as PKL files to power a Streamlit deployment.

#### Version 4: The "Indium Standard" Refactor (Final Version)
- **The Challenge**: Testing on synthetic/live data revealed poor precision and potential overfitting despite high training scores.
- **The Solution**: I performed a gap analysis against the high-standard engineering practices used by Indium Software (Indium Tech). By aligning my workflow with Indium’s industry expectations, I implemented the following critical upgrades:
  > **Modular Architecture**: Wrapped all EDA and processing steps into functions, reducing EDA execution time by 90% (now under 30 minutes).

  > **Leakage Prevention**: Integrated Scikit-Learn Pipelines to ensure strict separation between training and validation, eliminating data     leakage and improving real-world normalization.

  > **Business Impact Logic**: Added financial metrics to quantify "Total Fraud Loss Prevented," a mandatory standard for FinTech projects at   firms like Indium.

  > **Performance**: The entire pipeline (EDA + ML) now runs in just 1.5 hours. The refined Hybrid CatBoost model maintains an elite balance:   only 3 False Negatives against 94 False Positives.

  > **Deployment Success**: The updated PKL files now show high precision on synthetic data. By adjusting decision thresholds, the model        effectively identifies fraud patterns in real-time scenarios.

### Specific Deployement for the Version 4
- Standalone App: I’ve created a dedicated repository, JP_Morgan_and_Chase_Fraud_Detection_Deployment (link: https://github.com/Yogeswarachary/JP_Morgan_and_Chase_Fraud_Detection_Deployment), which houses the Streamlit application.
- Model Implementation: This deployment uses the best-performing models (saved as .pkl files) from Version 4.
- Handling Data Constraints: Since the original 6.3M row dataset is private, the deployment runs on a 10,000-row synthetic dataset generated via NumPy. It perfectly mirrors the original data patterns.
- Result Accuracy: By adjusting model thresholds, the system detects nearly all fraud cases. While the smaller data size (10k rows) can lead to a slight increase in False Positives, these are managed via Tiered Risk Actions (Block, Review, Allow).​

### Deployment
Used 1,000-row sample (800 non-fraud, 200 fraud; 9 columns, excluding targets). Developed Streamlit app (app.py) with frontend UI, hybridpipeline.py, and PKL models in a folder. Install CatBoost/Streamlit via pip, run streamlit run app.py for localhost UI: enter transaction data, click predict for hybrid pipeline output.

- Actual JPMORGAN Chase data is a 480 mb  CSV file. This is huge. So I've converted this CSV file into a parquet file and it has become just 260 mb file, so that while working on Python, access retrieval of data can become easier.
Here is the parquet file link: https://drive.google.com/file/d/1SahFUORh8oEP4AeQ5qve60JBqByrSjfu/view?usp=sharing

- I've prepared Power BI charts and a dashboard for this data. Here is the Google Drive link to the Power BI output.
Link: https://drive.google.com/file/d/1edqnEtztEK0-vQrQ2bu1a7SFiozXPDyK/view?usp=sharing

- Quick Presentation made with GenAI (Perplexity)
  Link: https://www.perplexity.ai/apps/e93b42d6-3b3a-4d44-ac8a-0d5211c34580

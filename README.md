# 📊 Marketing Funnel Intelligence & Sales Conversion Analytics

A data-driven B2B marketing funnel analytics project designed to analyze lead behavior, evaluate sales development performance, and predict lead-to-customer conversion using machine learning.

---

## 🚀 Project Overview

This project simulates a real-world B2B marketing funnel environment where Marketing Qualified Leads (MQLs) are analyzed to:

- Understand conversion behavior
- Evaluate Sales Development Representative (SDR) performance
- Identify high-value business segments
- Predict conversion probability using Logistic Regression

The dataset contains lead-level information including business profile, revenue declaration, stock size, and behavioral attributes.

---

## 🎯 Business Objective

The primary objectives of this project:

- Identify factors influencing lead conversion
- Measure conversion rates across business segments
- Evaluate SDR performance efficiency
- Build a predictive model to forecast lead conversion
- Provide data-driven recommendations for sales optimization

---

# 🏗 4-Day Development Sprint

---

## 🗓 Day 1 — Data Understanding & Cleaning

### ✔ Tasks Completed
- Loaded enterprise B2B funnel dataset
- Inspected missing values and data types
- Converted `won_date` to datetime
- Created target variable `Converted`
- Removed non-informative columns (`has_company`, `has_gtin`)
- Cleaned numeric fields and handled inconsistencies

### 📌 Key Learning
Real-world datasets often contain fully null columns and inconsistent categorical encodings that must be validated before modeling.

---

## 🗓 Day 2 — Funnel & Segment Analysis

### ✔ Tasks Completed
- Calculated overall conversion rate
- Analyzed conversion by:
  - Business Segment
  - Lead Type
  - Behaviour Profile
- Evaluated declared revenue patterns
- Identified high-performing segments

### 📊 Insights
- Certain business segments show significantly higher conversion probability.
- Leads with higher declared monthly revenue tend to convert more often.

---

## 🗓 Day 3 — Sales & Revenue Intelligence

### ✔ Tasks Completed
- Analyzed SDR-level performance
- Measured conversion rate per SDR
- Aggregated revenue contribution by Sales Representative
- Ranked sales performance

### 📈 Business Impact
- Identified top-performing SDRs
- Highlighted segments generating maximum revenue

---

## 🗓 Day 4 — Predictive Modeling

### ✔ Tasks Completed
- Selected relevant features
- Cleaned missing values
- Built Logistic Regression model
- Evaluated classification performance

### 🤖 Model Objective
Predict whether an MQL will convert into a customer based on business and behavioral attributes.

---

# 📊 Dataset Features

| Feature | Description |
|----------|------------|
| business_segment | Business industry type |
| lead_type | Lead acquisition category |
| lead_behaviour_profile | Engagement profile |
| average_stock | Average inventory range |
| business_type | Company classification |
| declared_product_catalog_size | Catalog size declared |
| declared_monthly_revenue | Monthly revenue estimate |
| Converted | Target variable (1 = Won, 0 = Not Won) |

---

# 📈 Key KPIs Analyzed

- Overall Conversion Rate
- Segment-wise Conversion Rate
- SDR Conversion Performance
- Revenue Distribution
- Predictive Model Accuracy

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

# 📂 Project Structure

```
marketing-funnel-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_funnel_analysis.ipynb
│   ├── 04_channel_performance.ipynb
│   └── 05_conversion_model.ipynb
│
├── README.md
└── requirements.txt
```

---

# 💡 Strategic Recommendations

- Prioritize high-converting business segments
- Allocate stronger SDR support to high-value leads
- Implement early behavioral scoring to identify likely converters
- Reduce manual evaluation by integrating predictive model insights

---

# 🏆 Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Business KPI Engineering
- Sales Funnel Analytics
- Machine Learning Classification
- Git Version Control
- Structured Project Architecture

---

# 🔮 Future Enhancements

- Random Forest / XGBoost model
- ROC Curve & AUC analysis
- Feature importance visualization
- Streamlit dashboard deployment
- Class imbalance handling (SMOTE)
- Model deployment API

---

## 👩‍💻 Author

Vaishnavi Shivakumar  
*Brains + Heart = Vaishnavi 💡💖 Always curious, always creating, always connecting.*

---

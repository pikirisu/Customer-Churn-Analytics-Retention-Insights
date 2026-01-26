# Customer Churn Analytics & Retention Strategy Dashboard

## Business Problem
Customer churn is a major revenue risk in subscription-based industries like telecom.
Retaining existing customers is significantly cheaper than acquiring new ones.

This project analyzes customer behavior to identify:
- High-risk churn segments
- Key drivers of churn
- Business-focused retention recommendations

## Dataset
- Telecom Customer Churn Dataset (Kaggle / IBM Sample)

## Tools Used
- Python (Pandas, NumPy)
- Data Visualization (Matplotlib, Seaborn)
- Streamlit Dashboard
- SQL-style KPI Aggregations
- ANN Model (supporting, not main focus)

## Key Analytics Questions Answered
- What is the overall churn rate?
- Which contract types churn the most?
- Does tenure affect churn probability?
- Which payment methods indicate churn risk?
- How much revenue is at risk due to churn?

## Key Insights
- Month-to-month customers churn ~3x more than yearly customers
- Customers with low tenure (<12 months) are the highest churn segment
- Electronic check users show higher churn rates
- Retention efforts should focus on high-value short-tenure users

## Deliverable: Interactive Dashboard
Streamlit App: [Live Demo Link Here]

Dashboard Sections:
- Churn Overview KPIs
- Segment Drilldowns
- Revenue Impact Estimation
- Retention Strategy Suggestions

## Business Recommendations
- Offer annual plan discounts to month-to-month customers
- Improve onboarding for new customers (first 6 months)
- Target retention campaigns by payment behavior

## Repository Structure

To utilize app - https://ann-classification-churn-apa3zvgpebjoxfp8jhwaza.streamlit.app/

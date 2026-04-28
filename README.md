# Python Data Science & Analytics

A collection of data science projects focusing on data exploration, 
visualization, and machine learning using Python.

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Task 1: Exploring and Visualizing the Iris Dataset

### Objective
Explore and understand a real-world dataset through data inspection 
and visual analysis.

### Dataset
The Iris dataset contains measurements of 150 flowers from 3 different 
species — Setosa, Versicolor, and Virginica. For each flower, 4 
measurements were recorded: sepal length, sepal width, petal length, 
and petal width.

### Approach
- Loaded the dataset using Seaborn's built-in dataset loader
- Inspected the structure using shape, columns, and head()
- Generated summary statistics using describe()
- Created visualizations to understand patterns in the data

### Visualizations
- **Scatter Plot** — shows the relationship between sepal length and 
petal length across species
- **Histogram** — shows the distribution of petal length across all flowers
- **Box Plot** — shows the spread and outliers of sepal width for each species

### Results & Insights
- Setosa flowers are clearly separable from the other two species based 
on petal measurements
- Petal length shows a bimodal distribution indicating two natural groups 
in the data
- Setosa has a noticeably wider sepal width compared to Versicolor and Virginica
- The scatter plot shows a strong positive correlation between sepal length 
and petal length

---

## Task 2: Credit Risk Prediction

### Objective
Predict whether a loan applicant is likely to default on a loan using 
machine learning classification.

### Dataset
The Loan Prediction Dataset contains information about 614 loan applicants 
from an Indian bank including their income, education, employment status, 
credit history, loan amount, and whether their loan was approved or rejected.

### Approach
- Loaded and inspected both training and testing CSV files
- Identified and handled missing values using mode for categorical columns 
and median for numerical columns
- Converted all categorical text columns to numerical values for model compatibility
- Visualized key features including loan amount distribution, education vs 
loan status, and applicant income
- Trained a Logistic Regression model on 80% of the data
- Evaluated the model on the remaining 20%

### Visualizations
- **Histogram** — distribution of loan amounts showing most common borrowing ranges
- **Count Plot** — education level vs loan approval status comparison
- **Histogram** — distribution of applicant income showing income spread

### Results & Insights
- Model achieved approximately 79% accuracy on unseen test data
- Credit history was the strongest predictor of loan approval
- The model was very good at identifying approved applicants but missed 
some defaulters
- Graduates had a higher loan approval rate compared to non-graduates
- Most applicants had incomes clustered in the lower range with a few 
high income outliers

---

## Task 3: Customer Churn Prediction

### Objective
Identify bank customers who are likely to close their account and leave 
the bank using machine learning classification.

### Dataset
The Churn Modelling Dataset contains information about 10,000 bank customers 
including their age, geography, gender, account balance, credit score, 
number of products, and whether they eventually left the bank (churned).

### Approach
- Loaded and inspected the dataset
- Dropped irrelevant columns such as row number, customer ID, and surname
- Encoded categorical features — Gender using binary mapping and Geography 
using One-Hot Encoding
- Visualized churn distribution, age vs churn, and balance vs churn
- Trained a Random Forest Classifier on 80% of the data
- Analyzed feature importance to understand what drives churn

### Visualizations
- **Count Plot** — overall churn vs retention distribution
- **Histogram** — age distribution comparing churned vs retained customers
- **Histogram** — account balance distribution comparing churned vs retained customers

### Results & Insights
- Age was one of the strongest predictors of churn — older customers churned more
- Customers with higher account balances were more likely to leave
- Geography played a significant role — German customers had higher churn rates
- The Random Forest model outperformed simpler models due to its ensemble approach

---

## Repository Structure

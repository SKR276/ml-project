# -*- coding: utf-8 -*-
"""miniproject_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19NEhW9ZGGgj-ff6VEnoUpBZ5iQWrimWO
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

cust = pd.read_csv("customer_churn.csv")
print(cust.head)

cust.columns.values

cust.dtypes

cust.TotalCharges = pd.to_numeric(cust.TotalCharges, errors='coerce')
cust.isnull().sum()

# Removing missing values
cust.dropna(inplace=True)
# Remove customer IDs from the data set
df2 = cust.iloc[:, 1:]
# Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No', value=0, inplace=True)

# Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()

"""**Exploring Churn Rate**"""

colors = ['#4D3425', '#E4512B']
ax = (cust['Churn'].value_counts() * 100.0 / len(cust)).plot(kind='bar', stacked=True, rot=0, color=colors,
                                                             figsize=(8, 6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers', size=14)
ax.set_xlabel('Churn', size=14)
ax.set_title('Churn Rate', size=14)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x() + .15, i.get_height() - 4.0,
            str(round((i.get_height() / total), 1)) + '%', fontsize=12, color='white', weight='bold',
            )

"""Algorithms"""

# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns=['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler

features = X.columns.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features
print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
result = model_lr.fit(X_train, y_train)
from sklearn import metrics

prediction_test = model_lr.predict(X_test)
# Print the prediction accuracy
print(metrics.accuracy_score(y_test, prediction_test))

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1,
                                  random_state=0, max_features="sqrt",
                                  max_leaf_nodes=30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print(metrics.accuracy_score(y_test, prediction_test))

"""ADA Boost"""

from sklearn.ensemble import AdaBoostClassifier

model_ada = AdaBoostClassifier()
# n_estimators = 50 (default value)
# base_estimator = DecisionTreeClassifier (default value)
model_ada.fit(X_train, y_train)
preds = model_ada.predict(X_test)
metrics.accuracy_score(y_test, preds)

"""XG Boost"""

from xgboost import XGBClassifier

model_xg = XGBClassifier()
model_xg.fit(X_train, y_train)
preds = model_xg.predict(X_test)
metrics.accuracy_score(y_test, preds)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# List of models and their names
models = [
    ("Logistic Regression", model_lr),
    ("XG Boost", model_xg),
    ("ADA Boost", model_ada),
    ("Random Forest", model_rf)
]

# Create empty dictionaries to store the evaluation metrics
accuracy_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}

# Iterate through each model to evaluate its performance
for name, clf in models:
    # Predict using the current model
    predictions = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores[name] = accuracy_score(y_test, predictions)
    precision_scores[name] = precision_score(y_test, predictions)
    recall_scores[name] = recall_score(y_test, predictions)
    f1_scores[name] = f1_score(y_test, predictions)

# Print the evaluation metrics for each model
print("Accuracy Scores:")
print(accuracy_scores)
print("\nPrecision Scores:")
print(precision_scores)
print("\nRecall Scores:")
print(recall_scores)
print("\nF1 Scores:")
print(f1_scores)

# Find the model with the best accuracy
best_model = max(accuracy_scores, key=accuracy_scores.get)
print("\nThe best model based on accuracy is:", best_model)


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib

# Generate sample data (replace this with your own data)
X, y = make_classification(n_samples=1000, n_features=10,n_classes=2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Fit the model with the training data
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')



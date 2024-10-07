#!/usr/bin/env python
# coding: utf-8

# In[6]:


#I chose to use the original data sets from Kaggle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the data 
data = pd.read_csv(r"C:\Users\hfccj\Downloads\diabetes.csv")

# Read the data in a more understanding structure
print(data.head())  
print(data.info()) 

#preprocess the data to train later
X = data.drop('Outcome', axis=1).values  # All columns except 'Outcome' are features
y = data['Outcome'].values  # The 'Outcome' column is the target (0 or 1)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Perceptron model
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


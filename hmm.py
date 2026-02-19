import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample dataset (replace this with your actual data)
X = np.array([[40], [50], [60], [70], [80], [90] , [5]])
y = np.array([45, 55, 65, 75, 85, 95 , 50])

# Create model
RandomForestRegModel = RandomForestRegressor(n_estimators=100, random_state=0)

# Train model
RandomForestRegModel.fit(X, y)

# Predict for 70 marks
X_marks = np.array([[70]])   # Must be 2D array
prediction = RandomForestRegModel.predict(X_marks)

print("Predicted value:", prediction)

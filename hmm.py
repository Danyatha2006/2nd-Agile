"""Random Forest Regression Example."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Sample dataset
x = np.array([[40], [50], [60], [70], [80], [90]])
y = np.array([45, 55, 65, 75, 85, 95])

# Create model
random_forest_reg_model = RandomForestRegressor()

# Train model
random_forest_reg_model.fit(x, y)

# Prediction
x_marks = [[70]]
print(random_forest_reg_model.predict(x_marks))

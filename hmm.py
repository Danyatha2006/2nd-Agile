"""Random Forest Regression Example."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Sample dataset
x = np.array([[30], [35], [60], [70], [80], [90]])
y = np.array([0, 0, 1, 1, 1, 1])

# Create model
random_forest_reg_model = RandomForestRegressor()

# Train model
random_forest_reg_model.fit(x, y)

# Prediction
x_marks = [[70]]
print(random_forest_reg_model.predict(x_marks))

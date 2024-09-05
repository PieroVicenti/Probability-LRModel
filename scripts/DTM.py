import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# Load the dataset
df = pd.read_csv("data/houses.csv")

df = df.join(pd.get_dummies(df.City, dtype=int))

df = df.drop("City", axis=1)

# Define features and target variable
X = df.drop(["Price", "Built Date"], axis=1)  # Features (independent variables)
y = df["Price"]  # Target variable (dependent variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regression model
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
model = DecisionTreeRegressor()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print('Best Params: ', best_params)
print('Best Model: ', best_model)
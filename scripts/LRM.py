import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv("data/houses.csv")

df = df.join(pd.get_dummies(df.City, dtype=int))

df = df.drop('City', axis=1)

# Define features and target variable
X = df.drop(["Price", "Built Date"], axis=1)  # Features (independent variables)
y = df["Price"]  # Target variable (dependent variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Create a linear regression model
model = LinearRegression() 

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Plot the predicted vs. actual prices with the regression line
plt.scatter(y_test, y_pred, label="Data points")
plt.plot(y_test, y_test, color='red', linestyle='--', label="Perfect fit line")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.show()

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
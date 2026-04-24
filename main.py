# Salary Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load dataset
url = "https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv"
salary = pd.read_csv(url)

# Display dataset info
print(salary.head())
print(salary.columns)

# Define features and target
X = salary[['Experience Years']]
y = salary['Salary']

# Visualization
plt.plot(X, y)
plt.xlabel('Experience Years')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.savefig("salary_plot.png")
plt.close()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2529
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test.values)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MAPE:", mape)
print("MSE:", mse)

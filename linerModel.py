import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a simple dataset for donut sales
np.random.seed(42)
customers = np.random.randint(10, 100, 50)  # Number of customers per day (50 days)
donuts_sold = 5 + 0.7 * customers + np.random.randn(50) * 10  # Base sales + 0.7 per customer + noise

# Reshape the data
X = customers.reshape(-1, 1)
y = donuts_sold.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficient (donuts per customer): {model.coef_[0][0]:.2f}")
print(f"Intercept (base number of donuts): {model.intercept_[0]:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

# Plot the results
plt.scatter(X_test, y_test, color='brown', label='Actual sales', alpha=0.7)
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted sales')
plt.xlabel('Number of Customers')
plt.ylabel('Number of Donuts Sold')
plt.title('Donut Sales vs. Number of Customers')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Predict donut sales for a given number of customers
new_customers = np.array([[75]])
predicted_sales = model.predict(new_customers)
print(f"\nPredicted donut sales for {new_customers[0][0]} customers: {predicted_sales[0][0]:.0f}")
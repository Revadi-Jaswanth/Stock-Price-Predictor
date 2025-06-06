import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample historical stock data
data = {
    'Day': list(range(1, 11)),
    'Price': [100, 102, 101, 105, 107, 110, 111, 115, 117, 120]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Day']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Predict future price
future_day = np.array([[11]])
future_price = model.predict(future_day)
print("Predicted price for day 11:", future_price[0])

# Plot
plt.scatter(X, y, color='blue', label='Actual Price')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

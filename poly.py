import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_train = [[5], [10], [12], [15], [18]]
y_train = [[3], [7], [10], [15], [22]]

# Testing set
x_test = [[5], [7], [11], [21]]
y_test = [[9], [14], [15], [23]]

# Train the Linear Regression model and plot a prediction
regress = LinearRegression()
regress.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regress.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_feature = PolynomialFeatures(degree = 2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
X_train_quadratic = quadratic_feature.fit_transform(x_train) #change X_train to x_train
X_test_quadratic = quadratic_feature.transform(x_test) #change X_train to x_train

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_feature.transform(xx.reshape(xx.shape[0], 1))

# Plotting the graph, giving the x-axis title and y-axis title
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c = 'r', linestyle = '--')
plt.title('Pressure Against Temperture')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(x_train, y_train)

# Display the graph and then printing out the training data and testing data
# with their quadratics
plt.show()
print(x_train)
print(X_train_quadratic)
print(x_test)
print(X_test_quadratic)


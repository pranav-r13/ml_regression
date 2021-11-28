import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("age-glucose-dataset.csv")
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,1].values

#Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .30, random_state = 0)

#Bulding model
from sklearn.linear_model import LinearRegression
regress = LinearRegression()

regress.fit(x_train, y_train)

#Predict
y_pred = regress.predict(x_test)
x_pred = regress.predict(x_train)

#Visualisation of training data
plt.title('Simple linear Regession (Training Dataset)')
plt.xlabel('Age')
plt.ylabel('Glucose Level')
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regress.predict(x_train), color = 'blue')
plt.show()


# Visualisation of test data
plt.title('Simple linear Regession (Testing Dataset)')
plt.xlabel('Age')
plt.ylabel('Glucose Level')
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regress.predict(x_test), color = 'blue')
plt.show()


# Visualisation of both test data and train data
plt.title('Simple linear Regession with test data and train data')
plt.xlabel('Age')
plt.ylabel('Glucose Level')
plt.scatter(x, y, color = 'red')
plt.plot(x, regress.predict(x), color='blue')
plt.show()

y_prd = regress.predict([[55]])
print("\nPredicting the glucose level using simple linear regression for age = 55: ",y_prd)
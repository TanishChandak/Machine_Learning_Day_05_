import pandas as pd
import matplotlib.pyplot as plt

# loading the dataset:
df = pd.read_csv('CarPrices1.csv')
print(df)

# Creating graph to see the relationship between (mileage and sell_price):
plt.scatter(x=df.mileage, y=df.sell_price)
plt.show()

# Creating graph to see the relationship between (age and sell_price):
plt.scatter(x=df.age, y=df.sell_price)
plt.show()

# independent variable:
X = df[['mileage', 'age']]
print(X)
y = df[['sell_price']]
print(y)

# Now, we train and test our data-sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# it will select the 80% of data through dataset which is RANDOMLY :
# and if you want to same data we can use the method called (random_state=10) for same data:
print(X_train)
print("The length of the X_train is: ",len(X_train))
print("The length of the X_test is: ", len(X_test))
print(y_train)
print("The length of the y_train is: ",len(y_train))
print("The length of the y_test is: ", len(y_test))

# Now, applying linear_Regression model:
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Now, train the model:
model.fit(X_train, y_train)
print(model.predict(X_test))
# print(model.predict(y_test))
print(model.score(X_test, y_test))
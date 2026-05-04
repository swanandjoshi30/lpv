import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

df = pd.read_csv("1_boston_housing.csv")

x = df.loc[:,df.columns != 'MEDV']
y = df.loc[:,df.columns == 'MEDV']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss = 'mse', metrics = ['mae'])

model.fit(X_train, y_train, epochs = 100 , verbose = 1)

mse, mae = model.evaluate(X_test, y_test)
print(mse)
print(mae)

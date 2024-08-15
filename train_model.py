import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

X_train = pd.read_csv('../../data/X_train_scaled.csv')
y_train = pd.read_csv('../../data/y_train.csv')

model = LinearRegression()
model.fit(X_train, y_train)

with open('../../models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

X_test = pd.read_csv('../../data/X_test_scaled.csv')
y_test = pd.read_csv('../../data/y_test.csv')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

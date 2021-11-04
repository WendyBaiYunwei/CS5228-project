from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd

l = pd.read_csv('small_test.csv').columns.tolist()
X_train = pd.read_csv('small_train.csv')[l]
X_test = pd.read_csv('small_test.csv')
y_train = pd.read_csv('small_train_y.csv')['price']
y_test = pd.read_csv('small_test_y.csv')['price']

model = RandomForestRegressor(n_estimators = 700, max_features = 7)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
print(mse)

# model = RandomForestRegressor(n_estimators = 700, max_features = 7)
# df = pd.read_csv('numerical_cleaned_train.csv')
# l = df.columns.tolist()
# l.remove('price')
# X = df[l]
# y = df['price']
# model.fit(X, y)
# final_X = pd.read_csv('numerical_cleaned_test.csv')
# y = model.predict(final_X)
# y = pd.DataFrame(y)
# y.to_csv('predictions.csv')
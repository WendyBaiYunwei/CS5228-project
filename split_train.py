from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('new_train.csv')
l = df.columns.tolist()
l.remove('price')
X = df[l]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)

def remove(x):
    return x
    # return x.drop(columns = 'Unnamed: 0')
remove(X_train).to_csv('small_train.csv')
remove(X_test).to_csv('small_test.csv')
remove(y_train).to_csv('small_train_y.csv')
remove(y_test).to_csv('small_test_y.csv')
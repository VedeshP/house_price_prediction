import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('housing.csv')

X = data.drop('median_house_value', axis=1)

X = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)

X = X.dropna()

y =data['median_house_value']
y = y[X.index]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def load_data_all():
    return X_train, X_val, X_test, y_train, y_val, y_test
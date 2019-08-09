import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from preprocess import MySimpleScaler

iris = load_iris()
scaler = MySimpleScaler()
X = scaler.preprocess(iris.data)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

with open ('model.pkl', 'wb') as f:
  pickle.dump(model, f)
with open ('preprocessor.pkl', 'wb') as f:
  pickle.dump(scaler, f)
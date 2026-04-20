from unittest.mock import inplace
import numpy as np
import pandas as pd

df=pd.read_csv("income.data.csv")
df.drop(columns="Unnamed: 0",inplace=True)

# print(df)
# print(df.shape)
class SimpleLinearRegressionNumericalView:
 def __init__(self):
  self.a0 = None
  self.a1 = None
  self.SSE = None
  self.MSE = None
def predict(self, X):
 X = np.asarray(X, dtype=float).reshape(-1)
 return self.a0 + self.a1 * X
@staticmethod
def sse(y, y_hat):
   return np.sum((y - y_hat) ** 2)
def fit(self, X, y, show_steps=True):
   X = np.asarray(X, dtype=float).reshape(-1)

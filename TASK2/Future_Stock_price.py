import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

#short term stock data 
data = yf.download('AAPL', start='2025-05-01')
# print(data)

# tream data
new_data = data.tail()
print(new_data)
# check the number of rows and columns and the null values
print("\n\n")
print(new_data.info())


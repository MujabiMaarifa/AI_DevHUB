# Stock Price Prediction Project 📈

## 📌 Objective
Predict next-day closing prices using historical market data with machine learning regression techniques.

## 🗃️ Dataset
**Yahoo Finance Market Data** (via `yfinance` Python library)  
- **Tickers**: AAPL (Apple), TSLA (Tesla), or any valid symbol
- **Features**:
  - Open, High, Low, Close (OHLC) prices
  - Volume
  - Adjusted Close
- **Time Period**: Customizable (default: 1 year daily data)

## 🛠️ Implementation

### 1. Data Fetching
```python
import yfinance as yf

# Fetch Apple data
data = yf.download('AAPL', start='2025-05-01')
print(data.head())

### sampled the data and split it for machine train_test_split x,y
from sklearn.model_selection import train_test_split
#trained my model using linear regression model and fit the data 

### how to run
pip install yfinance scikit-learn matplotlib
python future_price_prediction.ipynb

### 2. Model used
Linear Regression

import pandas as pd
import numpy as np
from ta import add_all_ta_features

# Load the data into a pandas dataframe
df = pd.read_csv('MSFT.csv')

# Remove unnecessary columns
df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

# Rename columns
df.columns = ['date', 'adj_close']

# Convert the date column to a pandas datetime object
df['date'] = pd.to_datetime(df['date'])

# Set the date column as the index of the dataframe
df.set_index('date', inplace=True)

# Fill in missing values using the forward-fill method
df.fillna(method='ffill', inplace=True)

# Create technical analysis features using the ta library
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

# Create new features: rolling window averages of the technical analysis features
df['MA10'] = df['adj_close'].rolling(window=10).mean()
df['MA50'] = df['adj_close'].rolling(window=50).mean()
df['RSI10'] = df['momentum_rsi'].rolling(window=10).mean()
df['RSI50'] = df['momentum_rsi'].rolling(window=50).mean()
df['ADX10'] = df['trend_adx'].rolling(window=10).mean()
df['ADX50'] = df['trend_adx'].rolling(window=50).mean()

# Create labels: the adjusted close price shifted by one day
df['label'] = df['adj_close'].shift(-1)

# Remove outliers: any data point more than three standard deviations from the mean
mean = np.mean(df['adj_close'])
std = np.std(df['adj_close'])
df = df[(df['adj_close'] >= mean - 3 * std) & (df['adj_close'] <= mean + 3 * std)]

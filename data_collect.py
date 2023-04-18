import requests
import pandas as pd
import io

# Define the API endpoint
url = 'https://query1.finance.yahoo.com/v7/finance/download/MSFT'

# Define the start and end dates for the data
start_date = '2015-01-01'
end_date = '2022-04-16'

# Define the query parameters
params = {
    'period1': int(pd.Timestamp(start_date).timestamp()),
    'period2': int(pd.Timestamp(end_date).timestamp()),
    'interval': '1d',
    'events': 'history',
    'includeAdjustedClose': 'true'
}

# Send a GET request to the API endpoint with the query parameters
response = requests.get(url, params=params)

# Load the data into a pandas dataframe
df = pd.read_csv(io.StringIO(response.text))

# Print the first few rows of the dataframe
print(df.head())

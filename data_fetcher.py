#####################################
# Class for fetching data 


import pandas as pd
import numpy as np
from binance.client import Client
from utils import get_interval_from_string #for the mapping of the interval




class DataFetcher:
    """
    A class to fetch historical candlestick data from the Binance API.
    
    The DataFetcher class allows you to retrieve market data from a specified client and 
    for a specified symbol.

    Attributes:
        client (Client): The Binance API client used for making requests.
        symbol (str): The market symbol to fetch data for (e.g., "BTCUSDT").


    Methods:
        __init__(client, symbol):
            Initializes the DataFetcher with the specified parameters.
        
        get_interval_from_string(interval_str):
            Converts a user-friendly interval string (e.g., "1DAY") into the corresponding Binance constant.
        
        fetch_data(interval, start_date, end_date):
            Fetches the historical candlestick data and returns it as a pandas DataFrame.

        Other functions will be documented in the memo 
    """


    # Now I can initiate the class
    def __init__(self, client, symbol):
        """
        Initializes the DataFetcher instance with the provided parameters.

        Parameters:
            client (Client): The Binance API client instance used for fetching data.
            symbol (str): The symbol for which to fetch the historical data (e.g., "BTCUSDT").
        
        """
        self.client = client
        self.symbol = symbol




    def fetch_data(self, interval, start_date, end_date):
        """
        Fetches the historical candlestick data for the specified symbol, interval, and date range.
        
        This method calls the Binance API and converts the raw data into a pandas DataFrame.
        
        Parameters: 
            interval (str): this is a string identifying the time interval you want to dowload the data (e.g 1DAY)
            start_date (str): Starting date from which you want to download the data, in Binance format (e.g. "1 Jan, 2025")
            end_date (str): Ending date until which you want to download the data, in Binance format (e.g. "3 Jan, 2025")
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the candlestick data with columns
                          ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        # Fetch historical klines
        candles = self.client.get_historical_klines(self.symbol, get_interval_from_string(interval), 
                                                    start_date, end_date)
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(candles, columns= columns)
        
        # Process DataFrame
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    # End of fetch_data() ----------------------------------------------------------------------------------------------------------


    def fetch_VWAP(self, interval, start_date, end_date):
        """
        Fetches the historical candlestick data and calculates the Volume Weighted Average Price (VWAP).
        
        This function retrieves the market data using the `fetch_data` method of the `DataFetcher` class, 
        then calculates the VWAP using the formula: VWAP = (Price * Volume) / Volume, where Price is the typical price 
        (average of high, low, and close).

        Parameters:
            interval (str): The time interval for the candlestick data (e.g., "1DAY", "1HOUR").
            start_date (str): The starting date for the data retrieval (e.g., "1 Jan, 2025").
            end_date (str): The ending date for the data retrieval (e.g., "3 Jan, 2025").

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical candlestick data along with a new column 'VWAP', 
                        representing the Volume Weighted Average Price.
            """
        df = DataFetcher.fetch_data(self, interval, start_date, end_date)
        
        # Calculate VWAP ( (Price * Volume) / Volume )
        df['VWAP'] = np.cumsum(df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)) / np.cumsum(df['volume'])
        
        return df
    # End of fetch_VWAP() ----------------------------------------------------------------------------------------------------------



    def VWAP(self, df):
        """
        This function is created to add a column to the dataframe to compute teh VWAP, the data frame should be created with 
        the class DataFetcher for consistency reasons (df = DataFetcher.fetch_data(self, interval, start_date, end_date))
        The newly added column is called "VWAP". The VWAP is computed as 
        VWAP = cumsum(Volume * (H + L + C)/3 ) / cumsum(Volume).
        
        This VWAP is implemented on a session basis (every session starts at midnight). 
        The function uses (where possible) vectors and numpy to be more efficient  
        """
        # Ensure the index is in datetime format
        #df = pd.DataFrame(df)
        df.index = pd.to_datetime(df.index, unit='ms')

        # Extract session date from index to reset VWAP daily
        df["session_date"] = df.index.date

        # Calculate Typical Price (TP)
        df['TP'] = (df['high'] + df['low'] + df['close']).values / 3  

        # Calculate session-based VWAP
        df['VWAP'] = df.groupby('session_date', group_keys=False).apply(
            lambda g: np.cumsum(g['volume'] * g['TP']) / np.cumsum(g['volume'])
        )

        # Drop columns created for the VWAP creation 
        df = df.drop(columns= ["TP","session_date"])

        #Return
        return df

    # End of VWAP() ----------------------------------------------------------------------------------------------------------------


    def RSI(self, df, period=14, column = "close"):
        """
        Calculate the Relative Strength Index (RSI) for a given dataset.
        
        Parameters:
        - df (pd.Series): Series of closing prices
        - period (int): Lookback period for RSI (default=14)
        - columns (str): Column from which to compute the RSI. It should not be changed (default = "close")
        
        Returns:
        - pd.Series: RSI values
        """
        #df = pd.DataFrame(df)
        # Ensure the 'close' column exists in the dataset
        if column not in df.columns:
            raise ValueError(f"The dataset must contain a {column} column.")
        
        # Initialise the function
        delta = df[column].diff(1)

        gain = np.where(delta > 0 , delta, 0)
        loss = np.where(delta < 0 , -delta, 0)

        # As a series 
        gain = pd.Series(gain, index = df.index)
        loss = pd.Series(loss, index = df.index)

        # Create averages
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        # Compute the Relative Strenght
        RS = avg_gain / avg_loss

        # Compute the Index (RSI)
        RSI = 100 - ( 100/(1+RS) )

        # Remove the first na value 
        if np.isnan(RSI[0]):
            RSI[0] = 0

        # Assing the RSI 
        df["RSI"] = RSI 

        # Return
        return df
    # End of RSI()--------------------------------------------------------------------------------------------------------------------------



    def SMA(self, df, period=14, column = "close"):
        """
    Calculate the Simple Moving Average (SMA) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data.
    - column (str): Column name for closing prices (default="close").
    - period (int): Lookback period for SMA (default=14).

    Returns:
    - pd.Series: SMA values added as a new column in the DataFrame.
        """
        #df = pd.DataFrame(df)
        # Ensure the 'close' column exists in the dataset
        if column not in df.columns:
            raise ValueError(f"The dataset must contain a {column} column.")
        
        # Calcualte it 
        df["SMA"] = df[column].rolling(window = period, min_periods=1).mean()

        return df
    # End of SMA()--------------------------------------------------------------------------------------------------------------------------




    def ATR(self, df, period = 14):
        """
        This function to add a column to a dataframe, The column contains the Average True Range (ATR), computed as 
        TR = max(H - L ; |H-C_prev| ; |L - C_prev|)
        ATR = 1/n * cumsum_1_to_n(TR_i)

        This is then smoothed with the Wilder's smoothing method
        ATR_i = ( (ATR_t-1 x (n-1))+ TR_i )/ n 
        This method allow to give to past observations less weight

        Teh computations are vectorized with numpys for optimized efficiency.

        Paramteres:
        df (pd.DataFrame): Dataframe created with the DataFetcher class (df = DataFetcher.fetch_data(self, interval, start_date, end_date))
        period (int): smoothing period over which compute the cumsum()
        """
        high_low = df["high"].values - df["low"].values
        high_close = np.abs(df["high"].values - df["close"].shift(1).fillna(df["close"].iloc[0]).values)
        low_close = np.abs(df['low'].values - df['low'].shift(1).fillna(df['close'].iloc[0]).values)

        # Define the true range
        true_range = np.maximum(high_low, high_close, low_close)

        # Wilder's smoothing method
        atr = np.empty_like(true_range) # Creates random values (faster than a vector with 0s)
        atr[:period] = np.mean(true_range[:period]) # This is the ATR until time i
        for i in range(period, len(atr)):  # For the rest, I use the Wilder's smoothing approach
            atr[i] = ( atr[i-1] * (period - 1) + true_range[i] ) / period

        # Return
        df["ATR"] = atr # Add a column to the dataframe
        return df
    #End of ATR() ----------------------------------------------------------------------------------------------------------------



    def fetch_VWAP_session(self, interval, start_date, end_date, time_session = "TV"):
        """
        Fetches the historical candlestick data and calculates the Volume Weighted Average Price (VWAP).
        This function calculate the VWAP for every session. It means that every day the calculation starts again. This is 
        more informative for trading, but it is slower and not applicable to time frames > 1 DAY 
        
        This function retrieves the market data using the `fetch_data` method of the `DataFetcher` class, 
        then calculates the VWAP using the formula: VWAP = (Price * Volume) / Volume, where Price is the typical price 
        (average of high, low, and close).

        Parameters:
            interval (str): The time interval for the candlestick data (e.g., "1DAY", "1HOUR").
            start_date (str): The starting date for the data retrieval (e.g., "1 Jan, 2025").
            end_date (str): The ending date for the data retrieval (e.g., "3 Jan, 2025").
            time_session (str): The desired time at what the session should start. The possible inputs are "TV" (tradin view)
            in which the VWAP indicator starts at 19, and "Midnight", in which the VWAP indicator start at 00:00. Baisc value is set to
            "TV".

        Returns:
            pd.DataFrame: A pandas DataFrame containing the historical candlestick data along with a new column 'VWAP', 
                        representing the Volume Weighted Average Price.
        """
        
        df = DataFetcher.fetch_data(self, interval, start_date, end_date)

        # Ensure the index is in datetime format
        df.index = pd.to_datetime(df.index, unit='ms')

        # Extract session date from index to reset VWAP daily
        if time_session == "TV":
            df['session_date'] = (df.index - pd.Timedelta(hours=19)).date 
        if time_session == "Midnight": 
            df["session_date"] = df.index.date

        # Calculate Typical Price (TP)
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3  

        # Calculate session-based VWAP
        df['VWAP'] = df.groupby('session_date', group_keys=False).apply(
            lambda g: (g['volume'] * g['TP']).cumsum() / g['volume'].cumsum()
        )

        # Keep only relevant columns (index remains as timestamp)
        df = df[['open', 'high', 'low', 'close', 'volume', 'VWAP']]

        return df
    # End of fetch_VWAP_session() ----------------------------------------------------------------------------------------------------------



            
    
# Example usage
if __name__ == "__main__": # __name__ == "__main__" esnures that this block is just run here 
    #(it does not run when I import the module in other .py files)
    client = Client(api_key="your_api_key", api_secret="your_api_secret")

    # Create a fetcher instance
    fetcher = DataFetcher(client, symbol="BTCUSDT")
    
    # Fetch and display data
    df = fetcher.fetch_data(interval= "1HOUR", start_date="1 Jan, 2023",
                        end_date="2 Jan, 2025")
    print(df.head())

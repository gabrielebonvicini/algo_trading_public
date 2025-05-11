# Functions to support the classes 
#####################################################

from binance.client import Client
import plotly.io as pio # This is for the charts
pio.renderers.default = "browser" # This is to plot the charts on a web page 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np


#############################################################
#################### General functions ######################
#############################################################



#Before the __init__, I want to create human readable variables
interval_mapping= {
    "1MIN": Client.KLINE_INTERVAL_1MINUTE,
    "5MIN": Client.KLINE_INTERVAL_5MINUTE,
    "15MIN": Client.KLINE_INTERVAL_15MINUTE,
    "30MIN": Client.KLINE_INTERVAL_30MINUTE,
    "1HOUR": Client.KLINE_INTERVAL_1HOUR,
    "2HOUR": Client.KLINE_INTERVAL_2HOUR,
    "4HOUR": Client.KLINE_INTERVAL_4HOUR,
    "6HOUR": Client.KLINE_INTERVAL_6HOUR,
    "8HOUR": Client.KLINE_INTERVAL_8HOUR,
    "12HOUR": Client.KLINE_INTERVAL_12HOUR,
    "1DAY": Client.KLINE_INTERVAL_1DAY,
    "3DAY": Client.KLINE_INTERVAL_3DAY,
    "1WEEK": Client.KLINE_INTERVAL_1WEEK
}


def get_interval_from_string(interval_str):
    """
    Converts a human-readable interval string (e.g., "1DAY") into the corresponding Binance constant.

    Parameters:
        interval_str (str): The interval string (Possible inputs: "1DAY", "1WEEK", "3DAY", "12HOUR", "8HOUR", "4HOUR", "6HOUR",
        "2HOUR", "1HOUR", "30MIN", "15MIN", "5MIN", "1MIN").
    
    Returns:
        str: The corresponding Binance constant for the specified interval.
    
    Raises:
        ValueError: If the interval string is not valid.
    """
    if interval_str not in interval_mapping:
        raise ValueError(f"Invalid interval: {interval_str}. Valid options are {', '.join(interval_mapping.keys())}.")
    return interval_mapping[interval_str]

# End of get_interval --------------------------------------------------------------------------------------------------------


def train_test_split(df, test_size):
    """
    Function to split the dataset between train and test dataset, without having to use scikit learn    
    
    Parameters:
    df (pd.DataFrame):
    test_size (float64): percentage of test size (tipically 0.2 or 0.3)
    """ 

    shuffled_indices = np.random.permutation(len(df))
    test_size_abs = int( len(df) * test_size )

    # Split the data 
    train_indices = shuffled_indices[:-test_size_abs] 
    test_indices = shuffled_indices[-test_size_abs:]

    X_train = df.iloc[train_indices]
    X_test = df.iloc[test_indices]

    return X_train, X_test










#############################################################
#################### Data Visualization #####################
#############################################################



def plot_VWAP(self, df):
    """
    Plots a candlestick chart with the VWAP line.
    Please note that the dataset supported is created with DataFetcher.fetch_VWAP()
    Please ref to Example usage 
    
    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing OHLCV data and a 'VWAP' column.

    # Example usage
        df_VWAP = fetch_VWAP(interval="1HOUR", start_date="1 Jan, 2023", end_date="2 Jan, 2025")
        plot_VWAP(df_VWAP)
    """

    # Ensure data is sorted by timestamp
    df = df.sort_index()

    # Create candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candles'
        )
    ])

    # Add VWAP as an orange line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color='orange', width=2)
    ))

    # Layout customization
    fig.update_layout(
        title="Candlestick Chart with VWAP",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"  # Dark theme for better contrast
    )

    # Show the plot
    fig.show()

# End of plot_VWAP() ----------------------------------------------------------------------------------------------------------




def plot_intro_strategy(df):
    """
    Plots a candlestick chart with entry prices, stop losses, and take profits.
    Please note that the dataset should include columns for Long and Short signals, 
    trade prices, stop losses, and take profits (as defined in your strategy).
    
    This function is created ad hoc to plot the results of Strategies.intro_strategy().
    At current implementation it is not generalized for other strategeis. 
    It can be generalized if the other strategeis create columns witht the same name 
    
    Parameters:
        df (pd.DataFrame): A pandas DataFrame containing OHLCV data with columns for:
            - Long_Signal, Long_Trade_Price, Long_stop_loss, Long_take_profit
            - Short_Signal, Short_Trade_Price, Short_stop_loss, Short_take_profit

    Example usage:
        df_strategy = fetch_data_with_signals()  # Assume this is your strategy dataframe
        plot_strategy_with_signals(df_strategy)
    """

    # Ensure data is sorted by timestamp
    df = df.sort_index()

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    )])

    # Add Long Entries, Stop Losses, and Take Profits
    long_entries = df[df['Long_Signal'] == True]
    fig.add_trace(go.Scatter(
        x=long_entries.index,
        y=long_entries['Long_Trade_Price'],
        mode='markers',
        name='Long Entry',
        marker=dict(symbol='triangle-up', color='green', size=10),
        text='Long Entry',
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=long_entries.index,
        y=long_entries['Long_stop_loss'],
        mode='markers',
        name='Long Stop Loss',
        marker=dict(symbol='cross', color='red', size=12),
        text='Long Stop Loss',
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=long_entries.index,
        y=long_entries['Long_take_profit'],
        mode='markers',
        name='Long Take Profit',
        marker=dict(symbol='circle', color='blue', size=10),
        text='Long Take Profit',
        hoverinfo='text'
    ))

    # Add Short Entries, Stop Losses, and Take Profits
    short_entries = df[df['Short_Signal'] == True]
    fig.add_trace(go.Scatter(
        x=short_entries.index,
        y=short_entries['Short_Trade_Price'],
        mode='markers',
        name='Short Entry',
        marker=dict(symbol='triangle-down', color='red', size=10),
        text='Short Entry',
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=short_entries.index,
        y=short_entries['Short_stop_loss'],
        mode='markers',
        name='Short Stop Loss',
        marker=dict(symbol='cross', color='green', size=12),
        text='Short Stop Loss',
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=short_entries.index,
        y=short_entries['Short_take_profit'],
        mode='markers',
        name='Short Take Profit',
        marker=dict(symbol='circle', color='blue', size=10),
        text='Short Take Profit',
        hoverinfo='text'
    ))

    # Layout customization
    fig.update_layout(
        title="Candlestick Chart with Strategy Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",  # Dark theme for better contrast
        legend=dict(title='Signal Type', x=0.01, y=0.99, traceorder='normal'),
        showlegend=True
    )

    # Adjust marker placement (slightly above/below candles)
    for i in range(len(df)):
        if pd.notna(df['Long_Trade_Price'].iloc[i]):
            fig.update_traces(marker=dict(symbol='triangle-up', color='green', size=10), selector=dict(name='Long Entry'))
        if pd.notna(df['Long_stop_loss'].iloc[i]):
            fig.update_traces(marker=dict(symbol='cross', color='red', size=12), selector=dict(name='Long Stop Loss'))
        if pd.notna(df['Long_take_profit'].iloc[i]):
            fig.update_traces(marker=dict(symbol='circle', color='blue', size=10), selector=dict(name='Long Take Profit'))
        if pd.notna(df['Short_Trade_Price'].iloc[i]):
            fig.update_traces(marker=dict(symbol='triangle-down', color='red', size=10), selector=dict(name='Short Entry'))
        if pd.notna(df['Short_stop_loss'].iloc[i]):
            fig.update_traces(marker=dict(symbol='cross', color='green', size=12), selector=dict(name='Short Stop Loss'))
        if pd.notna(df['Short_take_profit'].iloc[i]):
            fig.update_traces(marker=dict(symbol='circle', color='blue', size=10), selector=dict(name='Short Take Profit'))

    # Show the plot
    fig.show()

# Example Usage
# Assuming you have a DataFrame `df` with columns for the strategy signals and prices
# plot_strategy_with_signals(df)

# End of plot_intro_strategy()--------------------------------------------------------------------------------------------------------------


def plot_cumulative_pnl(df):
    """
    Plots a simple line chart of Cumulative_PnL over time.
    
    Parameters:
    df (pd.DataFrame): A dataframe with a datetime index and a column 'Cumulative_PnL'.
    """
    # Ensure index is datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Cumulative_PnL'], label='Cumulative PnL', color='b')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.title('Cumulative PnL Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
# End of plot_cumulative_pnl() ------------------------------------------------------------------
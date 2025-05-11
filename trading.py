import pandas as pd
from binance.client import Client
from data_fetcher import DataFetcher
from utils import get_interval_from_string # To convert the strings in python biannce into human readable
import numpy as np


#################################### Class to try strategies ####################################
# #################################### #################################### #################################### # 


class Strategies: 
    """
    A class to simply impelement and run strategies. The class it is used because this is not anymore a data fethcer for data inspection,
    but be cause we need to inspect the strategies in a different way.
    
    Attributes:
        client (Client): The Binance API client used for making requests.
        symbol (str): The market symbol to fetch data for (e.g., "BTCUSDT").


    Methods:
        __init__(client, symbol):
            Initializes the DataFetcher with the specified parameters.
        
        Other functions will be documented in the memo
    """

    def __init__(self, client, symbol):
        """
        Initializes the Strategy with the provided parameters.

        Parameters:
            client (Client): The Binance API client instance used for fetching data.
            symbol (str): The symbol for which to fetch the historical data (e.g., "BTCUSDT").
        """
        self.client = client
        self.symbol = symbol
        self.data_fetcher = DataFetcher(client, symbol)
    # End of init ----------------------------------------------------------------------------------------------------------------

   
    def intro_strategy(self, interval, starting_date, ending_date, target_RR):
        """
        This strategy has been created only to try the code and the class. Please refer to the memo for documentation regarding
        this strategy.
        The rationale behind this strategy is
        - If the two options are met, create a signal (previous close below VWAP, new close above VWAP).
        - After the signal, the trade enter if the price open higher than the previous high, or if hte price close > than the previous high (that meaning
        that it enters the trade when the previous high is crossed)

        Parameters: 
        interval (str): this is a string identifying the time interval you want to dowload the data (e.g 1DAY)
        start_date (str): Starting date from which you want to run the strategy, in Binance format (e.g. "1 Jan, 2025")
        end_date (str): Ending date until which you want to download to run the strategy, in Binance format (e.g. "3 Jan, 2025")
        target_RR (int): traget RR ration that you want to achieve
        """
        # Create the dataset 
        df = self.data_fetcher.fetch_data(interval=interval, start_date= starting_date, end_date=ending_date)

        # Add the VWAP and the ATR to the datset 
        df = self.data_fetcher.VWAP(df=df) # Output column "VWAP"
        df = self.data_fetcher.ATR(df=df) # Output column "ATR"

        # Implement the strategy (long and short definition)

        # Implement the strategy (long and short definition)
        df['Long_Entry'] = (df["close"].shift(2) < df["VWAP"].shift(2)) & (df["close"].shift(1) > df["VWAP"].shift(1)) # If previous close was below VWAP and now is above
        df["Short_Entry"] = (df["close"].shift(2) > df["VWAP"].shift(2)) & (df["close"].shift(1) < df["VWAP"].shift(1)) # Opposite for short

        # ATR as % of the closing price 
        df["ATR%"] = ( df["ATR"] / df["close"] ) 

        #A Strategy defintion 

        #A.1 Long
        #A.1.1 entry signal -> when the new candle break the previous high
        df['Long_Signal'] = (df['Long_Entry'] == True) & (df['high'] > df['high'].shift(1))
        #A.1.2 Define the entry price 
        df["Long_Trade_Price"] = np.where(df["Long_Signal"], df["high"].shift(1), np.nan) 
        #A.1.3 Define the stop loss 
        df['Long_stop_loss'] = df["low"].shift(1) - ( df["ATR%"] * df["Long_Trade_Price"] ) #Stop loss is below the previous low (as % of ATR) 
        #A.1.4 Define the take profit
        df["Long_take_profit"] = df["Long_Trade_Price"] + (target_RR * (df["Long_Trade_Price"] - df["Long_stop_loss"]))

        #A.2 Short
        #A.2.1 entry signal -> when the new candle break the prevoius high (this in live will change)
        df['Short_Signal'] = (df['Short_Entry'] == True) & (df['low'] < df['low'].shift(1)) 
        #A.2.2 Define the entry price
        df["Short_Trade_Price"] = np.where(df["Short_Signal"], df["low"].shift(1), np.nan)
        #A.2.3 Define the stop loss
        df["Short_stop_loss"] = df["high"].shift(1) + (df["ATR%"] * df["Short_Trade_Price"]) #Stop loss is below the previous low (as % of ATR)
        #A.2.4 Define the take profit
        df["Short_take_profit"] = df["Short_Trade_Price"] - (target_RR * (df["Short_stop_loss"] - df["Short_Trade_Price"]  ) ) 

        #B Trialing P&L 
        #B.1 Define the new columns
        df['Long_PnL_Percentage'] = np.nan
        df['Short_PnL_Percentage'] = np.nan
        df['Cumulative_PnL'] = 0  # To track cumulative P&L
        df["Long_Trade_Active"] = False
        df["Short_Trade_Active"] = False

        # From here to be finished + to be understood ####### 

        #B.2 Long P&L  
        # Loop through the DataFrame to calculate trailing stop and P&L for each row
        for i in range(1, len(df)):
        
            # For Long Trades (entry conditions based on Long_Trade_Price and exit based on Long_stop_loss or Long_take_profit)
            if df['Long_Signal'].iloc[i] == True and not df['Long_Trade_Active'].iloc[i-1]:  # If a new long trade is triggered and no trade was active
                df['Long_Trade_Active'].iloc[i] = True
                long_entry_price = df['Long_Trade_Price'].iloc[i]  # Set the entry price for the trade
                long_stop_loss = df['Long_stop_loss'].iloc[i]  # Set stop loss for the trade
                long_take_profit = df['Long_take_profit'].iloc[i]  # Set take profit for the trade
            
            if df['Long_Trade_Active'].iloc[i] == True:  # If a trade is active, check for exit conditions
                # For Long Trades
                #-- to be fixed #-- There should be close and also another price. Whatever price touches it should close the trade. 
                if df['low'].iloc[i] <= long_stop_loss:  # If the price hits the stop loss
                    long_pnl_percentage = ((long_stop_loss - long_entry_price) / long_entry_price) * 100  # Calculate P&L % for Long
                    df['Long_PnL_Percentage'].iloc[i] = long_pnl_percentage
                    df['Long_Trade_Active'].iloc[i] = False  # Mark the trade as closed
                elif df['high'].iloc[i] >= long_take_profit:  # If the price hits the take profit
                    long_pnl_percentage = ((long_take_profit - long_entry_price) / long_entry_price) * 100  # Calculate P&L % for Long
                    df['Long_PnL_Percentage'].iloc[i] = long_pnl_percentage
                    df['Long_Trade_Active'].iloc[i] = False  # Mark the trade as closed
                else:
                    df["Long_Trade_Active"].iloc[i+1] = True
        
            
        #B.2 Short P&L
            # For Short Trades (entry conditions based on Short_Trade_Price and exit based on Short_stop_loss or Short_take_profit)
            if df['Short_Signal'].iloc[i] == True and not df['Short_Trade_Active'].iloc[i-1]:  # If a new short trade is triggered and no trade was active
                df['Short_Trade_Active'].iloc[i] = True
                short_entry_price = df['Short_Trade_Price'].iloc[i]  # Set the entry price for the trade
                short_stop_loss = df['Short_stop_loss'].iloc[i]  # Set stop loss for the trade
                short_take_profit = df['Short_take_profit'].iloc[i]  # Set take profit for the trade
            
            if df['Short_Trade_Active'].iloc[i] == True:  # If a trade is active, check for exit conditions
                # For Short Trades
                if df['high'].iloc[i] >= short_stop_loss:  # If the price hits the stop loss
                    short_pnl_percentage = ((short_entry_price - short_stop_loss) / short_entry_price) * 100  # Calculate P&L % for Short
                    df['Short_PnL_Percentage'].iloc[i] = short_pnl_percentage
                    df['Short_Trade_Active'].iloc[i] = False  # Mark the trade as closed
                elif df['low'].iloc[i] <= short_take_profit:  # If the price hits the take profit
                    short_pnl_percentage = ((short_entry_price - short_take_profit) / short_entry_price) * 100  # Calculate P&L % for Short
                    df['Short_PnL_Percentage'].iloc[i] = short_pnl_percentage
                    df['Short_Trade_Active'].iloc[i] = False  # Mark the trade as closed
                else:
                    df["Short_Trade_Active"].iloc[i+1] = True
            

            # Update Cumulative P&L (accumulate the P&L percentage for each trade)
            # We need to ensure that only non-NaN P&L values are considered for accumulation.

            df['Cumulative_PnL'].iloc[i] = df['Cumulative_PnL'].iloc[i-1] + (df['Long_PnL_Percentage'].iloc[i] if pd.notna(df['Long_PnL_Percentage'].iloc[i]) else 0) + (df['Short_PnL_Percentage'].iloc[i] if pd.notna(df['Short_PnL_Percentage'].iloc[i]) else 0)


    #End of intro_strtategy()---------------------------------------------------------------------------------------------------------------


#################################### Trading Environment Class ####################################
# #################################### #################################### #################################### # 


import gym # This is the Open AI gym library utilized for Reinforcement Learning environment
from gym import spaces
from stable_baselines3 import PPO # Library to train AI models in reinforcement learning  

class TradinEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    

    The agent can take actions (buy, sell, hold) based on market indicators.
    The goal is to maximize profit while using stop-loss and take-profit levels.
    """
    def __init__(self,df):
        """
        Initialize the trading environment.

        Parameters:
        df (pd.DataFrame): Historical market data with indicators.
        execution_delay = 1
        """
        super(TradinEnv, self).__init__()

        # Store the dataset and set initial step 
        self.df = df
        self.current_step = 0 

        # Define action space: 3 possible actions (0 = Hold, 1 = Buy, 2 = Sell)
        self.action_space = spaces.Discrete(3)

        # Define the observation space: all column of the dataset that I have 
        self.observation_space = spaces.Box(  
            low = np.inf, high=np.inf, shape=( len(df.columns), ), dtype = np.float32 # Meaning it can take any number from all the columns of the df
        )

        # Initialize some trading varaibles (Note: still part of the class, just initializing them)
        self.balance = 1000 # Initial capital
        self.position = None # No active trade 
        self.entry_price = 0 # Price at which a trade was openend 


    def reset(self): 
        """
        Reset the environment at the start of a new episode. This function resets everything 
        when starts a new round 

        Returns:
        np.array: First observation (market indicators at step 0)
        """
        self.current_step = 0        
        self.balance = 1000
        self.position = None
        self.entry_price = 0
        return self._next_observation()

    def _next_observation(self):
        """
        Retrieve the current market state.

        Returns:
        np.array: Market indicators at the current step.
        """
        return self.df.iloc[self.current_step].values
    
    def step(self, action):
        """
        Execute an action (buy, sell, hold) and update the environment.

        Parameters:
        action (int): 0 (Hold), 1 (Buy), 2 (Sell)

        Returns:
        tuple: (new observation, reward, done flag, info)
        """
        current_price = self.df.iloc[self.current_step]["close"]
        entry_long = self.df.iloc[self.current_step].shift(1)["high"]
        entry_short = self.df.iloc[self.current_step].shift(1)["low"]
        atr_percent = self.df.iloc[self.current_step]["ATR"] / self.df.iloc[self.current_step]["close"] # It is used for stop loss and take profit
        previous_low = self.df["low"].shift(1)[self.current_step] # For the stop loss (long) and take profit (short)
        previous_high = self.df["high"].shift(1)[self.current_step] # For the stop loss (short) adn the take profit (long)

        reward = 0 # Default reward
        done = False # Track if episode is finished 

        # Execute action 
        if action == 1: # Buy (long)
            if self.position is None: # If the trade is closed 
                self.position = "long" # Change the position to long 
                self.entry_price = entry_long # Which is the previous high 
                self.stop_loss = previous_low - (atr_percent * self.entry_price)   
                self.take_profit = self.entry_price + (atr_percent * self.entry_price) 
        
        elif action == 2: # Sell (short)
            if self.position is None: # If the trade is closed 
                self.position = "short"
                self.entry_price = entry_short # This is the previous low.  
                self.stop_loss = previous_high + (atr_percent * self.entry_price) # Current price will be the entry price
                self.take_profit = self.entry_price - (atr_percent * self.entry_price)  

        
        # Cheeck if we are in a long or short setting 
        if self.position == "long": 
            if current_price >= self.take_profit: # Current price to be changed with whatever above TP
                reward = 1 # Define the reward for the supervised learning 
                self.position = None # Close the position
                self.balance = self.balance + (self.take_profit - self.entry_price) 
            elif current_price <= self.stop_loss:
                reward = -1 # Penalty for the loss
                self.position = None
                self.balance = self.balance + (self.stop_loss - self.entry_price) 

        elif self.position == "short":
            if current_price <= self.take_profit: # same comment as line 265 (tbd) 
                reward = 1 
                self.position = None
                self.balance = self.balance + (self.entry_price - self.take_profit)
            elif current_price >= self.stop_loss:
                reward = -1 
                self.position = None
                self.balance = self.balance + (self.entry_price - self.stop_loss )

        # Now we move to the next step 
        self.current_step += 1 # current step = current step + 1 
        if self.current_step >= len(self.df) - 1: # Condition to stop the loop
            done = True  

        return self._next_observation(), reward, done, {}
    
        
    def render(self):
        """
        Print the current state of the environment (for debugging).
        """
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")



#################################### Performance evaluation class ####################################
# #################################### #################################### #################################### # 

# to be added

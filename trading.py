import pandas as pd
from binance.client import Client
from data_fetcher import DataFetcher
from utils import get_interval_from_string # To convert the strings in python biannce into human readable
import numpy as np
import matplotlib.pyplot as plt
import gym # This is the Open AI gym library utilized for Reinforcement Learning environment
from gym import spaces
from stable_baselines3 import PPO # Library to train AI models in reinforcement learning  



#################################### Trading Environment Class for RL ####################################
# #################################### #################################### #################################### #


class TradinEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.

    The agent can take actions (buy, sell, hold) based on market indicators.
    The goal is to maximize profit while using stop-loss and take-profit levels.
    """
    def __init__(self, df):
        """
        Initialize the trading environment.

        Parameters:
        df (pd.DataFrame): Historical market data with indicators.
        """
        super(TradinEnv, self).__init__()

        # Store the dataset and set initial step 
        self.df = df
        self.current_step = 1
        self.balance = 1000 # Initial capital
        self.position = None # No active trade   
        self.pending_orders = []  # Store multiple pending orders
        self.execution_delay = 1 # To avoid the agent opening biased trade that look at the past 
        self.RR = 2
        self.reward = self.balance  

        # Define action space: 3 possible actions (0 = Hold, 1 = Buy, 2 = Sell) and "[...]"
        self.action_space = spaces.Discrete(3)
        "[...]"

        # Define the observation space: all columns of the dataset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )   

    def reset(self): 
        """
        Reset the environment at the start of a new episode.

        Returns:
        np.array: First observation (market indicators at step 0)
        """
        self.current_step = 1
        self.balance = 1000
        self.position = None
        self.pending_orders = []
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
        high = self.df.iloc[self.current_step]['high']
        low = self.df.iloc[self.current_step]['low']
        atr_percent = self.df.iloc[self.current_step]["ATR"] / self.df.iloc[self.current_step]["close"]
        previous_high = self.df["high"].shift(1)[self.current_step]
        previous_low = self.df["low"].shift(1)[self.current_step]

        done = False

        # Handle new actions
        if action == 1:  # Buy limit order
            self.position = "long"
            print(f"Step check: Position (t): {self.position}, Action: {action}, Current step: {self.current_step}")
            stop_price = previous_low   
            self._place_order("buy", previous_high, stop_price, atr_percent)

        elif action == 2:  # Sell limit order
            self.position = "short" 
            print(f"Step check: Position (t): {self.position}, Action: {action}, Current step: {self.current_step}")
            stop_price = previous_high
            self._place_order("sell", previous_low, stop_price, atr_percent)


        # Process existing pending orders
        reward = self._execute_pending_orders(high, low) 

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        
        return self._next_observation(), reward, done, {}

    def _place_order(self, order_type, entry_price, stop_price, atr):
        """
        Place a limit order. By using a function, I am able to identify every order which is placed (and therefore executed)
        through the step at which is placed or executed 

        Parameters:
        order_type (str): 'buy' or 'sell'.
        entry_price (float): Price at which the order will be executed.
        """


        # Define a converter to invert the sign of buy and sell 
        if order_type == "buy":
            buy_sell_converter = 1
        elif order_type == "sell":
            buy_sell_converter = -1 
        
        order = {
            "type": order_type,
            "entry_price": entry_price, 
            "step_placed": self.current_step,
            "take_profit": entry_price + ( ( abs(stop_price - entry_price) * self.RR ) * buy_sell_converter ), 
            "stop_loss" :  stop_price - ( (stop_price * atr) * buy_sell_converter )
        }

        self.pending_orders.append(order)


    def _execute_pending_orders(self, high, low):
        """
        Execute any pending limit orders if the price condition is met.

        Parameters:
        current_price (float): Current market price.
        high (float): Current high price.
        low (float): Current low price.
        """
        executed_orders = []
        #reward = 0

        for order in self.pending_orders:
            # Buy orders loop  
            if order["type"] == "buy":
                # Execute pending order (profit)
                if high >= order["take_profit"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay: 
                    executed_orders.append(order)
                    self.balance +=  (self.balance*( (order["take_profit"] - order["entry_price"])/order["entry_price"] ) )
                    self.reward = "[...]"
                    print(f"Executed order : {executed_orders} \n") # For evaluation of the environment 
                # Stop loss trigger (loss)
                elif low <= order["stop_loss"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay:
                    executed_orders.append(order)
                    self.balance += (self.balance*( (order["stop_loss"] - order["entry_price"])/order["entry_price"] ) )
                    self.reward = "[...]"
                    print(f"Executed order : {executed_orders} \n") # For evaluation of the environment

            # Sell order
            if order["type"] == "sell":
                # Execute pending order (profit)
                if low <= order["take_profit"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay: 
                    executed_orders.append(order)
                    self.balance +=  ( self.balance* ( (order["entry_price"] - order["take_profit"])/order["take_profit"] ) )
                    self.reward =  "[...]"
                    print(f"Executed order : {executed_orders} \n") # For evaluation of the environment  
                # Stop loss trigger (loss)
                elif high >= order["stop_loss"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay:
                    executed_orders.append(order)
                    self.balance += (self.balance*( (order["entry_price"] - order["stop_loss"])/order["stop_loss"] ) )
                    self.reward = "[...]"
                    print(f"Executed order : {executed_orders} \n") # For evaluation of the environment
        print(f"Reward: {self.reward}") #TBR
                    
        # Remove executed orders from the list
        self.pending_orders = [o for o in self.pending_orders if o not in executed_orders]
        return self.reward

    def render(self):
        """
        Print the current state of the environment (for debugging).
        """
        print(f"Step: {self.current_step}, Position (t-1): {self.position}, Balance: {self.balance}, Pending Orders: {self.pending_orders}")



#################################### Performance evaluation class ####################################
# #################################### #################################### #################################### # 


class PerformanceEvaluation:
    """
    This class is to evaluate the performance of the RL agent. It contains three main metrics (collected in three functions) which are 
    """
    def __init__(self, env, model, num_episodes = 100):
        # Setting the intial variables
        self.env = env 
        self.model = model 
        self.num_episodes = num_episodes
        self.reward_history = []
        self.balance_history = []

    def run_model(self, env, model): 
        """
        This function is to run the model and visualize the output after the environment has been created and 
        the model has been trained. 
        It should be intended as a function to be run when one don't want to assess the performance
        """
        
        obs = env.reset() # reset function put the state back to the initial state and return the first step of the dataset
        done = False 

        while not done:  # Keep running until done is True 
            action, _ = model.predict(obs) # Prediction using the trained model. Action is buy (1) or sell (2)
            obs, reward, done, _ = env.step(action) # Apply the action in the environment 
            env.render() # Print the state for debugging 
            
## End of run_model --------------------------------------------------------------------------------------------------------------


    def run_evaluation(self, show_plot = False, show_numerics = True):
        """
        This function run the evaluation of the performacne metrics, that include three main measures are: 
        Sample Efficiency: The time that the agent use to ahve a postiive reward. In the numeric format, is just the average of the 
        retun, showing therefore if the reward is postiive or negatvie. 

        Convergence Stability: This is the standard deviation of the reward. In the continuous format (i.e. in the plot) this measure is more indivative of the 
        improvement of the reward in time. The numeric outcome is jsut the average standard deviation

        Cumualtive P&L, this shows the cumulative P&L at the end of the trading process. 
        
        Parameters: 
        show_plot (bool): If true, the function display a plot of the performacne measures 
        show_numerics (bool): If true, thsi function will show just the numeric value of the outcome

        """
    
        obs = self.env.reset() # Resest the observations to the intitial condition (reset contains the function next_observation() ) 
        done = False # Reset the done parameter
    

        while not done: 
            action, _ = self.model.predict(obs) # Start creating the action based ont the collected observation
            obs, reward, done, _ = self.env.step(action) # use the action to predict the next obs, reward and done parameter 
                
            self.reward_history.append(reward) #Total reward is create outsidem and in the 
            self.balance_history.append(self.env.balance) # balance is created inside the env 

        if show_plot is True:
            self.plot_metrics()

        if show_numerics is True: 
            self.display_metrics()

## End of run_evaluation --------------------------------------------------------------------------------------------------------------

    def display_metrics(self):
        """
        Display numerical values for key metrics.
        """
        #Compute the usefule metrics
        sample_efficiency = np.mean(self.reward_history) # Average reward over time 
        convergence_stability = np.std(self.reward_history) # Variation of the rewards
        cumulative_profit_loss = self.balance_history[-1] if self.balance_history else 0 # return the value if the balance is non empty 

        # Print the metrics
        print("Performance Metrics:")
        print(f"Sample Efficiency: {sample_efficiency}")
        print(f"Convergence Stability: {convergence_stability}")
        print(f"Cumulative P&L: {cumulative_profit_loss}")

    def plot_metrics(self):
        episodes = np.arange(1, len(self.reward_history) + 1) # Create an array [1,2,3, ... len()+1]
        plt.figure( figsize = (12,4) )

        # Plot the sample efficiency (imrpovement in rewards over time i.e. is the agent learning?)
        plt.subplot(1,3,1)
        plt.plot(episodes, np.cumsum(self.reward_history)/episodes, label = "Sample Efficiency") # Plot the cumulated reward over the time 
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Reward / Episode")
        plt.title("Sample Efficiency")
        plt.legend()

        # Convergence Stabiliity (variance in reward over time)
        plt.subplot(1,3,2)
        rolling_std = [np.std(self.reward_history[max(0, i-10):i+1]) for i in range(len(self.reward_history))]
        # Exaple: When i = 5, the slice is self.reward_history[max(0, 5-10):5+1], which results in self.reward_history[0:6] (i.e., the first 6 elements).
        plt.plot(episodes, rolling_std, label = "Reward Variance")
        plt.xlabel("Episodes")
        plt.ylabel("Variance")
        plt.title("Convergence Stability")
        plt.legend()

        # Cumulative P&L (Final Balance over episodes)
        plt.subplot(1,3,3)
        plt.plot(episodes, self.balance_history, label = "Profit & Loss", color = "red")
        plt.xlabel("Episodes")
        plt.ylabel("Final Balance")
        plt.title("Cumulative P&L")
        plt.legend()

        plt.tight_layout()
        plt.show()

## End ofdisplay metrics  --------------------------------------------------------------------------------------------------------------




#################################### Class to try strategies (just a try) ####################################
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
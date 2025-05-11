import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO # Library to train AI models in reinforcement learning  
import matplotlib.pyplot as plt

class TradinEnv_v1(gym.Env):
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
        super(TradinEnv_v1, self).__init__()

        # Store the dataset and set initial step 
        self.df = df
        self.current_step = 1
        self.balance = 1000 # Initial capital
        self.position = None # No active trade   
        self.pending_orders = []  # Store multiple pending orders
        self.execution_delay = 1 # To avoid the agent opening biased trade that look at the past 
        self.RR = 2

        # Define action space: 3 possible actions (0 = Hold, 1 = Buy, 2 = Sell) and 5 possible RR (0 to 5)
        self.action_space = spaces.Discrete(3)

        # Define the observation space: all columns of the dataset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        ) # Should be changed to incorporate the RR ?  

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
        current_price = self.df.iloc[self.current_step]['close'] #TBR
        high = self.df.iloc[self.current_step]['high']
        low = self.df.iloc[self.current_step]['low']
        atr_percent = self.df.iloc[self.current_step]["ATR"] / self.df.iloc[self.current_step]["close"]
        previous_high = self.df["high"].shift(1)[self.current_step]
        previous_low = self.df["low"].shift(1)[self.current_step]

        reward = 0
        done = False

        # Handle new actions
        if action == 1:  # Buy limit order
            self.position = "long"
            print(f"Step check: Position (t): {self.position}, Action: {action}, Current step: {self.current_step}")#TBR
            stop_price = previous_low   
            self._place_order("buy", previous_high, stop_price, atr_percent)

        elif action == 2:  # Sell limit order
            self.position = "short" 
            print(f"Step check: Position (t): {self.position}, Action: {action}, Current step: {self.current_step}")#TBR
            stop_price = previous_high
            self._place_order("sell", previous_low, stop_price, atr_percent)


        # Process existing pending orders
        self._execute_pending_orders(high, low)

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
        # Define the variable 
        self.pending_order = None

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

        for order in self.pending_orders:
            # Buy orders loop  
            if order["type"] == "buy":
                # Execute pending order (profit)
                if high >= order["take_profit"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay: 
                    executed_orders.append(order)
                    self.balance +=  (self.balance*( (order["take_profit"] - order["entry_price"])/order["take_profit"] ) ) # this is wrong
                    print(f"Executed order : {executed_orders} \n")#TBR   
                # Stop loss trigger (loss)
                elif low <= order["stop_loss"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay:
                    executed_orders.append(order)
                    self.balance += (self.balance*( (order["stop_loss"] - order["entry_price"])/order["stop_loss"] ) ) # this is wrong
                    print(f"Executed order : {executed_orders} \n")#TBR

            # Sell order
            elif order["type"] == "sell":
                # Execute pending order (profit)
                if low <= order["take_profit"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay: 
                    executed_orders.append(order)
                    self.balance +=  ( self.balance* ( (order["entry_price"] - order["take_profit"])/order["entry_price"] ) )
                    print(f"Executed order : {executed_orders} \n")#TBR   
                # Stop loss trigger (loss)
                elif high >= order["stop_loss"] and abs(order["step_placed"] - self.current_step) >= self.execution_delay:
                    executed_orders.append(order)
                    self.balance += (self.balance*( (order["entry_price"] - order["stop_loss"])/order["entry_price"] ) )
                    print(f"Executed order : {executed_orders} \n")#TBR
                    
        # Remove executed orders from the list
        self.pending_orders = [o for o in self.pending_orders if o not in executed_orders]
        

    def render(self):
        """
        Print the current state of the environment (for debugging).
        """
        print(f"Step: {self.current_step}, Position (t-1): {self.position}, Balance: {self.balance}, Pending Orders: {self.pending_orders}")

# This class compared to the older one adds the element of "future", meaning that the order instead of being
# immediately executed, is opened as a pending order, and then executed at a later time. This code introduces
# limit orders by introducing the term self.pending_order["step_placed"] = self.current_step. This code stores the
# current step in a given place.


# From here there is the perofmrance evaluation of the 






# Creating a new class for performance evaluation 


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

    def run_evaluation(self, first_run = True, show_plot = False, show_numerics = True):
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
        episodes = np.arrange(1, len(self.reward_history) + 1) # Create an array [1,2,3, ... len()+1]
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



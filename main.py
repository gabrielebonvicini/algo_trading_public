"""
Algo strategy main tab. 
"""

#####################################
# Section 0: import dependencies 
#####################################
import binance 
import pandas as pd 
import numpy as np
import matplotlib
from data_fetcher import DataFetcher # To fetch historical data
from binance.client import Client # Retrive the API from binance 
from stable_baselines3 import PPO # A Library to train AI Models in RL & PPO popular algorithm
                                  # used for training AI agents in trading
from trading import TradinEnv # To create the trading environment
from utils import train_test_split

#####################################
# Section 1: set-up the API
#####################################


#Steps: 
 #Binance > API management > Algo_project
API_key = "[...]"
API_secret = "[...]"

#Activate the client 
client = Client(API_key,API_secret)


#####################################
# Section 2: Fetch historical data with the class
#####################################

# Variable assignation
symbol = "BTCUSDT"

# Interval = "1DAY" #Fetch 15 minute data
start_date =  "15 Jan, 2022"
end_date = "16 Feb, 2025"

DataFetcher_BTC = DataFetcher(client, symbol)

# In this case creates two datasets
#df1 = DataFetcher_BTC.fetch_data(interval = "1DAY", start_date = start_date, end_date = end_date)
df = DataFetcher_BTC.fetch_data(interval = "1MIN", start_date = start_date, end_date = end_date)

#####################################
# Section 3: Initialize the model 
#####################################

# Divide the dataset into train and test set 
df_train, df_test = train_test_split(df, test_size = 0.3)

# Create the environment for training and test 
training_env = TradinEnv(df_train) 
test_env = TradinEnv(df_test)

# Create the model 
model = PPO("MlpPolicy", training_env, verbose = 1) # PPO model to train the AI 
# MlpPolicy is a multi layer perceptron to learn (neural net)
# verbose to enable the logs while trainig. 0 to switch off

# Save the model 
model.save("trading_ppo_model")


#####################################
# Section 4: Backtest the model  
#####################################

# Load the model 
model = PPO.load("trading_ppo_model")

# Reset the environment for testing 
obs = test_env.reset()
done = False 

while not done:  # Keep running until done is True 
    action, _ = model.predict(obs) # Prediction using the trained model 
    obs, reward, done, _ = test_env.step(action) # Apply the action in the environment 
    test_env.render() # Print the state for debugging 


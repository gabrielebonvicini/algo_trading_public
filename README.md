# algo_trading

**Reinforcement Learning for Trading (Public Version)**

This repository contains the public version of an ongoing project that applies reinforcement learning (RL) to algorithmic trading. Currently, the focus is on developing the trading environment, particularly the reward function.

Parts of the original code that are not intended for public sharing have been removed and replaced with "[...]" as placeholders. The full project also includes a detailed memorandum documenting all functions and classes, as well as an archive of experimental results and notes.

---

## Project Status

This project is **under active development**.  
The current phase involves:
- Building the RL environment
- Designing and refining the reward function
- Experimenting with discrete action spaces

---

## Project Description

The aim of this project is to apply reinforcement learning techniques to financial trading. The environment is built with a **discrete action space** that includes:
- **Buy**
- **Sell**
- **Hold**

Additionally, the agent can choose from **five different target risk-reward (RR) ratios**, optimizing its decisions based on past trades. This RR optimization mechanism has been omitted in the public version.

At present, the dataset used for training contains a **limited set of trading indicators**. This is intentional, as the primary focus is on reward function development before expanding the data inputs.

The algorithmic trading logic and integration with a broker or execution layer will be developed once the model shows reliable performance. The complete project will eventually include:
- Performance monitoring
- Model validation
- Realistic backtesting

---

## Notes

- Certain sections have been intentionally excluded or redacted for confidentiality.
- This public version is intended for educational or collaborative purposes and may not be fully functional until the core logic is finalized.

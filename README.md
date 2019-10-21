# Black Scholes Neural Network GUI

## Credits

I got the idea to develop this gui after briefly reading the paper "Pricing options and computing implied volatilities
using neural networks" by Shuaiqiang Liu, Cornelis W. Oosterlee, and Sander M.Bohte. This paper may be found on 

https://https://arxiv.org/pdf/1901.08943.pdf
 

## Overview

The user is able to input a set of parameters which create a pandas dataframe containing values which are plugged into the Black Scholes formula. With this dataframe, the user is able to input the % of data will be used to test. The model will
learn to mimick the black scholes formula over time and will plot the weight matrices/vector and dollar error in the respective graphs programmed into the gui.


## Neural Network Diagram


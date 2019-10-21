# Black Scholes Neural Network GUI

## Credits

I got the idea to develop this gui after briefly reading the paper "Pricing options and computing implied volatilities
using neural networks" by Shuaiqiang Liu, Cornelis W. Oosterlee, and Sander M.Bohte. This paper may be found on 

https://arxiv.org/pdf/1901.08943.pdf
 

## Overview

The user is able to input a set of parameters which create a pandas dataframe containing values which are plugged into the Black Scholes formula. With this dataframe, the user is able to input the % of data will be used to test. The model will
learn to mimick the black scholes formula over time and will plot the weight matrices/vector and dollar error in the respective graphs programmed into the gui.

## Black Scholes Equation w/ Dividends (used in this GUI)


![Equation](https://latex.codecogs.com/gif.latex?BS_%7Bcall%7D%20%3D%20Se%5E%7B-qt%7DN%28d_1%29-Ke%5E%7B-rt%7DN%28d_2%29)
<br/><br/>
![Equation](https://latex.codecogs.com/gif.latex?BS_%7Bput%7D%20%3D%20Ke%5E%7B-rt%7DN%28-d_2%29%20-%20Se%5E%7B-qt%7DN%28-d_1%29)

## Neural Network Diagram

![Screenshot](https://github.com/MoSharieff/BSNeuralNet/blob/master/images/nnet.png)

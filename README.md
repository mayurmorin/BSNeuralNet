# Black Scholes Neural Network GUI

## Credits

I got the idea to develop this gui after briefly reading the paper "Pricing options and computing implied volatilities
using neural networks" by Shuaiqiang Liu, Cornelis W. Oosterlee, and Sander M.Bohte (2018, September). This paper may be found on 

https://arxiv.org/pdf/1901.08943.pdf

Our methods are different, with my main objective to not have the best model, but rather visualize weights being adjusted as training is taking place.
 

## Overview

This gui allows you to visualize a neural networks weights being adjusted while it is training to learn option prices. There are three groups of weights in this model, with the sizes being (7x4), (4x3), & (3x1). The graph is drawn in "Neural Network Diagram"

The user is able to select a default dataset, or input their own parameters. Once the model is trained, the user can test the model and a graph of the absolute value dollar error between the actual option prices and predicted option prices will display. A bit of an explination. Epoch rate is the percentage of epochs I choose to skip to allow more accurate normalized minimum and maximum bound values. 

## Black Scholes Equation w/ Dividends (used in this GUI)


![Equation](https://latex.codecogs.com/gif.latex?BS_%7Bcall%7D%20%3D%20Se%5E%7B-qt%7DN%28d_1%29-Ke%5E%7B-rt%7DN%28d_2%29)
<br/><br/>
![Equation](https://latex.codecogs.com/gif.latex?BS_%7Bput%7D%20%3D%20Ke%5E%7B-rt%7DN%28-d_2%29%20-%20Se%5E%7B-qt%7DN%28-d_1%29)

## Neural Network Diagram

![Screenshot](https://github.com/MoSharieff/BSNeuralNet/blob/master/images/nnet.png)

## Running Example

In order to run this GUI, execute
```sh
> python3 main.py
```

![Alt](https://github.com/MoSharieff/BSNeuralNet/blob/master/images/demo.gif)

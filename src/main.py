'''
    Black Scholes Neural Network
    Author: Mo Sharieff

    This gui allows a user to input parameters to create a simulated dataset containing
    variables to be plugged into the Black Scholes function. A nerual network trains on the
    created dataset to best configure itself to price options. The visual board displays
    three graphs at the top to show the weights in the neural net change as the training
    is taking place. The bottom portion contains a graph of the dollar error between the
    predicted values and the actual option prices. The other portion is the control
    panel which takes your inputs.

'''


import numpy as np
import pandas as pd
import random as rd

import tkinter as tk
import tkinter.ttk as ttk

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Contains all of your lambda functions and defined variables
class items:

    resolution = lambda self, x, y:  '{}x{}'.format(x, y)
    variables = ('Stock', 'Strike', 'RiskFree', 'Dividend', 'Volatility', 'Maturity', 'Type')
    graph_fig = (4, 3.5)
    
# Contains the black scholes equation solver
class blackscholes:

    # Calculates Probability of D1 & D2 using simpsons integration
    def N(self, x, n=1111):
        s = lambda n: [1 if i == 0 or i == n - 1 else 2 if i % 2 == 0 else 4 for i in range(n)]
        f = lambda x: np.exp(-x**2/2.0)/np.sqrt(2.0*np.pi)
        a = -40
        dx = (x - a) / (n - 1)
        Total = np.sum([c*f(a + i * dx) for i, c in enumerate(s(n))])
        return round(dx/3.0 * Total, 4)

    # Solves for D1 & D2
    def d(self, S, K, r, q, v, t):
        neu = np.log(S/K) + (r - q + 0.5*v**2/2.0)*t
        den = v*np.sqrt(t)
        return neu / den, neu / den - den

    # Solves for a call or put option price
    def BS(self, S, K, r, q, v, t, op):
        d1, d2 = self.d(S, K, r, q, v, t)
        if op == 'call':
            return S * np.exp(-q*t) * self.N(d1) - K * np.exp(-r*t) * self.N(d2)
        else:
            return K * np.exp(-r*t) * self.N(-d2) - S * np.exp(-q*t) * self.N(-d1)

    # Creates a pandas frame with your dataset containing Black Scholes input variables
    def BSDataset(self, S=(), K=(), r=(), q=(), v=(), m=(), size=1000):
        vh = []
        for i in range(size):
            stock_price = rd.randint(int(S[0]*100), int(S[1]*100))/100
            if rd.randint(0, 1) == 0:
                op_type = 'call'
                strike_price = round(stock_price + rd.randint(K[0], K[1]), 0)
            else:
                op_type = 'put'
                strike_price = round(stock_price - rd.randint(K[0], K[1]), 0)
            risk_free = rd.randint(int(r[0]*1000), int(r[1]*1000))/1000
            dividend = rd.randint(int(q[0]*1000), int(q[1]*1000))/1000
            volatility = rd.randint(int(v[0]*100), int(v[1]*100))/100
            maturity = rd.randint(m[0], m[1])/12
            vh.append([stock_price, strike_price, risk_free, dividend, volatility, maturity, op_type])
        return pd.DataFrame(data=vh, columns=self.variables)


# Contains various command functions
class gridcmds: 

    # Translates your inputs to a full fledged dataset
    def update_dataset(self):
        IP = self.IP
        self.bs = self.BSDataset(S=(float(IP['Stock']['down'].get()), float(IP['Stock']['up'].get())),
                               K=(float(IP['Strike']['down'].get()), float(IP['Strike']['up'].get())),
                               r=(float(IP['RiskFree']['down'].get()), float(IP['RiskFree']['up'].get())),
                               q=(float(IP['Dividend']['down'].get()), float(IP['Dividend']['up'].get())),
                               v=(float(IP['Volatility']['down'].get()), float(IP['Volatility']['up'].get())),
                               m=(int(IP['Maturity']['down'].get()), int(IP['Maturity']['up'].get())),
                               size=int(self.rows.get()))
        N = len(self.bs['Stock'])
        M = int(float(self.ttr_input.get())/100 * N)
        self.trainset = self.bs[:M]
        self.testset = self.bs[M:]
        self.epoch_rate = float(self.epr.get()) / 100
        print(self.trainset, '\n\n', self.testset)


# Contains graph functions
class graphs:

    # Graph of your first weight matrix
    def weightOne(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW1 = FigureCanvasTkAgg(fig, frame)
        self.pltW1 = fig.add_subplot(111, projection='3d')
        tk.Label(frame, text='Weight Matrix (1)').grid(row=1, column=1)
        self.cvW1.get_tk_widget().grid(row=2, column=1)

    # Graph of your second weight matrix
    def weightTwo(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW2 = FigureCanvasTkAgg(fig, frame)
        self.pltW2 = fig.add_subplot(111, projection='3d')
        tk.Label(frame, text='Weight Matrix (2)').grid(row=1, column=2)
        self.cvW2.get_tk_widget().grid(row=2, column=2)

    # Graph of your third weight matrix
    def weightThree(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW3 = FigureCanvasTkAgg(fig, frame)
        self.pltW3 = fig.add_subplot(111)
        tk.Label(frame, text='Weight Matrix (3)').grid(row=1, column=3)
        self.cvW3.get_tk_widget().grid(row=2, column=3)

    # Graph of your dollar error
    def errorGraph(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.errCV = FigureCanvasTkAgg(fig, frame)
        self.pltErr = fig.add_subplot(111)
        tk.Label(frame, text='Pricing Error').grid(row=3, column=1)
        self.errCV.get_tk_widget().grid(row=4, column=1)


# Contains your neural network functions
class neural:

    # Sigmoid function w/ derivative option
    def sigmoid(x, d=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if d:
            return f * (1.0 - f)
        else:
            return f

    # This function trains your model weights
    def trainModel(self):
        
        for epoch, (S, K, r, q, v, t, op) in enumerate(self.trainset.values):
            # Normalization

           if epoch < int(self.epoch_rate * len(self.trainset.values)):
               print('hehe shits and giggles')
           else:
               print(S, r, v)

    # This function tests your models strength
    def testModel(self):
        pass
                
    
# Your main class, the center of everything. The first step is to inherit
# all of the subclasses
class gridboard(tk.Tk,
                                    items,
                                    blackscholes,
                                    gridcmds,
                                    graphs,
                                    neural):

    # Initialize your main GUI
    def __init__(self):
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, 'Black Scholes Neural Net')
        self.geometry(self.resolution(1300, 820))        

        # Initialize graph frame
        graphFrame = tk.Frame(self)
        graphFrame.pack(side=tk.TOP)
        self.weightOne(graphFrame)
        self.weightTwo(graphFrame)
        self.weightThree(graphFrame)
        self.errorGraph(graphFrame)

        # Initialize control frame
        controlFrame = tk.Frame(graphFrame)
        controlFrame.grid(row=4, column=2)
        self.control_frame(controlFrame)

        # Initalize train frame
        trainFrame = tk.Frame(graphFrame)
        trainFrame.grid(row=4, column=3)
        self.train_frame(trainFrame)

    # Holds the training button
    def train_frame(self, frame):
        tk.Button(frame, text='Train Model', command=lambda: self.trainModel()).grid(row=1, column=1)

    # Holds the control panel frame which contains inputs for training and testing ratio, dataset size, and min/max values of inputs
    def control_frame(self, gframe):
        frame = tk.Frame(gframe)
        dual = tk.Frame(gframe)
        
        tk.Label(frame, text='Train/Test Ratio: ').grid(row=1, column=1)
        self.ttr_input = ttk.Entry(frame, width=5, justify='center')
        self.ttr_input.grid(row=1, column=2)
        tk.Label(frame, text='%').grid(row=1, column=3)
        tk.Label(frame, text='  Rows').grid(row=1, column=4)
        self.rows = ttk.Entry(frame, width=7, justify='center')
        self.rows.grid(row=1, column=5)
        tk.Label(frame, text='  ').grid(row=1, column=6)
        tk.Button(frame, text='Update DataSet', command=lambda: self.update_dataset()).grid(row=1, column=7)
        tk.Label(frame, text='Epoch Rate: ').grid(row=2, column=1)
        self.epr = ttk.Entry(frame, width=5, justify='center')
        self.epr.grid(row=2, column=2)
        tk.Label(frame, text='%').grid(row=2, column=3)
        frame.pack(side=tk.BOTTOM)

        tk.Label(dual, text=' ').grid(row=1, column=1)
        tk.Label(dual, text='Min').grid(row=2, column=1)
        tk.Label(dual, text='Max').grid(row=3, column=1)
        self.IP = {}
        for ii, vv in enumerate(self.variables):
            if vv != "Type":
                tk.Label(dual, text=vv).grid(row=1, column=2 + ii)
                up_inp = ttk.Entry(dual, width=6, justify='center')
                
                dn_inp = ttk.Entry(dual, width=6, justify='center')
                dn_inp.grid(row=2, column=2 + ii)
                up_inp.grid(row=3, column=2 + ii)
                
                self.IP[vv] = {'up': up_inp, 'down': dn_inp}
        tk.Label(dual, text=' ').grid(row=4, column=1)
        dual.pack(side=tk.TOP)
                                               

# Run the gui
gridboard().mainloop()

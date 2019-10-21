"""
    Black Scholes Neural Network
    Author: Mo Sharieff

    This gui allows a user to input parameters to create a simulated dataset containing
    variables to be plugged into the Black Scholes function. A nerual network trains on the
    created dataset to best configure itself to price options. The visual board displays
    three graphs at the top to show the weights in the neural net change as the training
    is taking place. The bottom portion contains a graph of the dollar error between the
    predicted values and the actual option prices. The other portion is the control
    panel which takes your inputs.

"""


import numpy as np
import pandas as pd
import random as rd
import time

import tkinter as tk
import tkinter.ttk as ttk

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Contains all of your lambda functions and defined variables
class items:

    resolution = lambda self, x, y:  "{}x{}".format(x, y)
    variables = ("Stock", "Strike", "RiskFree", "Dividend", "Volatility", "Maturity", "Type")
    graph_fig = (4, 3.5)

    num_line = lambda self, n: [i+1 for i in range(n)]
    lister = lambda self, x: [i for i in x]

# Contains the black scholes equation solver
class blackscholes:

    # Calculates Probability of D1 & D2 using simpsons integration
    def N(self, x, n=171):
        s = lambda n: [1 if i == 0 or i == n - 1 else 2 if i % 2 == 0 else 4 for i in range(n)]
        f = lambda x: np.exp(-x**2/2.0)/np.sqrt(2.0*np.pi)
        a = round(x - 11, 0)
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
        if op == "call":
            return S * np.exp(-q*t) * self.N(d1) - K * np.exp(-r*t) * self.N(d2)
        else:
            return K * np.exp(-r*t) * self.N(-d2) - S * np.exp(-q*t) * self.N(-d1)



    # Creates a pandas frame with your dataset containing Black Scholes input variables
    def BSDataset(self, S=(), K=(), r=(), q=(), v=(), m=(), size=1000):
        vh = []
        for i in range(size):
            stock_price = rd.randint(int(S[0]*100), int(S[1]*100))/100
            if rd.randint(0, 1) == 0:
                op_type = "call"
                strike_price = round(stock_price + rd.randint(K[0], K[1]), 0)
            else:
                op_type = "put"
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
    def update_dataset(self, test=False):
        IP = self.IP
        if test:
            self.bs = self.BSDataset(S=(114.33, 167.55),
                                     K=(10, 20),
                                     r=(0.05, 0.08),
                                     q=(0.01, 0.04),
                                     v=(0.25, 0.61),
                                     m=(3, 9),
                                     size=200)
            N = len(self.bs["Stock"])
            M = int(float(0.8*N))
            self.trainset = self.bs[:M]
            self.testset = self.bs[M:]
            self.epoch_rate = 0.01
            self.error_level = 0.00001

        else:
            self.bs = self.BSDataset(S=(float(IP["Stock"]["down"].get()), float(IP["Stock"]["up"].get())),
                                   K=(float(IP["Strike"]["down"].get()), float(IP["Strike"]["up"].get())),
                                   r=(float(IP["RiskFree"]["down"].get()), float(IP["RiskFree"]["up"].get())),
                                   q=(float(IP["Dividend"]["down"].get()), float(IP["Dividend"]["up"].get())),
                                   v=(float(IP["Volatility"]["down"].get()), float(IP["Volatility"]["up"].get())),
                                   m=(int(IP["Maturity"]["down"].get()), int(IP["Maturity"]["up"].get())),
                                   size=int(self.rows.get()))
            N = len(self.bs["Stock"])
            M = int(float(self.ttr_input.get())/100 * N)
            self.trainset = self.bs[:M]
            self.testset = self.bs[M:]
            self.epoch_rate = float(self.epr.get()) / 100
            self.error_level = float(self.err_lvl.get())
        print("Dataset Refreshed")


# Contains graph functions
class graphs:

    # Graph of your first weight matrix
    def weightOne(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW1 = FigureCanvasTkAgg(fig, frame)
        self.pltW1 = fig.add_subplot(111, projection="3d")
        tk.Label(frame, text="Weight Matrix (1)").grid(row=1, column=1)
        self.cvW1.get_tk_widget().grid(row=2, column=1)

    # Graph of your second weight matrix
    def weightTwo(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW2 = FigureCanvasTkAgg(fig, frame)
        self.pltW2 = fig.add_subplot(111, projection="3d")
        tk.Label(frame, text="Weight Matrix (2)").grid(row=1, column=2)
        self.cvW2.get_tk_widget().grid(row=2, column=2)

    # Graph of your third weight matrix
    def weightThree(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.cvW3 = FigureCanvasTkAgg(fig, frame)
        self.pltW3 = fig.add_subplot(111)
        tk.Label(frame, text="Weight Matrix (3)").grid(row=1, column=3)
        self.cvW3.get_tk_widget().grid(row=2, column=3)

    # Graph of your dollar error
    def errorGraph(self, frame):
        fig = Figure(figsize=self.graph_fig, dpi=100)
        self.errCV = FigureCanvasTkAgg(fig, frame)
        self.pltErr = fig.add_subplot(111)
        tk.Label(frame, text="Pricing Error ($)").grid(row=3, column=1)
        self.errCV.get_tk_widget().grid(row=4, column=1)

    # Plots your weights
    def plotWeights(self, w1, w2, w3):
        s0, d0 = w1.shape
        s1, d1 = w2.shape
        s2 = len(w3)
        x, y = np.meshgrid(self.num_line(d0), self.num_line(s0))
        z = np.array(w1)
        self.pltW1.plot_surface(x, y, z, cmap='jet', edgecolor='black', linewidth=0.4, alpha=0.6)
        self.cvW1.draw()
        x, y = np.meshgrid(self.num_line(d1), self.num_line(s1))
        z = np.array(w2)
        self.pltW2.plot_surface(x, y, z, cmap='jet', edgecolor='black', linewidth=0.4, alpha=0.6)
        self.cvW2.draw()
        y = [float(f[0]) for f in w3]
        x = self.num_line(len(y))
        self.pltW3.bar(x, y, color='purple', alpha=0.6, edgecolor='black', linewidth=0.4)
        self.cvW3.draw()
        time.sleep(0.0001) # Makes sure plotter renders with a split break

    # Plots your dollar error
    def plotDollar(self, y):
        x = self.num_line(len(y))
        self.pltErr.cla()
        self.pltErr.plot(x, y, color='limegreen')
        for xx, yy in zip(x, y):
            if yy >= -self.error_level and yy <= self.error_level:
                self.pltErr.scatter(xx, yy, color='red')
        self.errCV.draw()
        time.sleep(0.0001) # Makes sure plotter renders with a split break

    def clearBoard(self):
        for plot in (self.pltW1, self.pltW2, self.pltW3, self.pltErr):
            plot.cla()
        for canvas in (self.cvW1, self.cvW2, self.cvW3, self.errCV):
            canvas.draw()

# Contains your neural network functions
class neural:

    # Unit conversion class
    class modeler:

        def __init__(self, parent):
            self.hold_vars = {i:[] for i in parent.variables if i != "Type"}
            self.hold_output = []
            self.out_stats = {'min': 0, 'max': 0}
            self.norm_stats = {i:{"min": None, "max": None} for i in parent.variables if i != "Type"}

            self.dollar_error = []

        # Deposit current data inputs for normalization
        def __call__(self, deposit, outvals):
            for i, j in deposit.items():
                self.hold_vars[i].append(j)
            self.hold_output.append(outvals)
            self.min_max()

        # Calculate the min and max of each variable
        def min_max(self):
            for i in self.hold_vars:
                self.norm_stats[i]["min"] = np.min(self.hold_vars[i])
                self.norm_stats[i]["max"] = np.max(self.hold_vars[i])
            self.out_stats["min"] = np.min(self.hold_output)
            self.out_stats["max"] = np.max(self.hold_output)

        # Use a formula to return a value between 0 & 1
        def normalize(self, deposit, outvals, optype):
            params = []
            for i in deposit:
                params.append([(deposit[i] - self.norm_stats[i]["min"]) / (self.norm_stats[i]["max"] - self.norm_stats[i]["min"])])
            params.append([0 if optype == 'call' else 1])
            result_val = (outvals - self.out_stats["min"]) / (self.out_stats["max"] - self.out_stats["min"])
            return np.array(params), result_val

        #def regularize(self, tag, # LEFT OFF HERE


    # Sigmoid function w/ derivative option
    def sigmoid(self, x, d=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if d:
            return f * (1.0 - f)
        else:
            return f

    # This function trains your model weights
    def trainModel(self):

        # Weights will be stored here for testing later
        #self.WEIGHTS = {'W1': None, 'W2': None, 'W3' None}

        self.model = self.modeler(self)

        def clearPlots():
            # Every weight plot is cleared on every training epoch
            for plot in (self.pltW1, self.pltW2, self.pltW3, self.pltErr):
                plot.cla()
            self.errCV.draw()

        # Define our weight arrays
        W1 = np.array([[rd.random() for j in range(4)] for i in range(7)])
        W2 = np.array([[rd.random() for j in range(3)] for i in range(4)])
        W3 = np.array([[rd.random()] for j in range(3)])

        try:
            # Training loop goes through our entire training set
            for epoch, (S, K, r, q, v, t, op) in enumerate(self.trainset.values):
                # Normalization of variables
                items = {"Stock": S, "Strike": K, "RiskFree": r, "Dividend": q, "Volatility": v, "Maturity": t}
                actual_price = self.BS(S, K, r, q, v, t, op)

                self.model(items, actual_price)
                INPUT, OUTPUT = self.model.normalize(items, actual_price, op)

                if epoch % 5 == 0:
                    clearPlots()

                if epoch < int(self.epoch_rate * len(self.trainset.values)):
                    pass # Gather some data in the beginning to normalzie effectively
                else:

                    # BEING TRAINING MODEL

                    self.epoch_meter.configure(text='Training: {} Epochs Left'.format(len(self.trainset.values) - epoch - 1))

                    # I use absolute values for safety, I have had this glitch
                    _err1 = [[100]]
                    while abs(_err1[0][0]) > self.error_level:

                        # Compute Layer 1
                        X1 = W1.transpose().dot(INPUT)
                        L1 = self.sigmoid(X1)

                        # Compute Layer 2
                        X2 = W2.transpose().dot(L1)
                        L2 = self.sigmoid(X2)

                        # Compute Layer 3
                        X3 = W3.transpose().dot(L2)
                        EST_OUTPUT = self.sigmoid(X3)

                        _err1 = (OUTPUT - EST_OUTPUT)**2
                        _dw1 = 2*(OUTPUT - EST_OUTPUT)*self.sigmoid(X3, d=True)

                        _err2 = W3.dot(_dw1)
                        _dw2 = _err2 * self.sigmoid(X2, d=True)

                        _err3 = W2.dot(_dw2)
                        _dw3 = _err3 * self.sigmoid(X1, d=True)

                        W1 += INPUT.dot(_dw3.transpose())
                        W2 += L1.dot(_dw2.transpose())
                        W3 += L2.dot(_dw1.transpose())


                # Plot weights
                if epoch % 4 == 0: clearPlots()
                self.plotWeights(W1, W2, W3)

                print('Training Epoch: ', epoch + 1)
        except Exception as e:
            print("Training Error: ", e)

        self.WEIGHTS = {}
        for ii, jj in zip(("W1", "W2", "W3"), (W1, W2, W3)):
            self.WEIGHTS[ii] = jj

        self.epoch_meter.configure(text="Ready to test model")

    # This function tests your models strength
    def testModel(self):

        for epoch, (S, K, r, q, v, t, op) in enumerate(self.testset.values):
            self.epoch_meter.configure(text='Testing: {} Epochs Left'.format(len(self.testset.values) - epoch - 1))

            items = {"Stock": S, "Strike": K, "RiskFree": r, "Dividend": q, "Volatility": v, "Maturity": t}
            actual_price = self.BS(S, K, r, q, v, t, op)

            self.model(items, actual_price)
            INPUT, OUTPUT = self.model.normalize(items, actual_price, op)

            X1 = self.WEIGHTS["W1"].transpose().dot(INPUT)
            L1 = self.sigmoid(X1)

            X2 = self.WEIGHTS["W2"].transpose().dot(L1)
            L2 = self.sigmoid(X2)

            X3 = self.WEIGHTS["W3"].transpose().dot(L2)
            EST_OUTPUT = self.sigmoid(X3)

            estimated_price = EST_OUTPUT[0][0] * (self.model.out_stats["max"] - self.model.out_stats["min"]) + self.model.out_stats["min"]
            self.model.dollar_error.append(actual_price - estimated_price)

            self.plotDollar(self.model.dollar_error)

        self.epoch_meter.configure(text=".....")
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
        tk.Tk.wm_title(self, "Black Scholes Neural Net")
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

    # Holds the training and testing button
    def train_frame(self, frame):
        tk.Button(frame, text="Train Model", command=lambda: self.trainModel()).grid(row=1, column=1)
        tk.Button(frame, text="Test Model", command=lambda: self.testModel()).grid(row=2, column=1)
        tk.Button(frame, text="Clear Graphs", command=lambda: self.clearBoard()).grid(row=3, column=1)
        self.epoch_meter = tk.Label(frame, text='.......')
        tk.Label(frame, text='\t').grid(row=4, column=1)
        self.epoch_meter.grid(row=5, column=1)

    # Holds the control panel frame which contains inputs for training and testing ratio, dataset size, and min/max values of inputs
    def control_frame(self, gframe):
        frame = tk.Frame(gframe)
        dual = tk.Frame(gframe)

        tk.Label(frame, text="Train/Test Ratio: ").grid(row=1, column=1)
        self.ttr_input = ttk.Entry(frame, width=5, justify="center")
        self.ttr_input.grid(row=1, column=2)
        tk.Label(frame, text="%").grid(row=1, column=3)
        tk.Label(frame, text="  Rows: ").grid(row=1, column=4)
        self.rows = ttk.Entry(frame, width=7, justify="center")
        self.rows.grid(row=1, column=5)
        tk.Label(frame, text="  ").grid(row=1, column=6)
        tk.Button(frame, text="Update DataSet", command=lambda: self.update_dataset()).grid(row=1, column=7)
        tk.Label(frame, text="Epoch Rate: ").grid(row=2, column=1)
        self.epr = ttk.Entry(frame, width=5, justify="center")
        self.epr.grid(row=2, column=2)
        tk.Label(frame, text="%").grid(row=2, column=3)
        tk.Label(frame, text=" Error: ").grid(row=2, column=4)
        self.err_lvl = ttk.Entry(frame, width=7, justify="center")
        self.err_lvl.grid(row=2, column=5)
        tk.Button(frame, text="Default DataSet", command=lambda: self.update_dataset(test=True)).grid(row=2, column=7)
        frame.pack(side=tk.BOTTOM)

        tk.Label(dual, text=" ").grid(row=1, column=1)
        tk.Label(dual, text="Min").grid(row=2, column=1)
        tk.Label(dual, text="Max").grid(row=3, column=1)
        self.IP = {}
        for ii, vv in enumerate(self.variables):
            if vv != "Type":
                tk.Label(dual, text=vv).grid(row=1, column=2 + ii)

                dn_inp = ttk.Entry(dual, width=6, justify="center")
                up_inp = ttk.Entry(dual, width=6, justify="center")

                dn_inp.grid(row=2, column=2 + ii)
                up_inp.grid(row=3, column=2 + ii)

                self.IP[vv] = {"up": up_inp, "down": dn_inp}
        tk.Label(dual, text=" ").grid(row=4, column=1)
        dual.pack(side=tk.TOP)


# Run the gui
gridboard().mainloop()

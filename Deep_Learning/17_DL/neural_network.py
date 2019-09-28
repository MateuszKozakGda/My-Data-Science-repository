import numpy as np
import matplotlib.pyplot as plt

class neural_network(object):
    """Class neural network.
    Right now only 2 activation type are used but soon I will add more!!"""
    def __init__(self, net_size, learnig_rate, lambd, epochs, activator="relu"):
        np.random.seed(1)
        self.net_size = net_size
        self.lr = learnig_rate
        self.lambd = lambd
        self.epochs = epochs
        self.activator_func = activator

    def __fit(self, X, Y):
        self.X = X
        self.Y = Y
        return self.X, self.Y

    def __sigmoid(self, X):
        self.sigmoid_result = 1/(1+np.exp(-X))
        return self.sigmoid_result

    def __relu(self, X):
        self.tmp = np.zeros(X.shape)
        self.tmp[X>=0] = X[X>=0]
        return self.tmp

    def __sigmoid_derrivative(self, X):
        return self.__sigmoid(X)*(1-self.__sigmoid(X))

    def __relu_derrivative(self, X):
        self.tmp1 = np.zeros(X.shape)
        self.tmp1[X >= 0] = 1
        return self.tmp1

    def __activation(self, X,  isDerr, activator="relu"):
        if activator == "relu":
            if isDerr == True:
                return self.__relu_derrivative(np.array(X))

            else:
                return self.__relu(np.array(X))

        if  activator == "sigmoid":
            if isDerr == True:
                return self.__sigmoid_derrivative(np.array(X))
            else:
                return self.__sigmoid(np.array(X))

    def __init_wiegths(self, net_size, X):
        self.layers_num = len(net_size)
        np.random.seed(1)
        self.W1 = []
        self.b1 =[]
        self.input_shape = X.shape[0]
        for i in range(self.layers_num-1):
            self.tmpW = 2*np.random.random((net_size[i], net_size[i+1]))-1
            self.tmpb = np.zeros((net_size[i+1],1))
            self.W1.append(self.tmpW)
            self.b1.append(self.tmpb)

        return self.W1, self.b1

    def __forward(self, X, W, b, activator="relu"):
        self.lenW = len(W)
        self.Z = [0]
        self.A = [X]
        #1 layer node sum
        self.Z1 = np.dot(X, W[0]) #+ b[0]
        #1 layer node activation
        self.aOut1 = self.__activation(self.Z1, False, activator)

        #result for next layer
        self.Z.append(self.Z1)
        self.A.append(self.aOut1)

        ##forward nodes mutliplying
        for i in range(1, self.lenW):
            self.tmpz = np.dot(self.A[-1],W[i]) #+ b[i]
            self.aOut = self.__activation(self.tmpz, False, activator)
            self.Z.append(self.tmpz)
            self.A.append(self.aOut)

        return self.Z, self.A

    def __last_layer_Backward(self, Alast ,Y, Zlast, Aprev, activator="relu"):

        self.dErr_daOut = Alast-Y
        self.dZ_dW = Aprev
        self.daOut_dZ = self.__activation(Zlast, True, activator)
        self.b_last = np.sum(self.dErr_daOut, keepdims=True, axis=1)
        ##weights change
        self.delta = self.dErr_daOut*self.daOut_dZ
        self.dErr_dW = np.dot(self.dZ_dW.T, self.delta)

        return self.dErr_dW, self.delta, self.b_last

    def __hiden_layer_backward(self, delta_prev, W_next, Z, Act_out_prev, activator="relu"):
        self.m = Z.shape[1]

        ##derrivatives
        self.dErr_daOut1 = np.dot(delta_prev, W_next.T)
        self.dZ_dW1 = Act_out_prev
        self.daOut_dZ1 = self.__activation(Z, True, activator)
        self.b_hidden = np.sum(self.dErr_daOut1, keepdims=True, axis=1)

        ##weights change
        self.delta1 = self.dErr_daOut1*self.daOut_dZ1
        self.dErr_dW1 = np.dot(self.dZ_dW1.T, self.delta1)

        return self.dErr_dW1, self.delta1, self.b_hidden

    def __backward(self, A, Z, Y, W, activator="relu"):
        """Backward propagation"""
        self.layers_num2 = len(A)-1
        self.dErr_list = []
        self.b_error_list = []

        self.dErr_dW_Last, self.delta_last, self.b_last = self.__last_layer_Backward(A[self.layers_num2],
                                                                        Y, Z[self.layers_num2],
                                                                        A[self.layers_num2-1],
                                                                        activator)
        self.dErr_list.append(self.dErr_dW_Last)
        self.b_error_list.append(self.b_last)

        for i in range(self.layers_num2-1,0,-1):
            self.dErr, self.delta_last, self.b_last = self.__hiden_layer_backward(self.delta_last,
                                                                     W[i],
                                                                     Z[i],
                                                                     A[i-1],
                                                                     activator)
            self.dErr_list.append(self.dErr)
            self.b_error_list.append(self.b_last)

        return self.dErr_list[::-1], self.b_error_list[::-1]

    def __weights_update(self, W, b ,DError, berror, lr, lambd):

        self.num = len(DError)
        self.W_list_2 = []
        self.b_list_2 =[]

        for i in range(self.num):
            self.m2 = W[i].shape[1]
            W[i] = W[i]-lr*((1/self.m2)*DError[i]+(lambd/self.m2)*W[i])
            #b[i] = b[i]-lr*((1/self.m2)*berror[i]+(lambd/self.m2)*b[i])
            self.W_list_2.append(W[i])
            self.b_list_2.append(b[i])

        return self.W_list_2, self.b_list_2

    def __mse_reg(self, Ypred, Yref, W, lambd):
        self.L2_weight_sum_list = []
        self.cost = np.mean(np.square(Ypred-Yref))

        for w in W:
            self.L2_weight_square = np.sum(np.square(w))
            self.L2_weight_sum_list.append(self.L2_weight_square)

        self.L2_weight_sum2 = np.array(self.L2_weight_sum_list).sum()
        self.L2_weight_sum3 = self.L2_weight_sum2*(lambd/(2*Ypred.shape[1]))
        self.cost_regularized = self.cost + self.L2_weight_sum3

        return self.cost_regularized

    def fit_train(self, X, Y, draw_plot=True):
        self.error_vector = []
        self.X_input, self.Y_input = self.__fit(X, Y)

        self.W, self.b1 = self.__init_wiegths(self.net_size, self.X_input)
        for i in range(self.epochs):
            self.Zx, self.Ax = self.__forward(self.X_input, self.W, self.b1 ,self.activator_func)

            if i%100 == 0:
                print(f"Cost in Epoch {i}: {self.__mse_reg(self.Ax[-1], Y, self.W, self.lambd)}")

            self.Derr, self.berr = self.__backward(self.Ax, self.Zx, Y, self.W, self.activator_func)
            ##weights update
            self.W, self.b1 = self.__weights_update(self.W, self.b1, self.Derr, self.berr ,self.lr, self.lambd)
            self.error_vector.append(self.__mse_reg(self.Ax[-1], Y, self.W, self.lambd))


        if draw_plot==True:
            plt.plot([i for i in range(self.epochs)], self.error_vector)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.title("Cost change in terms of number of training Epochs")
            plt.show()

            print(f"Reference label: \n{Y}")
            print(f"Predicted labels: \n{self.Ax[-1]}")

        else:
            print(f"Reference label: \n{Y}")
            print(f"Predicted labels: \n{self.Ax[-1]}")

    def fit_train_min(self, X, Y, draw_plot=True):
        self.error_vector = []
        self.acc = []
        self.X_input, self.Y_input = self.__fit(X, Y)

        self.W, self.b1 = self.__init_wiegths(self.net_size, self.X_input)
        for i in range(self.epochs):
            self.Zx, self.Ax = self.__forward(self.X_input, self.W, self.b1 ,self.activator_func)

            self.Derr, self.berr = self.__backward(self.Ax, self.Zx, Y, self.W, self.activator_func)
            ##weights update
            self.W, self.b1 = self.__weights_update(self.W, self.b1, self.Derr, self.berr ,self.lr, self.lambd)
            self.error_vector.append(self.__mse_reg(self.Ax[-1], Y, self.W, self.lambd))

            self.yargmax = np.argmax(Y, axis=1)
            self.y_outArgMax = np.argmax(self.Ax[-1], axis=1)
            self.acc.append(accuracy_score(self.yargmax, self.y_outArgMax))
            self.error = self.Ax[-1]-Y

            if i%100 == 0:
                #print(f"Cost in Epoch {i}: {self.__mse_reg(self.Ax[-1], Y, self.W, self.lambd)}")
                print(f"Accurancy of Epoch {i}: {round(accuracy_score(self.yargmax, self.y_outArgMax), 4) * 100}%")
                print(f"Error of Epoch {i}: {np.mean(np.abs(self.error))}")
                print("===========================================")

        if draw_plot==True:
            plt.plot([i for i in range(self.epochs)], self.error_vector)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.title("Cost change in terms of number of training Epochs")
            plt.show()

            print(f"Reference label: \n{np.argmax(Y, axis=1)}")
            print(f"Predicted labels: \n{np.argmax(self.Ax[-1], axis=1)}")

        else:
            print(f"Reference label: \n{np.argmax(Y, axis=1)}")
            print(f"Predicted labels: \n{np.argmax(self.Ax[-1], axis=1)}")

    def predict(self, X_pred):
        self.z_pred, self.a_pred = self.__forward(X_pred, self.W, self.b1, activator=self.activator_func)
        print(f"Prediction: \n {np.argmax(self.a_pred[-1], axis=1)}")
        return self.a_pred[-1]

X = np.array([[0,0,1],
             [1,1,1],
             [1,0,1],
             [0,1,1]])

Y = np.array([[1,0,0,1]]).T
epochs=20000
lambd = 0.015
lr =0.025
actName = "relu"
net_size = [X.shape[1],5,Y.shape[1]]

model = neural_network(net_size, learnig_rate=lr, lambd=0.15, epochs=epochs, activator=actName)
model.fit_train(X, Y, draw_plot=True)


X_pr = np.array([[0,0,0],
             [0,0,0],
             [0,0,0],
             [0,0,10]])
model.predict(X_pr)


from sklearn.datasets import load_digits
dane = load_digits()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

imgs = dane.images

df = pd.DataFrame(dane.target)

samplesNum = dane.target.shape[0]

X_img = imgs.reshape((samplesNum, -1))


df_X = pd.DataFrame(X_img)

encoder = OneHotEncoder(sparse=False)
y_ = encoder.fit_transform(df)
#print(X_train)

X_train, X_test, y_train, y_test = train_test_split(df_X, y_, random_state=2, test_size = 0.3)

from sklearn.metrics import accuracy_score

net_size_mininst = [X_train.shape[1],25,50,50,25,y_train.shape[1]]
model2 = neural_network(net_size=net_size_mininst, lambd=0.75, learnig_rate=0.05, epochs=3000, activator="sigmoid")
model2.fit_train_min(X_train, y_train)
model2.predict(X_test)


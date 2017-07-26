import numpy as np
import matplotlib.pyplot as plot

# in perceptron, step function is used as a activation function
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_for_numpy_array(x):
    y = x > 0 # bool array
    return y.astype(np.int) # convert bool array to int array (true:1, false:0)

# in neural network, sigmoid function is one of the frequently used activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def idendify_function(x):
	return x

def softmax(x): 
    if x.ndim == 2: 
        x = x.T 
        x = x - np.max(x, axis=0) 
        y = np.exp(x) / np.sum(np.exp(x), axis=0) 
        return y.T  

    x = x - np.max(x) # 오버플로 대책 
    return np.exp(x) / np.sum(np.exp(x)) 

#x = np.arange(-5.0, 5.0, 0.1)
#y = sigmoid(x)
#plot.plot(x, y)
#plot.ylim(-0.1, 1.1) # y scope
#plot.show() # sigmoid is continuous graph but step function is not
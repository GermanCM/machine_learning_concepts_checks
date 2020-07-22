# %%
import numpy as np 

n=1000
t=1.4
sigma_R = t*0.001
min_value_t = t-sigma_R
max_value_t = t+sigma_R
y_data = min_value_t + (max_value_t - min_value_t) * np.random.rand(n,1)
x_data=np.array(range(1000))

#%%
from matplotlib import pyplot

pyplot.scatter(x_data, y_data)
pyplot.show()

#%%
m=0
c=0
L=0.001
epochs=10 #/iterations
early_stop = 0.001

#%%
import numpy as np 

y_pred=m*x_data+c
current_cost=(1/n)*np.sum(np.square(np.array(y_data)-np.array(y_pred)))
#%%
x_data=x_data.reshape(-1, )
y_data=y_data.reshape(-1, )
y_pred=y_pred.reshape(-1, )
for i in range(epochs):
    y_pred=np.multiply(m, x_data)+c
    print('y_pred: ', y_pred)
    D_m=(-2/n)*np.dot(np.array(x_data), np.subtract(np.array(y_data), np.array(y_pred)))
    print('D_m: ', D_m)
    D_c=(-2/n)*np.sum(y_data-y_pred)
    print('D_m: ', D_m)
    m=m+L*D_m
    c=c+L*D_c

    current_cost=(1/n)*np.sum(np.square(np.array(y_data)-np.array(y_pred)))
    if current_cost < early_stop:
        break

    '''
    pyplot.scatter(x_data, y_data)
    pyplot.plot(x_data, y_pred, color='red')
    pyplot.show()
    '''

# %%

#VERSIÓN DE: http://charlesfranzen.com/posts/multiple-regression-in-python-gradient-descent/
import math
import random
import numpy as np

# get the cost (error) of the model
def computeCost(X, y, coeff):  
    inner = np.power((predict_output(X, coeff) - y), 2)
    return np.sum(inner) / (len(X))

def predict_output(feature_matrix, coefficients):
    ''' Returns an array of predictions
    
    inputs - 
        feature_matrix - 2-D array of dimensions data points by features
        coefficients - 1-D array of estimated feature coefficients
        
    output - 1-D array of predictions
    '''
    predictions = np.dot(feature_matrix, coefficients)
    return predictions

def feature_derivative(errors, feature):
    N = len(feature)
    derivative = (2)*np.dot(errors, feature)
    return(derivative)

def gradient_descent_regression(H, y, initial_coefficients, eta, epsilon, max_iterations=10000):
    ''' Returns coefficients for multiple linear regression.
    
    inputs - 
        H - 2-D array of dimensions data points by features
        y - 1-D array of true output
        initial_coefficients - 1-D array of initial coefficients
        eta - float, the step size eta
        epsilon - float, the tolerance at which the algorithm will terminate
        max_iterations - int, tells the program when to terminate
    
    output - 1-D array of estimated coefficients
    '''
    converged = False
    w = initial_coefficients
    iteration = 0
    cost=[]
    while iteration < max_iterations:
        pred = predict_output(H, w)
        residuals = pred-y
        gradient_sum_squares = 0
        for i in range(len(w)):
            partial = feature_derivative(residuals, H[:, i])
            gradient_sum_squares += partial**2
            w[i] = w[i] - eta*partial
        '''gradient_magnitude = math.sqrt(gradient_sum_squares)  TOCHECK
           if gradient_magnitude < epsilon:
            converged = True'''
        iteration += 1
        cost.append(computeCost(H, y, w))
    return w, cost


# %%
# %%
import numpy as np 
'''
n=1000
t=1.4
sigma_R = t*0.001
min_value_t = t-sigma_R
max_value_t = t+sigma_R
y_data = min_value_t + (max_value_t - min_value_t) * np.random.rand(n,1)
x_data=np.array(range(1000))
y_data = 3*x_data
'''
from numpy.random import randn, seed
from numpy import std
from scipy.stats import linregress
from matplotlib import pyplot

# seed random number generator
seed(1)
# prepare data
x_data = 20 * randn(1000) + 100
y_data = x_data + (10 * randn(1000) + 50)

pyplot.hist(x_data)
pyplot.show()
pyplot.scatter(x_data, y_data)
pyplot.show()


# %%
n=len(x_data)
feature_matrix = np.zeros(n*2) 
feature_matrix.shape = (n, 2) 
feature_matrix[:,0] = 1
feature_matrix[:,1] = x_data
initial_coefficients = np.array([0.05, 0.01])

y_data=y_data.reshape(1000,)
coef, cost = gradient_descent_regression(feature_matrix, y_data, initial_coefficients, eta=0.001, epsilon=1, max_iterations=10000)
print(coef, cost[-1])

# %%

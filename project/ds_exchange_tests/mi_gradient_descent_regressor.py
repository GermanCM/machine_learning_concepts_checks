# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
DOCUMENTACIÃN SOBRE LINEAR MODELS:
http://scikit-learn.org/stable/modules/linear_model.html
''' 


# %%
'''
multiple linear regression:
The idea with gradient descent is that for each iteration, we compute the gradient of the error term in order to figure out 
the appropriate direction to move our parameter vector. In other words, we're calculating the changes to make to our parameters 
in order to reduce the error, thus bringing our solution closer to the optimal solution (i.e best fit).
fuente: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
'''
#%%
#VERSIÃN DE: http://charlesfranzen.com/posts/multiple-regression-in-python-gradient-descent/
import math
import random
import numpy as np

# get the cost (error) of the model
def computeCost(y_true, y_predicted):  
    inner = np.power((y_predicted - y_true), 2)
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
        cost.append(computeCost(y, pred))
    return w, cost


# %%
def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " +                          "not be normalized. Please do not include features with only a single value " +                          "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

#ENCONTRAR EL COEFF. DE REGRESIÃN LINEAL: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
def find_r2_score(labels_test, predicted_outputs):
    from sklearn.metrics import r2_score
    corr_coeff = r2_score(labels_test, predicted_outputs)
    print('the value of r2 is: ', corr_coeff)


# %%
#predicciones con gradient_descent_data_identity_2D.csv:
import pandas as pd
from pandas import DataFrame, Series 
import numpy as np
'''
from azureml import Workspace
ws = Workspace(
    workspace_id='b692ccad88f84e139ce2040473db008f',
    authorization_token='Z6AoPT8cBvrepyHGXuilNqTfGMH2dOws60ak0Bpapx+K20mGq179RcgLIZp/28qyzb+jDDYeTyVH1nHHRLzh9Q==',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['gradient_descent_data_identity_2D.csv']
final_df = ds.to_dataframe() 
'''
final_df = pd.read_csv(r'.\data\gradient_descent_data_identity.csv', sep=';')
#split dataframe into columns
final_df.rename(columns={'Y': 'W'}, inplace=True)

#%%
'''
final_df[['X', 'Y','W']] = pd.DataFrame(
      [x.split(';;') for x in final_df['X;;Y;;W'].tolist()])
'''
#construyo el features dataframe
X = final_df['X']
#Y = final_df['Y']
TARGET = final_df['W']
#%%
X = pd.to_numeric(X)
#Y = pd.to_numeric(Y)
TARGET = 2*(pd.to_numeric(TARGET))
#meto cierta desviaciÃ³n en datos apra que no salga perfectamente 1
TARGET[10:18] = np.multiply(TARGET[10:18], 0.8)   
TARGET[30:48] = np.multiply(TARGET[30:48], 1.1)
#ahora creo el dictionary:
final_df_DICT = {'X': X}
#y ahora el dataframe
H = pd.DataFrame(final_df_DICT) 

# %%
n = len(H['X'])
#feature_matrix hace de training data 
feature_matrix = np.zeros(n*2) 
feature_matrix.shape = (n, 2) 
feature_matrix[:,0] = 1 
feature_matrix[:,1] = H['X'] 

#%%
#normalize features# normalize features: http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
feature_matrix, mu, sigma = normalize_features(feature_matrix)

initial_coefficients = np.zeros(len(feature_matrix[0]))

coef, cost = gradient_descent_regression(feature_matrix, TARGET, initial_coefficients, 0.0001, 1, max_iterations=1000) 
print('coef: {}'.format(coef))
print('cost: {}'.format(cost[-1]))


# %%
#el Ãºltimo elemento de la serie 'cost' es '0.1308', igual que el coste computado siguiente:
my_predictions = predict_output(feature_matrix, coef)
computeCost(TARGET, my_predictions) 

from matplotlib import pyplot

pyplot.scatter(feature_matrix[:, 1], TARGET)
pyplot.scatter(feature_matrix[:, 1], my_predictions, color='r')
#pyplot.plot(x, yhat_bootstrapped, color='r')
pyplot.show()

#%%[markdown]
# Repito con otro dataset:
from matplotlib import pyplot
import numpy as np 
from numpy.random import randn, seed
'''
x_data = 20 * randn(1000) + 100
y_data = x_data + (10 * randn(1000) + 50)
'''
n=1000
t=1.4
sigma_R = t*0.001
min_value_t = t-sigma_R
max_value_t = t+sigma_R
y_data = min_value_t + (max_value_t - min_value_t) * np.random.rand(n,1)
x_data=np.array(range(1000))

pyplot.scatter(x_data, y_data)
pyplot.show()
pyplot.show()

#%%
import pandas as pd 

final_df=pd.DataFrame({'X': x_data, 'W': y_data.reshape(len(y_data), )})
#construyo el features dataframe
X = final_df['X']
TARGET = final_df['W']
X = pd.to_numeric(X)
#Y = pd.to_numeric(Y)
TARGET = pd.to_numeric(TARGET)
#ahora creo el dictionary:
final_df_DICT = {'X': X}
#y ahora el dataframe
H = pd.DataFrame(final_df_DICT)

#%%[markdown]
## Ejemplo para DS exchange question:
#%%
import pandas as pd 
import numpy as np

n=1000
t=1.4
sigma_R = t*0.001
min_value_t = t-sigma_R
max_value_t = t+sigma_R
y_data = min_value_t + (max_value_t - min_value_t) * np.random.rand(n,1)
x_data=np.array(range(1000))

final_df_DICT = {'X': x_data}
H = pd.DataFrame(final_df_DICT)
feature_matrix = np.zeros(n*2) 
feature_matrix.shape = (n, 2) 
feature_matrix[:,0] = 1 
feature_matrix[:,1] = H['X'] 
feature_matrix
#%%
#standardize features
feature_matrix = (feature_matrix - feature_matrix.mean()) / feature_matrix.std()
target_data = y_data.reshape(len(y_data), )
#%%
w = [0, 0]
L=0.0001
epochs=1000
iteration = 0
cost=[]
while iteration < epochs:
    pred = np.dot(feature_matrix, w)
    residuals = pred-target_data
    gradient_sum_squares = 0
    for i in range(len(w)):
        partial = 2*np.dot(residuals, feature_matrix[:, i])
        gradient_sum_squares += partial**2
        w[i] = w[i] - L*partial
    
    iteration += 1
    computed_cost = np.sum(np.power((pred - target_data), 2)) / n

    cost.append(computed_cost)

print('coef: {}'.format(w))
print('cost: {}'.format(cost[-1]))
#%%
from matplotlib import pyplot

my_predictions = np.dot(feature_matrix, w)
pyplot.scatter(feature_matrix[:, 1], target_data)
pyplot.scatter(feature_matrix[:, 1], my_predictions, color='r')

pyplot.show()

#%%
'''
# ANIMATION TEST

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

fig = plt.figure(figsize=(10,6))
plt.xlim(x_data.min(), x_data.max())
plt.ylim(y_data.min(), y_data.max())
plt.xlabel('X',fontsize=20)
plt.ylabel('title',fontsize=20)
plt.title('ANIMATION TEST',fontsize=20)

data = final_df
def animate(i):
    #select data range
    x_data_i = data['X'][:int(i+1)]
    y_data_i = data['W'][:int(i+1)]
    p = sns.lineplot(x=x_data_i, y=y_data_i, data=data, color="y")
    p.tick_params(labelsize=17)
    plt.setp(p.lines,linewidth=7)

ani = FuncAnimation(fig, animate, frames=17, repeat=True)
#                    init_func=init, blit=True)
plt.show()
'''


######################################
# %%
#grÃ¡fica de la evoluciÃ³n de la funciÃ³n coste
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,5))  
ax.plot(np.arange(len(cost)), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  
plt.show()


# %%
#PLOT 3D: https://matplotlib.org/examples/mplot3d/lines3d_demo.html

#Check the predicted values for the training 
predictions_on_training_set = predict_output(feature_matrix, coef)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
z = predictions_on_training_set 
x = H['X']
y = H['Y']
ax.plot(x, y, z, label='TEST LINE')
ax.scatter(x, y, TARGET)
ax.legend()

plt.show()


# %%
'''
ahora para predecir el valor de un nuevo dato de entrada with the 'predict_output(feature_matrix, coefficients)'
'''
# 1.- creo la input data matrix: para el punto (X, Y):
import pandas as pd
from pandas import DataFrame, Series 
import numpy as np

X = [15]
Y = [15]
#ahora creo el dictionary:
H_DICT = {'X': X,'Y': Y}
#y ahora el dataframe
H = DataFrame(H_DICT)

# 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
n = 1
#feature_matrix hace de training data
feature_to_predict = np.zeros(n*3)
feature_to_predict.shape = (n, 3)
feature_to_predict[:,0] = 1
feature_to_predict[:,1] = H['X']
feature_to_predict[:,2] = H['Y']

#normalize features#
feature_to_predict = (feature_to_predict - mu) / sigma

coefficients = coef

outp = predict_output(feature_to_predict, coefficients)
print(feature_to_predict)
print(outp)

# %% [markdown]
# ### Determinista en predicciÃ³n una vez entrenado, y determinista en entrenamiento

# %%
for i in range(3):
    coef, cost = gradient_descent_regression(feature_matrix, TARGET, initial_coefficients, 6e-5, 1) 
    outp = predict_output(feature_to_predict, coef)
    print(outp)


# %%
#ENCONTRAR EL COEFF. DE REGRESIÃN LINEAL: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
#labels test & predicted_outputs array:
y_true = []
y_predicted_outputs = []
x = 100
y = 100
for i in range (10): 
    x += i
    y += i
    X = [x]
    Y = [y]
    #ahora creo el dictionary:
    H_DICT = {'X': X,'Y': Y}
    print(H_DICT)
    #y ahora el dataframe
    H = DataFrame(H_DICT)
    # 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
    n = 1
    #feature_matrix test
    feature_matrix_to_Test = np.zeros(n*3)
    feature_matrix_to_Test.shape = (n, 3)
    feature_matrix_to_Test[:,0] = 1
    feature_matrix_to_Test[:,1] = H['X']
    feature_matrix_to_Test[:,2] = H['Y']
    #normalize features#
    feature_matrix_to_Test = (feature_matrix_to_Test - mu) / sigma
    #predict value
    predict_value = predict_output(feature_matrix_to_Test, coefficients)
    #aÃ±adir resultado al conjunto de valores predichos
    y_predicted_outputs.append(float(predict_value))
    y_true.append(2*x)

print(y_true)
print(y_predicted_outputs)
#coeff. r2:
corr_coeff = find_r2_score(y_true, y_predicted_outputs)


# %%
#INCISO: ver por quÃ© estos dos arrays no me dan un coeff. de correlaciÃ³n = 1 
#source: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
labels_test_2 = [1, 2, 3, 4, 5]  #[100, 101, 103, 106, 110, 115, 121, 128, 136, 145]
predicted_outputs_2 = [2, 4, 6, 8, 10] #[200.00000000000057, 202.00000000000063, 206.00000000000063, 212.00000000000074, 220.0000000000008, 230.0000000000009, 242.00000000000108, 256.0000000000012, 272.0000000000014, 290.0000000000016]

corr_coeff = find_r2_score(labels_test_2, predicted_outputs_2)


# %%
import pandas as pd
from pandas import DataFrame, Series 
import numpy as np

X = [15]
Y = [15]
#ahora creo el dictionary:
H_DICT = {'X': X,'Y': Y}
#y ahora el dataframe
H = DataFrame(H_DICT)

# 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
n = 1
#feature_matrix hace de training data
feature_to_predict = np.zeros(n*3)
feature_to_predict.shape = (n, 3)
feature_to_predict[:,0] = 1
feature_to_predict[:,1] = H['X']
feature_to_predict[:,2] = H['Y']


feature_to_predict = (feature_to_predict - mu) / sigma


# %%
TARGET[:10]


# %%
'''GOOD READ!!! 
- https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent
- http://sdsawtelle.github.io/blog/output/week2-andrew-ng-machine-learning-with-python.html
'''
#ahora con SCIKIT-Learn SGDRegressor
from sklearn import linear_model

clf = linear_model.SGDRegressor()
clf.fit(feature_matrix, TARGET)
print(feature_to_predict) 


# %%
for i in range(3):
    print('SGDRegressor prediction: ', clf.predict(feature_to_predict))
   

# %% [markdown]
# ### Determinista en predicciÃ³n una vez entrenado, pero no determinista en entrenamiento

# %%
for i in range(3):
    sgd_regressor = linear_model.SGDRegressor()
    sgd_regressor.fit(feature_matrix, TARGET)
    print('SGDRegressor prediction: ', sgd_regressor.predict(feature_to_predict))

# %% [markdown]
# ### Comparamos costes entre mi clasificador manual y los de scikit

# %%
#total cost 
my_predictions = predict_output(feature_matrix, coef)
computeCost(TARGET, my_predictions) 


# %%
sgd_regressor_predictions = sgd_regressor.predict(feature_matrix)
computeCost(TARGET, sgd_regressor_predictions)


# %%
#ahora con SCIKIT-Learn linear regression
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(feature_matrix, TARGET)
print(feature_to_predict)


# %%
linear_regressor.predict(feature_to_predict)

# %% [markdown]
# ### Determinista en predicciÃ³n una vez entrenado, y determinista en entrenamiento

# %%
for i in range(3):
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(feature_matrix, TARGET)
    print('Linear_regressor prediction: ', linear_regressor.predict(feature_to_predict))


# %%
computeCost(TARGET, linear_regressor.predict(feature_matrix))

# %% [markdown]
# ## AquÃ­ vemos comparativa entre mÃ©todos deterministas (mi training GD que se inicializa siempre igual o el Linear Regression) VS el SGD estocÃ¡stico (en entrenamiento) de SCIKIT 

# %%



# %%
#ahora uso scikit dando features sin normalizar

final_df_DICT = {'X': X, 'Y': Y}
#y ahora e exl dataframe
H = pd.DataFrame(final_df_DICT)
n = len(H['X'])
print(H)


# %%
print(W)


# %%
from sklearn import linear_model

clf_3 = linear_model.LinearRegression()
clf_3.fit(H, W)
print(clf_3.coef_)
print(clf_3.intercept_)


# %%
clf_3.predict([15, 15])


# %%



# %%
#ENCONTRAR EL COEFF. DE REGRESIÃN LINEAL: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
#labels test & predicted_outputs array:
labels_test = []
predicted_outputs = []
x = 100
y = 100
for i in range (10): 
    x += i
    y += i
    X = [x]
    Y = [y]
    #ahora creo el dictionary:
    H_DICT = {'X': X,'Y': Y}
    #y ahora el dataframe
    H = DataFrame(H_DICT)
    # 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
    n = 1
    #feature_matrix hace de training data
    feature_matrix_to_Test = np.zeros(n*3)
    feature_matrix_to_Test.shape = (n, 3)
    feature_matrix_to_Test[:,0] = 1
    feature_matrix_to_Test[:,1] = H['X']
    feature_matrix_to_Test[:,2] = H['Y']
    #normalize features#
    feature_matrix_to_Test = (feature_matrix_to_Test - mu) / sigma
    #predict value
    predicted_value = clf.predict(feature_matrix_to_Test)
    #predicted_value_LinearR = clf_2.predict(feature_matrix_to_Test)
    #aÃ±adir resultado al conjunto de valores predichos
    predicted_outputs.append(float(predicted_value))
    labels_test.append(2*x)  
    
print(labels_test)
print(predicted_outputs)
#coeff. r2:
corr_coeff = find_r2_score(labels_test, predicted_outputs)


# %%
#probar tb el r2 con clf.score de scikitlearn

#COMPARAR LA r2 ENTRE LOS TRAINGING DATA Y LOS TESTING DATA (NO QUEREMOS VER OVERFITTING)


# %%
#ENCONTRAR EL COEFF. DE REGRESIÃN LINEAL: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
#labels test & predicted_outputs array:
labels_test = []
predicted_outputs = []
x = 100
y = 100
for i in range (10): 
    x += i
    y += i
    X = [x]
    Y = [y]
    #ahora creo el dictionary:
    H_DICT = {'X': X,'Y': Y}
    #y ahora el dataframe
    H = DataFrame(H_DICT)
    # 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
    n = 1
    #feature_matrix hace de training data
    feature_matrix_to_Test = np.zeros(n*3)
    feature_matrix_to_Test.shape = (n, 3)
    feature_matrix_to_Test[:,0] = 1
    feature_matrix_to_Test[:,1] = H['X']
    feature_matrix_to_Test[:,2] = H['Y']
    #normalize features#
    feature_matrix_to_Test = (feature_matrix_to_Test - mu) / sigma
    #predict value
    predicted_value = clf_2.predict(feature_matrix_to_Test)
    #aÃ±adir resultado al conjunto de valores predichos
    predicted_outputs.append(float(predicted_value))
    labels_test.append(2*x)

print(labels_test)
print(predicted_outputs)
#coeff. r2:
corr_coeff = find_r2_score(labels_test, predicted_outputs)


# %%
#HASTA AQUÃ VEO QUE PUEDO USAR EL 'SGDRegressor' DE SCIKIT-LEARN como anÃ¡logo (mirar tb quÃ© da sin normalizar las features)


# %%
'''
Si lo uso para problema unidimensional (1 feature) sin usar aÃºn 'epsilon':
'''
#Y SI FUERA DE 1-D:
import pandas as pd
from pandas import DataFrame, Series 
import numpy as np

from azureml import Workspace
ws = Workspace(
    workspace_id='b692ccad88f84e139ce2040473db008f',
    authorization_token='Z6AoPT8cBvrepyHGXuilNqTfGMH2dOws60ak0Bpapx+K20mGq179RcgLIZp/28qyzb+jDDYeTyVH1nHHRLzh9Q==',
    endpoint='https://studioapi.azureml.net'
)
ds = ws.datasets['gradient_descent_data_identity.csv']
final_df = ds.to_dataframe() 

#split dataframe into columns
final_df[['X', 'Y']] = pd.DataFrame(
      [x.split(';') for x in final_df['X;Y'].tolist()])
#construyo el features dataframe
X = final_df['X']
Y = final_df['Y']
print(Y)

X = pd.to_numeric(X)
Y = pd.to_numeric(Y)
#ahora creo el dictionary:
H_DICT = {'X': X}
#y ahora el dataframe
H = pd.DataFrame(H_DICT)


# %%
#normalize features#
H, mu, sigma = normalize_features(H)  
Y = (Y - Y.mean()) / Y.std()

n = 100
#feature_matrix hace de training data
feature_matrix = np.zeros(n*2)
feature_matrix.shape = (n, 2)
feature_matrix[:,0] = 1
feature_matrix[:,1] = H['X']

initial_coefficients = np.array([0., 0.])

learning_rate = 0.0001
num_iterations = 1000
#coef, cost = gradient_descent_regression(feature_matrix, Y, initial_coefficients, learning_rate, 0.01, num_iterations)
coef, cost = gradient_descent_regression(feature_matrix, Y, initial_coefficients, learning_rate, num_iterations)
print(coef)


# %%

computeCost(feature_matrix, Y, coef) 


# %%
'''SÃ QUE ESTE ALGORITMO DE REGRESIÃN ME ESTÃ DANDO LO ESPERADO PARA LOS DATOS INTRODUCIDOS
ahora para predecir el valor de un nuevo dato de entrada eith the 'predict_output(feature_matrix, coefficients)'
'''
# 1.- creo la input data matrix: para el punto (102):
import pandas as pd
from pandas import DataFrame, Series 
import numpy as np

#points = pd.read_csv(r'C:\Users\gcabrera.ai\Desktop\MIS\DATA_SCIENCE_COURSES\INTRO_TO_DATA_SCIENCE\gradient_descent_data_identity.csv', delimiter=",")
   
X = [102]
#ahora creo el dictionary:
H_DICT = {'X': X}
#y ahora el dataframe
H = DataFrame(H_DICT)
print(H)

# 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
n = 1
#feature_matrix hace de training data
feature_matrix = np.zeros(n*2)
feature_matrix.shape = (n, 2)
feature_matrix[:,0] = 1
feature_matrix[:,1] = H['X']

coefficients = coef

preddict_value = predict_output(feature_matrix, coefficients)
print(preddict_value)


# %%
#ENCONTRAR EL COEFF. DE REGRESIÃN LINEAL: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
#labels test & predicted_outputs array:
labels_test = []
predicted_outputs = []
x = 102
for i in range (10): 
    x += i
    X = [x]
    #ahora creo el dictionary:
    H_DICT = {'X': X}
    #y ahora el dataframe
    H = DataFrame(H_DICT)

    # 2.- utilizo, con los coeficientes obtenidos, la 'predict_output'
    n = 1
    #feature_matrix hace de training data
    feature_matrix = np.zeros(n*2)
    feature_matrix.shape = (n, 2)
    feature_matrix[:,0] = 1
    feature_matrix[:,1] = H['X']
    #predict value
    preddict_value = predict_output(feature_matrix, coefficients)
    #aÃ±adir resultado al conjunto de valores predichos
    predicted_outputs.append(float(preddict_value))
    labels_test.append(x)

print(labels_test)
print(predicted_outputs)
#coeff. r2:
corr_coeff = find_r2_score(labels_test, predicted_outputs)


# %%
'''Y usando 'epsilon'
'''
'''
multiple linear regression:
The idea with gradient descent is that for each iteration, we compute the gradient of the error term in order to figure out 
the appropriate direction to move our parameter vector. In other words, we're calculating the changes to make to our parameters 
in order to reduce the error, thus bringing our solution closer to the optimal solution (i.e best fit).
fuente: http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
'''

#VERSIÃN DE: http://charlesfranzen.com/posts/multiple-regression-in-python-gradient-descent/
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
    while not converged:
        if iteration > max_iterations:
            print('Exceeded max iterations\nCoefficients: ', w)
            return w, cost
        pred = predict_output(H, w)
        residuals = pred-y
        gradient_sum_squares = 0
        for i in range(len(w)):
            partial = feature_derivative(residuals, H[:, i])
            #en 'partial', hemos implementado la parcial de la funciÃ³n coste respecto al atributo i; dicha parcial es la
            #suma de dicha parcial en cada muestra (esto es, producto del error de la muestra * coordenada i de dicha muestra) 
            gradient_sum_squares += partial**2
            w[i] = w[i] - eta*partial
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < epsilon:
            converged = True
        iteration += 1
        cost.append(computeCost(H, y, w))
    return w, cost


# %%
#PLOT 3D: https://matplotlib.org/examples/mplot3d/lines3d_demo.html

#PLOT 3D: https://matplotlib.org/examples/mplot3d/lines3d_demo.html

 
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
z_training = [3, 5, 7, 4, 5, 6, 8, 2, 5, 6, 9, 18, 10, 9, 11, 16, 17, 18, 11, 19]
#ahora creo el dictionary:
test_DICT = {'X': x, 'Y': y}
#y ahora el dataframe
H = pd.DataFrame(test_DICT)

n = len(H['X'])
#feature_matrix hace de training data 
feature_matrix = np.zeros(n*3) 
feature_matrix.shape = (n, 3) 
feature_matrix[:,0] = 1 
feature_matrix[:,1] = H['X'] 
feature_matrix[:,2] = H['Y']

#normalize features# normalize features: http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
feature_matrix, mu, sigma = normalize_features(feature_matrix)

initial_coefficients = np.zeros(len(feature_matrix[0]))

coef, cost = gradient_descent_regression(feature_matrix, z_training, initial_coefficients, 6e-5, 1) 
print(coef, cost[-1])

#Check the predicted values for the training
predictions_on_training_set = predict_output(feature_matrix, coef)


# %%
#Check the predicted values for the training 
predictions_on_training_set = predict_output(feature_matrix, coef)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

z_pred = predictions_on_training_set 
x = H['X']
y = H['Y']

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(x, y, z_pred, label='TEST LINE')
ax.scatter(x, y, z_training)
ax.legend()

plt.show()


# %%



#%%[markdown]
## Confidence intervals for model parameters:
'''
The bootstrap resampling method can be used as a nonparametric method for
calculating condence intervals, nominally called bootstrap condence intervals. The bootstrap
is a simulated Monte Carlo method where samples are drawn from a xed nite dataset with
replacement and a parameter is estimated on each sample. This procedure leads to a robust
estimate of the true population parameter via sampling. <p>
The procedure can be used to estimate the skill of a predictive model by tting the model on
each sample and evaluating the skill of the model on those samples not included in the sample.
The mean or median skill of the model can then be presented as an estimate of the model skill
when evaluated on unseen data. Condence intervals can be added to this estimate by selecting
observations from the sample of skill scores at specic percentiles. <p>

statistics = [] <p>
for i in bootstraps: <p>
    sample = select_sample_with_replacement(data) <p>
    stat = calculate_statistic(sample) <p>
    statistics.append(stat) 
'''
# %%
# bootstrap confidence intervals
from numpy.random import seed, rand, randint
from numpy import mean, median, percentile

# seed the random number generator
seed(1)
# generate dataset
dataset = 0.5 + rand(1000) * 0.5

from matplotlib import pyplot

pyplot.hist(dataset)
pyplot.show()
pyplot.show()

#%%
# bootstrap
scores = list()
for _ in range(100):
    # bootstrap sample
    indices = randint(0, 1000, 1000)
    sample = dataset[indices]
    #print('sample size:', len(sample))
    # calculate and store statistic
    statistic = mean(sample)
    scores.append(statistic)
#%%[markdown]
### Size of the bootstraped samples = whole dataset size; lo que pasa es que hay repetición de puntos

#%%
print('mean of the population = %.3f' % mean(dataset))
print('50th percentile (median) of the bootstrap samples means = %.3f' % median(scores))

#%%
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
lower = max(0.0, percentile(scores, lower_p))
print('%.1fth percentile = %.3f' % (lower_p, lower))
# calculate upper percentile (e.g. 97.5)
upper_p = (100 - alpha) + (alpha / 2.0)
# retrieve observation at upper percentile
upper = min(1.0, percentile(scores, upper_p))
print('%.1fth percentile = %.3f' % (upper_p, upper))

#%%[markdown]
'''
Hasta aquí tenemos un ejemplo de obtención de la estimación de un parámetro (en este caso el valor medio de una 
población) a partir del resampleo (mediante bootstrapping) de los datos, estimado como el valor medio de las medias 
de dichas muestras <p>
Ahora vamos a aplicar este método para estimar, en lugar de la media de una población, los coeficientes de una
regresión
'''
# %%[markdown]
#### Generación del dataset:

from numpy.random import randn
from numpy import std
from scipy.stats import linregress
# seed random number generator
seed(1)
# prepare data
x = 20 * randn(1000) + 100
y = x + (10 * randn(1000) + 50)
# summarize
print('x: mean=%.3f stdv=%.3f' % (mean(x), std(x)))
print('y: mean=%.3f stdv=%.3f' % (mean(y), std(y)))
# plot
pyplot.scatter(x, y)
pyplot.show()
pyplot.show()

#%%[markdown]
# Make model fit with the whole dataset
b1, b0, r_value, p_value, std_err = linregress(x, y)

yhat_whole_ds = b0 + b1 * x
# plot data and predictions
pyplot.scatter(x, y)
pyplot.plot(x, yhat_whole_ds, color='y')
pyplot.title('Model fit with the whole datasets at once')
pyplot.show()

# %%
import numpy as np 

regression_scores = list()
for _ in range(100):
    # bootstrap sample
    indices = randint(0, 1000, 50)
    x_sample = x[indices]
    y_sample = y[indices]
    
    # simple linear regression model
    b1_sample, b0_sample, r_value, p_value, std_err = linregress(x_sample, y_sample)
    regression_scores.append([b1_sample, b0_sample])

b0_scores = np.array(regression_scores)[:, 1]
b1_scores = np.array(regression_scores)[:, 0]

#%%
b0_estimate = median(b0_scores)
print('b0 50th percentile (median) = %.3f' % median(b0_scores))
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
#b0_lower = max(0.0, percentile(b0_scores, lower_p))
perc_5th = percentile(b0_scores, lower_p)
#print('%.1fth percentile = %.3f' % (lower_p, b0_lower))
# calculate upper percentile (e.g. 97.5)
upper_p = 100 - (alpha / 2.0)
# retrieve observation at upper percentile
#b0_upper = min(1.0, percentile(b0_scores, upper_p))
perc_95th = percentile(b0_scores, upper_p)
#print('%.1fth percentile = %.3f' % (upper_p, b0_upper))
print('confidence interval: ', [perc_5th.round(2), perc_95th.round(2)])
print('b0 estimate: ', [perc_5th.round(2), b0_estimate.round(2), perc_95th.round(2)])

#%%
b1_estimate = median(b1_scores)
print('b1 50th percentile (median) = %.3f' % median(b1_scores))
# calculate 95% confidence intervals (100 - alpha)
alpha = 5.0
# calculate lower percentile (e.g. 2.5)
lower_p = alpha / 2.0
# retrieve observation at lower percentile
#b1_lower = max(0.0, percentile(b1_scores, lower_p))
perc_5th = percentile(b1_scores, lower_p)
#print('%.1fth percentile = %.3f' % (lower_p, b1_lower))
# calculate upper percentile (e.g. 97.5)
upper_p = 100 - (alpha / 2.0)
# retrieve observation at upper percentile
#b1_upper = min(1.0, percentile(b1_scores, upper_p))
perc_95th = percentile(b1_scores, upper_p)
#print('%.1fth percentile = %.3f' % (upper_p, b1_upper))
print('confidence interval: ', [perc_5th.round(2), perc_95th.round(2)])
print('b1 estimate: ', [perc_5th.round(2), b1_estimate.round(2), perc_95th.round(2)])

#%%
# Make model fit with the whole dataset
b1, b0, r_value, p_value, std_err = linregress(x, y)
yhat_whole_ds = b0 + b1 * x
yhat_bootstrapped = b0_estimate + b1_estimate * x
# plot data and predictions
pyplot.scatter(x, y)
pyplot.plot(x, yhat_whole_ds, color='y')
pyplot.plot(x, yhat_bootstrapped, color='r')
pyplot.title('Model fit with the whole dataset at once + with bootstraping of 50 observations each resample')
pyplot.show()

#%%[markdown]
## Check: se está ajustando el modelo correctamente mediante bootstrapping?
# Make model fit with the whole dataset
b1, b0, r_value, p_value, std_err = linregress(x[:100], y[:100])

yhat_sub_ds = b0 + b1 * x
# plot data and predictions
pyplot.scatter(x, y)
pyplot.plot(x, yhat_sub_ds, color='y')
pyplot.plot(x, yhat_bootstrapped, color='r')
pyplot.title('Model fit with only 100 observations VS bootstraping of 100 observations each resample')
pyplot.show()

#%%[markdown]
## We now train with the whole dataset, but the sample sizes in the bootstrapping methodology uses 100-sized samples:
b1, b0, r_value, p_value, std_err = linregress(x, y)

yhat_whole_ds = b0 + b1 * x
yhat_bootstrapped = b0_estimate + b1_estimate * x
# plot data and predictions
pyplot.scatter(x, y)
pyplot.plot(x, yhat_whole_ds, color='y')
pyplot.plot(x, yhat_bootstrapped, color='r')
pyplot.title('Model fit with the whole dataset at once + with bootstraping of 100 observations each resample')
pyplot.show()


#%%[markdown]
'''We now train with the whole dataset, but the sample sizes in the bootstrapping methodology uses 100-sized samples:'''
# WITH WHOLE DATASET
b1, b0, r_value, p_value, std_err = linregress(x, y)
yhat_whole_ds = b0 + b1 * x

### WITH BOOTSTRAP SAMPLES 
regression_scores = list()
for _ in range(100):
    # bootstrap sample
    indices = randint(0, 1000, 550)
    x_sample = x[indices]
    y_sample = y[indices]
    
    # simple linear regression model
    b1_sample, b0_sample, r_value, p_value, std_err = linregress(x_sample, y_sample)
    regression_scores.append([b1_sample, b0_sample])

b0_scores = np.array(regression_scores)[:, 1]
b1_scores = np.array(regression_scores)[:, 0]
b0_estimate = median(b0_scores)
b1_estimate = median(b1_scores)

yhat_bootstrapped = b0_estimate + b1_estimate * x

### MODELS VALIDATION
from sklearn.metrics import mean_squared_error

yhat_whole_ds_rmse = mean_squared_error(y, yhat_whole_ds).round(2)
yhat_bootstrapped_rmse = mean_squared_error(y, yhat_bootstrapped).round(2)

### PLOTS
import plotly.graph_objects as go

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    name='ground_truth'))
fig.add_trace(go.Scatter(x=x, y=yhat_whole_ds,
                    mode='lines+markers',
                    name='yhat_whole_ds'))
fig.add_trace(go.Scatter(x=x, y=yhat_bootstrapped,
                    mode='lines+markers',
                    name='yhat_bootstrapped'))
fig.update_layout(annotations=[dict(xref='paper',
                                    yref='paper',
                                    x=0.5, y=1,
                                    showarrow=False,
                                    text='yhat_whole_ds_rmse: {}, yhat_bootstrapped: {}'.format(yhat_whole_ds_rmse, 
                                                                                                yhat_bootstrapped_rmse))])

fig.show()

#%%[markdown]
'''
Bootstrapping shows us that, even with 50-obervations sized-samples, we get a very good aproximation to the 
right linear regression model, whereas using a 100 sized-sample to train only once is not enough, as expected.

Bootstrapping shows us that, even with 50-obervations sized-samples, we get a very good aproximation to the 
right linear regression model, whereas using a 100 sized-sample to train only once is not enough, as expected

Bootstrapping shows an excellent performance when applying this to samples with the size half of the dataset for instance; 
look for a good compromise to prevent overfitting, but good performance... 
'''



# %%[markdown]
## LA ESTIMACIÓN DE PARAMS CON BOOTSTRAP PARECE IR CORRECTAMENTE
## PROBAMOS PREDICTION INTERVALS:
import numpy as np
from scipy.stats import linregress

regression_scores = list()
example_values_to_predict_its_intervals = np.array([x[3].round(2), x[int(len(x)/2)].round(2), x[int(len(x)/1.5)].round(2)])
#%%
predictions_with_residuals = list()
for _ in range(100):
    # 1.- take a bootstrap sample
    indices = randint(0, 1000, 900)
    x_sample = x[indices]
    y_sample = y[indices]
    # 2.- fit simple linear regression model
    b1_sample, b0_sample, r_value, p_value, std_err = linregress(x_sample, y_sample)
    #print('b1_sample: {}'.format(b1_sample))
    yhat_bootstrapped_sample = b0_sample + (np.multiply(b1_sample, x_sample))
    # 3.- find residuals:
    bootstrap_model_residuals = yhat_bootstrapped_sample - y_sample
    # 4.- take 3 random residuals
    three_random_residuals =np.random.choice(bootstrap_model_residuals, 3)
    # 5.- add each of the 3 residuals to the prediction of the 3 input values of interest
    bootst_predictions = b0_sample + b1_sample * example_values_to_predict_its_intervals
    bootst_predictions_with_residuals = bootst_predictions + three_random_residuals
    predictions_with_residuals.append(bootst_predictions_with_residuals)

predictions_with_residuals=np.array(predictions_with_residuals)

#%%   
pred_int_x1 = (np.percentile(predictions_with_residuals[:, 0], 2), 
               np.percentile(predictions_with_residuals[:, 0], 50),
               np.percentile(predictions_with_residuals[:, 0], 98))

print('pred_int_x1: {}'.format(pred_int_x1))

pred_int_x2 = (np.percentile(predictions_with_residuals[:, 1], 2), 
               np.percentile(predictions_with_residuals[:, 1], 50),
               np.percentile(predictions_with_residuals[:, 1], 98))

print('pred_int_x2: {}'.format(pred_int_x2))

pred_int_x3 = (np.percentile(predictions_with_residuals[:, 2], 2), 
               np.percentile(predictions_with_residuals[:, 2], 50),
               np.percentile(predictions_with_residuals[:, 2], 98))

print('pred_int_x3: {}'.format(pred_int_x3))


#%%[markdown]
## VS el resultado obtenido entrenando sobre todo el set:
b1_whole, b0_whole, r_value, p_value, std_err = linregress(x, y)
yhat_whole = b0_whole + (np.multiply(b1_whole, example_values_to_predict_its_intervals))

yhat_whole

# %%[markdown]
# Cuanto mayor tamaño tengan los bootstrap smaples esperamos (como parece pasar) mejor resultado de predicción


# %%

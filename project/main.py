
#%%
import logging
from datetime import datetime

logger_name=str(datetime.now().date()) + '.log'
logging.basicConfig(filename=logger_name,level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(logger_name)

from project.dataset_plots import generic_plots
from project.statistical_tests import distributions_generator
import pandas as pd
import numpy as np 
# %%
try:
    dist_generator=distributions_generator.Distributions_generator()
    normal_data_1 = dist_generator.generate_random_normal_distribution(100, 20, 1000)
    normal_data_2 = dist_generator.generate_random_normal_distribution(50, 10, 1000) + normal_data_1
    
    normal_data_df = pd.DataFrame({'normal_data_1': normal_data_1, 'normal_data_2': normal_data_2})
    generic_plots_obj = generic_plots.Dataset_plots(normal_data_df)
    generic_plots_obj.plot_series_histogram('normal_data_1', 
                                            figsize_rows=5, figsize_cols=3)
    generic_plots_obj.plot_series_histogram('normal_data_2', 
                                            figsize_rows=5, figsize_cols=3)

    from numpy.random import randn
    data = pd.Series(5 * randn(100) + 50)
    generic_plots_obj.plot_QQ_plot(data)

except Exception as exc:
        logger.exception('raised exception at {}: {}'.format(logger.name, exc))

#%%
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(normal_data_1, normal_data_2)
print('Pearsons correlation: %.3f' % corr)

#%%[markdown]
## Repetimos proceso para datos no normales:
# %%
try:
    dist_generator=distributions_generator.Distributions_generator()
    random_data_1 = dist_generator.generate_random_distribution(20, 1000)
    random_data_2 = dist_generator.generate_random_distribution(10, 1000) #+ random_data_1
    
    random_data_df = pd.DataFrame({'random_data_1': random_data_1, 'random_data_2': random_data_2})
    generic_plots_obj = generic_plots.Dataset_plots(random_data_df)
    generic_plots_obj.plot_series_histogram('random_data_1', 
                                            figsize_rows=5, figsize_cols=3)
    generic_plots_obj.plot_series_histogram('random_data_2', 
                                            figsize_rows=5, figsize_cols=3)

    generic_plots_obj.plot_QQ_plot(random_data_1)
    generic_plots_obj.plot_QQ_plot(random_data_2)

except Exception as exc:
        logger.exception('raised exception at {}: {}'.format(logger.name, exc))

#%%[markdown]
## Vemos que claramente no son distribuciones normales: veamos qué resultado obtenemos si aplicamos Pearson corr
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr_2, _ = pearsonr(random_data_1, random_data_2)
print('Pearsons correlation: %.3f' % corr_2)

#%%[markdown]
### Mismo coef. de Pearson corr! Y si aplicamos el de Spearman tanto a la normal como random?
# calculate spearman✬s correlation
from scipy.stats import spearmanr
coef_normal_data, p = spearmanr(normal_data_1, normal_data_2)
print('Spearmans correlation coefficient: %.3f' % coef_normal_data)
# interpret the significance
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)
#%%
coef_rand_data, p = spearmanr(random_data_1, random_data_2)
print('Spearmans correlation coefficient: %.3f' % coef_rand_data)
# interpret the significance
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

#%%

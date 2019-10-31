#%%
import logging
from datetime import datetime

logger_name=str(datetime.now().date()) + '.log'
logging.basicConfig(filename=logger_name,level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(logger_name)

from project.dataset_plots import generic_plots
#from dataset_plots import generic_plots
from project.statistical_tests import distributions_generator
#from statistical_tests import distributions_generator
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

    generic_plots_obj.plot_QQ_plot(normal_data_1)
    generic_plots_obj.plot_QQ_plot(normal_data_2)

except Exception as exc:
        logger.exception('raised exception at {}: {}'.format(logger.name, exc))

# %%
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
    
    random_data_df = pd.DataFrame({'random_data_1': random_data_1, 
                                   'random_data_2': random_data_2})
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

#%%[markdown]
""" The random distributions are not correlated, as expected. 
Pearson's and Spearman's correlation coefficients are the same 
in this case, let's try with another non-parametric distributions.
"""
# %%
try:
    random_beta_data_1 = dist_generator.generate_beta_distribution(2, 3, 10000)
    random_beta_data_2 = dist_generator.generate_beta_distribution(3, 3, 10000)
    
    random_beta_data_df = pd.DataFrame({'random_beta_data_1': random_beta_data_1, 
                                   'random_beta_data_2': random_beta_data_2})
    generic_plots_obj = generic_plots.Dataset_plots(random_beta_data_df)
    generic_plots_obj.plot_series_histogram('random_beta_data_1', 
                                            figsize_rows=5, figsize_cols=3)
    generic_plots_obj.plot_series_histogram('random_beta_data_2', 
                                            figsize_rows=5, figsize_cols=3)

    generic_plots_obj.plot_QQ_plot(random_beta_data_1)
    generic_plots_obj.plot_QQ_plot(random_beta_data_2)

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

# %%[markdown]
""" As another statistical test, we can check if two distributions come from 
    the same population. In this case, our null hypothesis is that the two samples
    comme from the same distribution (i.e. their mean values do not differ a significant
    amount)"""
# %%
dist_generator=distributions_generator.Distributions_generator()
normal_data_1 = dist_generator.generate_random_normal_distribution(100, 20, 1000)
normal_data_2 = dist_generator.generate_random_normal_distribution(50, 10, 1000) + normal_data_1
# %%[markdown]
### We now in advance that 'normal_data_1' and 'normal_data_2' differ, so we should get a tiny p-value:
# %%     
from project.statistical_tests import hypothesis_tester

data_diff = normal_data_1, normal_data_2
data_equal = normal_data_2, normal_data_2
hypothesis_tester_obj_diff = hypothesis_tester.DiffMeansPermute(data_diff)
print('pvalue for normal_data_1, normal_data_2: ', hypothesis_tester_obj_diff.PValue())
#%%
hypothesis_tester_obj_equal = hypothesis_tester.DiffMeansPermute(data_equal)
print('pvalue for normal_data_2, normal_data_2: ', hypothesis_tester_obj_equal.PValue())


#%%
def add_value_to_input(input_value, value_to_add):
    return input_value+value_to_add

def apply_operation_to_all_elements(data_container, operation_function):
    try:
        import pandas as pd 
        if type(data_container)==pd.core.frame.DataFrame:
            return data_container.applymap(operation_function)
        elif type(data_container)==pd.core.frame.Series:
            return data_container.apply(operation_function)

    except Exception as exc:
        raise exc

#%%
data_container_ = normal_data_df.iloc[:2]
data_container_.applymap(lambda x: add_value_to_input(x, 1))
# %%
data_container_ = normal_data_df.iloc[:2].normal_data_2
data_container_.apply(lambda x: add_value_to_input(x, 1))

# %%

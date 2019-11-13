# %%
import pandas as pd 
import numpy as np

# source: https://pandas.pydata.org/pandas-docs/version/0.20/cookbook.html
def makeCartesianProduct(array_x, array_y):
    import pandas as pd
    import itertools

    return pd.DataFrame.from_records(itertools.product(array_x.reshape(-1, ), array_y.reshape(-1, )), columns=['x', 'y'])

jobs_ids_array = np.random.choice(range(10), 6, replace=False) 
jobs_ids_cartesian_prod = makeCartesianProduct(jobs_ids_array, jobs_ids_array)
jobs_ids_cartesian_prod

#%%
def check_if_job_ids_equality(index, df=None):
    if df.iloc[index][0]==df.iloc[index][1]:
        return False
    else: 
        return True

jobs_ids_cartesian_prod_indexes = jobs_ids_cartesian_prod.index
equal_indexes = pd.Series(jobs_ids_cartesian_prod_indexes).apply(lambda x: check_if_job_ids_equality(index=x, df=jobs_ids_cartesian_prod))
jobs_ids_cartesian_prod['equal_indexes'] = equal_indexes
jobs_ids_cartesian_prod

# %%[markdown]
# # # Ahora rellenamos el contenido del df resultado:
#%%
results_df = pd.DataFrame(index = jobs_ids_array, columns = jobs_ids_array).fillna(True)
results_df_numpied=results_df.to_numpy()
np.fill_diagonal(results_df_numpied, False)
results_df=pd.DataFrame(results_df_numpied)
results_df
#%%
def f1(x):
    if x==True:
        return 1
    else:
        return False

def f2(x):
    if x==True:
        return 2
    else:
        return False

def f3(x):
    if x==True:
        return 3
    else:
        return False

functions_dictionary = {"f1": f1,
                  "f2": f2,
                  "f3": f3}

def apply_function(functions_dict, function_name, x):
    return functions_dict[function_name](x)
 
rule_index_name = 'f2'
results_df_app_map = results_df.applymap(lambda x: apply_function(functions_dictionary, rule_index_name ,x))
results_df_app_map

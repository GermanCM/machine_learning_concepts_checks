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






# %%[markdown]
# # # Implementación:

import numpy as np
import time 
import pandas as pd 

start_time = time.time()

jobs_df = pd.DataFrame({'A': pd.Series(np.random.randint(1, size=10)), 
                        'B': pd.Series(np.random.randint(1, size=10)),
                        'C': pd.Series(np.random.randint(1, size=10)), 
                        'D': pd.Series(np.random.randint(1, size=10))})

# source: https://pandas.pydata.org/pandas-docs/version/0.20/cookbook.html
def makeCartesianProduct(array_x, array_y):
    import pandas as pd
    import itertools 

    return pd.DataFrame.from_records(itertools.product(array_x.reshape(-1, ), array_y.reshape(-1, )), columns=['x', 'y'])

jobs_ids_array = np.array(jobs_df.index)  
jobs_ids_cartesian_prod = makeCartesianProduct(jobs_ids_array, jobs_ids_array)

end_time = time.time()
print('operation time con prod. cartesiano: ', end_time-start_time)
print('número de combinaciones: ', len(jobs_ids_cartesian_prod))
# %%[markdown]
## VS con bucles for:
# %%
start_time = time.time()
l = len(jobs_df)
jobs_ids_combinations = pd.DataFrame(columns=['x', 'y'])
for i in range(l):
    for j in range(l):
        jobs_ids_combinations = jobs_ids_combinations.append(pd.Series({'x': i, 'y': j}), ignore_index=True)

end_time = time.time()
print('operation time con bucle for: ', end_time-start_time)
print('número de combinaciones: ', len(jobs_ids_combinations))
#%%
def check_if_job_ids_equality(index, df=None):
    if df.iloc[index][0]==df.iloc[index][1]:
        #return (df.iloc[index][0], df.iloc[index][1]), False
        return False
    else: 
        #return (df.iloc[index][0], df.iloc[index][1]), True
        return True
start_time = time.time()
jobs_ids_cartesian_prod_indexes = jobs_ids_cartesian_prod.index
equal_indexes = pd.Series(jobs_ids_cartesian_prod_indexes).apply(lambda x: check_if_job_ids_equality(index=x, df=jobs_ids_cartesian_prod))
end_time = time.time()
print('operation time with apply(check_if_job_ids_equality): ', end_time-start_time)

# %%
jobs_ids_combinations['equal_indexes'] = equal_indexes
jobs_ids_combinations

# %%[markdown]
# # # VS
# %%
start_time = time.time()

equal_indexes = pd.Series([])
for i in jobs_ids_combinations.index:
    if jobs_ids_cartesian_prod.iloc[i][0]==jobs_ids_cartesian_prod.iloc[i][1]:
        equal_indexes = equal_indexes.append(pd.Series(False))
    else: 
        equal_indexes = equal_indexes.append(pd.Series(True))

end_time = time.time()
print('operation time with for loop: ', end_time-start_time)


# %%
jobs_ids_combinations['equal_indexes_for_loop'] = equal_indexes.values
jobs_ids_combinations




# esto que sigue es para aplicar una función de u diccionario
# %%
def p1(x):
    return x+1

def p2(x):
    return x+4

myDict = {
    "p1": p1,
    "p2": p2
}

def myMain(dict_f, name, x):
    return dict_f[name](x)
 
# %%
myMain(myDict, 'p1', 3)

# %%

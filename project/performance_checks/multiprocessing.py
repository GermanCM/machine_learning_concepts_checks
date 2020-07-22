#%%
print('Multiprocessing POC')

# create random data 
random_data = [randint(10,1000) for i in range(1,10000001)]

data = pd.DataFrame({'Number' : random_data})
data.head()

#########################################################
# FUENTE: https://medium.com/@urban_institute/using-multiprocessing-to-make-python-code-faster-23ea5ef996ba
#%%
import time
import multiprocessing 

#%%
import multiprocessing as mp
print('cpu counter has detected {} logical cores'.format(mp.cpu_count()))

#%%
def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'

def multiprocessing_func(x):
    y = x*x
    time.sleep(2)
    print('{} squared results in a/an {} number'.format(x, basic_func(y)))
    
if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(0,10):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
    print('That took {} seconds'.format(time.time() - starttime))

# %%

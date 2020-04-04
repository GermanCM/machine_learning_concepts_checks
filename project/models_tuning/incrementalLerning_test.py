# %%[markdown]
## Incremental learning checks
### source: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# %%[markdown]
### Dataset extraction
import pandas as pd 
from sklearn import datasets

ds = datasets.load_iris()
mask_target_not_2 = [(ds['target']==0)|(ds['target']==1)]
#%%
features_values_2_D = ds['data'][:, 0:2][mask_target_not_2]

#%%
target_values = ds['target'][mask_target_not_2]

#%%
X = features_values_2_D
y = target_values

# %%
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
        Parameters
        ------------
        eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes over the training dataset.
        shuffle : bool (default: True)
        Shuffles training data every epoch if True
        to prevent cycles.
        random_state : int
        Random number generator seed for random weight
        initialization.
        Attributes
        -----------
        w_ : 1d-array
        Weights after fitting.
        cost_ : list
        Sum-of-squares cost function value averaged over all
        training samples in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10,
                    shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and
        n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1: #if we have more than one sample
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        
        return self
        
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        import numpy as np

        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=1 + m)
        
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
    
        return cost

    # adaline schema: project\models_tuning\pics\adaline_schema.PNG

    def net_input(self, X):
        """Calculate net input"""
        
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""

        return np.where(self.activation(self.net_input(X))
                        >= 0.0, 1, -1)

def calculate_cost(predictions, true_values):
    """ Sum of Squared Error Cost function """
    import numpy as np

    cost = np.sum([np.square(predictions[i]-true_values[i]) for i in range(0, len(predictions))])

    return cost

# %%
import numpy as np 
'''
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
'''
import pandas as pd 

path = r'C:\Users\gcabreram\Google Drive\mi_GitHub\machine_learning_concepts_checks\project\models_tuning\data\iris_ds.csv'
df = pd.read_csv(path, sep=';', header=None, encoding='utf-8')
df.tail()

#%%
import matplotlib.pyplot as plt
import numpy as np
# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
#extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# %%[markdown]
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

# %%[markdown]
### Ahora probamos a entrenar el modelo sobre un 90% de los datos con fit, y el 10% de uno en uno con partial_fit
# %%[markdown]
# Shuffle data and get traiing and test data
import numpy as np

r = np.random.RandomState(42).permutation(len(X_std))
X_std_shuffled = X_std[r]  
y_shuffled = y[r]

train_len_90 = int(0.9*len(X_std_shuffled))
X_std_train = X_std_shuffled[:train_len_90]
y_train = y_shuffled[:train_len_90]
X_std_test = X_std_shuffled[train_len_90:]
y_test = y_shuffled[train_len_90:]

#%%
### Train models on the whole train dataset and tested on 10% of data
ada = AdalineSGD(n_iter=100, eta=0.01, random_state=1)
ada_model_whole_data = ada.fit(X_std_train, y_train)
print('coefficients with all the dataset and 100 iters: {}'.format(ada_model_whole_data.w_))
predictions = [ada_model_whole_data.net_input(X_std_test[i]) for i in range(0, len(X_std_test))]
test_cost = calculate_cost(predictions, y_test)
print('cost on test set with all the dataset and 100 iters: {}'.format(test_cost)) 
#%%
import matplotlib.pyplot as plt 

plot_decision_regions(X_std, y, classifier=ada_model_whole_data)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada_model_whole_data.cost_) + 1), ada_model_whole_data.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
#%%
'''
from mlxtend.plotting import plot_decision_regions as mlx_plot_dec_reg
# Plotting decision regions
mlx_plot_dec_reg(X_std_test, y_test, clf=ada_model_whole_data)
'''
#%%[markdown]
### Train on 80% of training data (to simulate online learning afterwords)
initial_80_percent_train_length = int(0.8*len(X_std_train))
X_std_train_80 = X_std[:initial_80_percent_train_length]
y_train_80 = y[:initial_80_percent_train_length]
assert len(y_train_80)<len(y_train)

ada = AdalineSGD(n_iter=100, eta=0.01, random_state=1)
ada_model_80_ = ada.fit(X_std_train_80, y_train_80)
print('coefficients with all the dataset and 100 iters: {}'.format(ada_model_80_.w_))
predictions_80 = [ada_model_80_.net_input(X_std_test[i]) for i in range(0, len(X_std_test))]
test_cost_80 = calculate_cost(predictions_80, y_test)
print('cost on test set with 80 percent dataset and 100 iters: {}'.format(test_cost_80)) 
#%%
if test_cost<test_cost_80:
    print('cost from model trained with all data is lower than cost with model trained with 80 percent as expected')
#%%
plot_decision_regions(X_std, y, classifier=ada_model_80_)
plt.title('Adaline - Stochastic Gradient Descent with 80 percent data')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada_model_80_.cost_) + 1), ada_model_80_.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
#%%
'''
from mlxtend.plotting import plot_decision_regions as mlx_plot_dec_reg
# Plotting decision regions
mlx_plot_dec_reg(X_std_test, y_test, clf=ada_model_whole_data)
'''
#%%[markdown]
### Train on 60% of training data (to simulate online learning afterwords)
initial_60_percent_train_length = int(0.6*len(X_std_train))
X_std_train_60 = X_std[:initial_60_percent_train_length]
y_train_60 = y[:initial_60_percent_train_length]
assert len(y_train_60)<len(y_train)

ada = AdalineSGD(n_iter=100, eta=0.01, random_state=1)
ada_model_60_ = ada.fit(X_std_train_60, y_train_60)
print('coefficients with all the dataset and 100 iters: {}'.format(ada_model_60_.w_))
predictions_60 = [ada_model_60_.net_input(X_std_test[i]) for i in range(0, len(X_std_test))]
test_cost_60 = calculate_cost(predictions_60, y_test)
print('cost on test set with 60 percent dataset and 100 iters: {}'.format(test_cost_60)) 
#%%
if test_cost_80<test_cost_60:
    print('cost from model trained with 80 percent data is lower than cost with model trained with 60 percent as expected')
#%%[markdown]
### Plot model decision boundary on 60 percent of data
plot_decision_regions(X_std, y, classifier=ada_model_60_)
plt.title('Adaline - Stochastic Gradient Descent with 60 percent data')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada_model_60_.cost_) + 1), ada_model_60_.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# %%[markdown]
### Y ahora comprobamos el efecto de incremental learning:
current_ada_model = ada_model_60_
for i in range(1, len(y)-initial_60_percent_train_length-1):
    index = (initial_60_percent_train_length + i) - initial_60_percent_train_length+(i-1)
    X_single_sample = X_std[index]
    y_single_sample = y[index]

    ada_model_60_partially_refitted = current_ada_model.partial_fit(X_single_sample, y_single_sample)
    predictions_60 = [ada_model_60_partially_refitted.net_input(X_std_test[i]) for i in range(0, len(X_std_test))]
    test_cost_60_increm = calculate_cost(predictions_60, y_test)
    current_ada_model = ada_model_60_partially_refitted
    print('cost of model trained with 60 percent + 1 sample of the dataset: {}'.format(test_cost_60_increm)) 
#%%
### Plot model decision boundary on whole data
plot_decision_regions(X_std, y, classifier=current_ada_model)
plt.title('Adaline - SGD with 60 percent + incremental data')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(current_ada_model.cost_) + 1), current_ada_model.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# %%[markdown]
### MI PARTIAL FIT PARECE FUNCIONAR REENTRENÁNDOSE CON CADA MUESTRA ADICIONAL SOBRE EL MODELO YA PRE ENTRENADO CON PARTE DEL DATASET
### NO OBSTANTE, PARECE ENTRENARSE MEJOR CUANDO "VE" TODOS LOS DATOS SIMULTÁNEAMENTE
### MIRAR ANIMACIÓN DE GRÁFICAS (EVOLUCIÓN DE LA SUPERFICIE DE DECISIÓN) EN STREAMLIT

# %%

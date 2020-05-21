#%% [markdown]
# ### GridSearchCV simple example

#%%
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
iris

#%% [markdown]
# #### Binary target for this example

#%%
two_classes_mask = (iris.target==1)|(iris.target==2)
iris_attributes = iris.data[two_classes_mask]
iris_target = iris.target[two_classes_mask]
iris_target

#%% [markdown]
# ### GridSearchCV implementation

#%%
grid_params = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
scoring_metrics = ['precision'] #, 'roc_auc']
svc = svm.SVC(gamma="scale")

iris_attributes_train, iris_target_train = iris_attributes[10:90], iris_target[10:90] 

clf = GridSearchCV(estimator=svc, param_grid=grid_params, cv=5, scoring=scoring_metrics, refit='precision')
clf.fit(iris_attributes_train, iris_target_train)     

#%% [markdown]
# #### Acces to results

#%%
clf.cv_results_['mean_test_precision']

#%%
clf.cv_results_

#%%
import numpy as np 

print('best score: {}'.format(clf.best_score_))
print('mean value of the best model k scores: {}'.format(np.array([1, 1, 1, 1, 0.88888889]).mean()))

#%% [markdown]
# ### <font color='green'> Best params found for roc_auc </font>

#%%
clf.best_params_

#%% [markdown]
# ### <font color='green'> Best estimator predictions </font>

#%%
test_sample=iris_attributes[3].reshape(1, -1)
clf.best_estimator_.predict(test_sample)

#%%[markdown]
''' Check: 'best_estimator_' ya te devuelve el modelo 
reentrenado en todo el set'''
#%%
from sklearn.metrics import roc_auc_score, precision_score
import numpy as np

iris_attributes_valid = np.append(iris_attributes[:10],iris_attributes[90:], axis=0)
iris_target_valid = np.append(iris_target[:10],iris_target[90:])

best_model = clf.best_estimator_
refit_best_model = best_model.fit(iris_attributes_train, iris_target_train)

print('best_model precision: ', precision_score(iris_target_valid, 
                     best_model.predict(iris_attributes_valid)))
print('refit_best_model precision: ', 
       precision_score(iris_target_valid, 
                     refit_best_model.predict(iris_attributes_valid)))

# %%
from sklearn.svm import SVC

no_tuned_model = SVC(C= 2, kernel= 'rbf') 
no_tuned_model.fit(iris_attributes_train, iris_target_train)
print('best_model precision: ', 
       precision_score(iris_target_valid, 
                       no_tuned_model.predict(iris_attributes_valid)))

# %%
# y ahora comparo con la precision que me devuelve incluyéndolo como parte del grid search
grid_params = {'kernel':('linear', 'rbf'), 'C':[1, 2, 10]}
scoring_metrics = ['precision'] #, 'roc_auc']
svc = svm.SVC(gamma="scale")

iris_attributes_train, iris_target_train = iris_attributes[10:90], iris_target[10:90] 

clf = GridSearchCV(estimator=svc, param_grid=grid_params, cv=5, scoring=scoring_metrics, refit='precision')
clf.fit(iris_attributes_train, iris_target_train)     

#%%
clf.cv_results_['mean_test_precision']

# %%
clf.cv_results_

# %%
best_model = clf.best_estimator_
refit_best_model = best_model.fit(iris_attributes_train, iris_target_train)

print('best_model precision: ', precision_score(iris_target_valid, 
                     best_model.predict(iris_attributes_valid)))
print('refit_best_model precision: ', 
       precision_score(iris_target_valid, 
                     refit_best_model.predict(iris_attributes_valid)))

# %%[markdown]
## vs

# %%
clf.best_score_

# %%[markdown]
'''Conclusión: diría que el hecho de que el best_score_ sea menor que el score del 
   mismo modelo (i.e. el best_estimator_) se debe a que hay más variabilidad en los test scores
   en el proceso k-fold CV, y el valor medio suele dar más bajo'''


# %%

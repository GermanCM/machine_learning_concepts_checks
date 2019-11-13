#%% [markdown]
# ### GridSearchCV simple example

#%%
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()

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
scoring_metrics = ['precision', 'roc_auc']
svc = svm.SVC(gamma="scale")

clf = GridSearchCV(estimator=svc, param_grid=grid_params, cv=5, scoring=scoring_metrics, refit='roc_auc')
clf.fit(iris_attributes[20:88], iris_target[20:88])     

#%% [markdown]
# #### Acces to results

#%%
clf.cv_results_

#%% [markdown]
# ### <font color='green'> Best params found for roc_auc </font>

#%%
clf.best_params_

#%% [markdown]
# ### <font color='green'> Best estimator found for roc_auc </font>

#%%
clf.best_estimator_

#%% [markdown]
# ### <font color='green'> Best estimator predictions </font>

#%%
test_sample=iris_attributes[3].reshape(1, -1)
clf.best_estimator_.predict(test_sample)

#%%[markdown]
''' Check: 'best_estimator_' ya te devuelve el modelo 
reentrenado en todo el set'''
#%%
from sklearn.metrics import roc_auc_score
import numpy as np

iris_test_attrs = np.append(iris_attributes[:20],iris_attributes[88:], axis=0)
iris_test_target = np.append(iris_target[:20],iris_target[88:])

best_model = clf.best_estimator_
refit_best_model = clf.best_estimator_.fit(iris_attributes[20:88], iris_target[20:88])

print('best_model roc_auc_score: ', 
       roc_auc_score(iris_test_target, 
                     best_model.predict(iris_test_attrs)))
print('refit_best_model roc_auc_score: ', 
       roc_auc_score(iris_test_target, 
                     refit_best_model.predict(iris_test_attrs)))

#%%
iris_test_target
best_model.predict(iris_test_attrs)
#%%
best_model
refit_best_model
#%%


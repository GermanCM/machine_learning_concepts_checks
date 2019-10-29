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
clf.fit(iris_attributes, iris_target)     

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
test_sample=iris_attributes[-3].reshape(1, -1)
clf.best_estimator_.predict(test_sample)


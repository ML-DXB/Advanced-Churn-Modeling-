# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:22:11 2018

@author: GeorgiosTzallasRegka
"""
# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline


# load data
df = pd.read_csv('pima-indians-diabetes.csv', delimiter=",")
df2 = pd.read_csv('pima-indians-diabetes2.csv', delimiter=",")
dataset.info


#converting test column which is categorical to numerical negatif=0 ,and positif=1
df2['test'].replace(['negatif','positif'],[0,1],inplace=True)

plt.figure(figsize=(12,6))
sb.heatmap(df2[df2.columns[0:]].corr(),annot=True)

#As, we can see Glucose level and age are two main factors that are cooreralted with diabetes in this case 

sb.jointplot(x='test',y='age',data=df2)

# split data into X and y using iloc 
X = df2.iloc[:,0:8]
y = df2.iloc[:,8]

model = XGBClassifier()
model.fit(X, y)


# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot single tree
plot_tree(model)
plt.show()
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict_proba(X_test)


# Evaluate predictions
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("ROC AUC ", roc_auc)





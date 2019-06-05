#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs/mini_lab'))
	print(os.getcwd())
except:
	pass
#%%
#Add library references
import pandas as pd
import numpy as np
import seaborn as sns
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
df_cols = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket'
]
cat_cols = [
    "workclass",
    "education_num",
    "marital_status", 
    "race", 
    "gender"]

cont_cols = [
    "age", 
    "capital_gain", 
    "capital_loss",                  
    "hours_per_week"]

target_col = "target"

drop_cols = [
    'fnlwgt',
    "native_country", 
    "education", 
    "occupation",           
    "relationship"]

#%%
df_training = pd.read_csv("data/adult-training.csv",
    names=df_cols, 
    skipinitialspace = True)

df_test = pd.read_csv("data/adult-test.csv",
    names = df_cols,
    skipinitialspace = True,
    skiprows=1)

#%%
df_training[target_col] = (df_training["income_bracket"]
    .apply(lambda x: ">50K" in x)).astype(int)
df_test[target_col] = (df_test["income_bracket"]
    .apply(lambda x: ">50K" in x)).astype(int)


#%%

df_training.drop(drop_cols,axis=1,inplace=True)

df_test.drop(drop_cols,axis=1,inplace=True)

#%%
def convert_dummy(df,cols):
    dummies = []
    for cat in cols:
        dummy = pd.get_dummies(df[cat],
        drop_first=True)
        dummies.append(dummy)
        df.drop(cat,axis=1,inplace=True)
    
    return pd.concat([df,*dummies], axis=1)

df_training_dum = convert_dummy(df_training.copy(),cat_cols)
df_test_dum = convert_dummy(df_test.copy(),cat_cols)

#%%
X_train = df_training_dum.drop(columns=["income_bracket",target_col])
y_train = df_training_dum[target_col]
X_test = df_test_dum.drop(columns=["income_bracket",target_col])
y_test = df_test_dum[target_col]
# X = np.c_[np.ones((X.shape[0], 1)), X]
# y = y[:, np.newaxis]
# theta = np.zeros((X.shape[1], 1))

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

#%%
print(classification_report(y_test,predictions))
print("Accuracy:",accuracy_score(y_test, predictions))

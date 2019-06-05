#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs'))
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
df_headers = [
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
df_census = pd.read_csv("data/adult-training.csv",
    names=df_headers, 
    index_col=False)

df_census.head(10)

#%%
print("Structure of data:\n",df_census.shape,"\n")
print("Count of missing values:\n",df_census.isnull().sum().sort_values(ascending=False),"\n")


#%%
for i in df_headers:
    
    print(i, 
    "type: {}".format(df_census[i].dtype),
    "# unique: {}".format(df_census[i].nunique()),
    sep="\n  ", end="\n\n")
    
print("Summary Statistic's:\n",round(df_census.describe().unstack(),2),"\n")

#%%
education_categories = list(df_census.education.unique())

print(df_census.groupby(['education','gender'])['gender'].count().unstack())
    
#%%
secondary = [
    'education',
    'gender',
    'race',
    'marital_status',
    'relationship',
    'native_country',
    'workclass'
    ]
for i in secondary:
    print(df_census.groupby([i,'income_bracket'])[i].count().unstack(), end="\n\n")

#%%

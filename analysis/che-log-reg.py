#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs/mini_lab'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#%%
COLUMNS = ["age", "workclass", "fnlwgt", "education",  "education_num", "marital_status", "occupation", "relationship", "race", 
"gender", "capital_gain", "capital_loss", "hours_per_week",  "native_country", "income"]

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", 
                       "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


#%%
df_train = pd.read_csv("data/adult-training.csv",names = COLUMNS,)

df_test = pd.read_csv("data/adult-test.csv",names = COLUMNS,skiprows=1)


#%%
# make 1 master csv file from test and train
df_master = pd.concat([df_test, df_train], axis=0)
df_master.info()


#%%
df_master['income'].value_counts()


#%%
# Change income bracket values that have a . at end and remove space 
df_master = df_master.replace(to_replace=(' >50K.', ' >50K'),value='>50K')
df_master = df_master.replace(to_replace=(' <=50K.', ' <=50K'),value='<=50K')


#%%
# make 1 for >50k and 0 for <=50K in income column
df_master['income'] = df_master['income'].apply(lambda x: 1 if x=='>50K' else 0)


#%%
df_master.education.unique()


#%%
# Make all education values that didnt graduate HS 'No Diploma'
# the 2 associate values to Associates
# some college and HS-grad to Diploma
df_master = df_master.replace(to_replace=(' 1st-4th', ' 5th-6th',' 7th-8th',' 9th', ' 10th', ' 11th', ' 12th', ' Preschool'),value=' No Diploma')
df_master = df_master.replace(to_replace=(' Assoc-acdm', ' Assoc-voc'),value=' Associates')
df_master = df_master.replace(to_replace=(' Some-college', ' HS-grad'),value=' Diploma')

df_master['education'] = df_master['education'].str.strip()


#%%
# Put countries in their native region continent
df_master = df_master.replace(to_replace=(' United-States', ' Honduras', ' Mexico',' Puerto-Rico',' Canada', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Guatemala', ' El-Salvador' ),value='North America')
df_master = df_master.replace(to_replace=(' Cuba', ' Jamaica', ' Trinadad&Tobago', ' Haiti', ' Dominican-Republic' ),value='Caribbean')
df_master = df_master.replace(to_replace=(' South', ' Cambodia',' Thailand',' Laos', ' Taiwan', ' China', ' Japan', ' India', ' Iran', ' Philippines', ' Vietnam', ' Hong'),value='Asia')
df_master = df_master.replace(to_replace=(' England', ' Germany', ' Portugal', ' Italy', ' Poland', ' France', ' Yugoslavia',' Scotland', ' Greece', ' Ireland', ' Hungary', ' Holand-Netherlands'),value='Europe') 
df_master = df_master.replace(to_replace=(' Columbia', ' Ecuador', ' Peru'),value='South America')
df_master = df_master.replace(to_replace=(' ?'),value='Other')                                       


#%%
sns.countplot(x = 'income', data = df_master, palette='hls')
plt.show()
# plt.savefig('count plot')


#%%
df_master.groupby('income').mean()


#%%
df_master.info()


#%%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#%%
target = df_master['income'].values
features = df_master[['age', 'workclass', 'fnlwgt', 'education',  'education_num', 'marital_status', 'occupation', 'relationship', 'race', 
"gender", 'capital_gain', 'capital_loss', 'hours_per_week',  'native_country',]].copy()


#%%
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)


#%%
preprocess = make_column_transformer(
    (['age', 'fnlwgt', 'education_num','capital_gain', 'capital_loss', 'hours_per_week'] ,make_pipeline(SimpleImputer(), StandardScaler())),
    (['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country'], OneHotEncoder()))


#%%
model = make_pipeline(
    preprocess,
    LogisticRegression(solver='liblinear'))


#%%
model.fit(X_train, y_train)
print("logistic regression score: %f" % model.score(X_test, y_test))


#%%




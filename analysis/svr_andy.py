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
# Change income bracket values that have a . at end and remove space 
df_master = df_master.replace(to_replace=(' >50K.', ' >50K'),value='>50K')
df_master = df_master.replace(to_replace=(' <=50K.', ' <=50K'),value='<=50K')


#%%
# make 1 for >50k and 0 for <=50K in income column
df_master['income'] = df_master['income'].apply(lambda x: 1 if x=='>50K' else 0)

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
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#Transforming categorical to numerical
df_master['workclass'] = le.fit_transform(df_master['workclass'])
df_master['education'] = le.fit_transform(df_master['education'])
df_master['marital_status'] = le.fit_transform(df_master['marital_status'])
df_master['occupation'] = le.fit_transform(df_master['occupation'])
df_master['relationship'] = le.fit_transform(df_master['relationship'])
df_master['race'] = le.fit_transform(df_master['race'])
df_master['gender'] = le.fit_transform(df_master['gender'])
df_master['native_country'] = le.fit_transform(df_master['native_country'])

df_master.head()

#split up the target 'income' out from df_master
cols = [col for col in df_master.columns if col not in ['income_bracket']]
data = df_master[cols]
target = df_master['income']
data.head(2)



#%%
from sklearn.model_selection import train_test_split
setsdata_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.20, random_state = 10)
#%%

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svc_model = LinearSVC(random_state=0)
pred = svc_model.fit(data_train, target_train).predict(data_test)
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

#%%

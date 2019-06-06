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
    "marital_status", 
    "occupation",
    "race", 
    "gender",
    "relationship"]

cont_cols = [
    "age", 
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"]

drop_cols = [
    'fnlwgt',
    "native_country",
    "education"]

target_col = "target"

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
# Make all education values that didnt graduate HS 'No Diploma'
# the 2 associate values to Associates
# some college and HS-grad to Diploma
replace_edu_no = ('1st-4th', '5th-6th','7th-8th','9th', '10th', '11th', '12th', 'Preschool')
replace_edu_associate = ('Assoc-acdm', 'Assoc-voc')
replace_edu_diploma = ('Some-college', 'HS-grad')

df_training.education = df_training.education.replace(to_replace=replace_edu_no,value='No Diploma')
df_training.education = df_training.education.replace(to_replace=replace_edu_associate,value='Associates')
df_training.education = df_training.education.replace(to_replace=replace_edu_diploma,value='Diploma')

df_test.education = df_test.education.replace(to_replace=replace_edu_no,value='No Diploma')
df_test.education = df_test.education.replace(to_replace=replace_edu_associate,value='Associates')
df_test.education = df_test.education.replace(to_replace=replace_edu_diploma,value='Diploma')

df_training['education'] = df_training['education'].str.strip()
df_test['education'] = df_test['education'].str.strip()

#%%
# Put countries in their native region continent
replace_northA = ('United-States', 'Honduras', 'Mexico','Puerto-Rico','Canada', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Guatemala', 'El-Salvador')
replace_carib = ('Cuba', 'Jamaica', 'Trinadad&Tobago', 'Haiti', 'Dominican-Republic')
replace_asia = ('South', 'Cambodia','Thailand','Laos', 'Taiwan', 'China', 'Japan', 'India', 'Iran', 'Philippines', 'Vietnam', 'Hong')
replace_europe = ('England', 'Germany', 'Portugal', 'Italy', 'Poland', 'France', 'Yugoslavia','Scotland', 'Greece', 'Ireland', 'Hungary', 'Holand-Netherlands')
replace_sa = ('Columbia', 'Ecuador', 'Peru')
replace_other = ('?')
df_training.native_country = df_training.native_country.replace(to_replace=replace_northA,value='North America')
df_training.native_country = df_training.native_country.replace(to_replace=replace_carib,value='Caribbean')
df_training.native_country = df_training.native_country.replace(to_replace=replace_asia,value='Asia')
df_training.native_country = df_training.native_country.replace(to_replace=replace_europe,value='Europe') 
df_training.native_country = df_training.native_country.replace(to_replace=replace_sa,value='South America')
df_training.native_country = df_training.native_country.replace(to_replace=replace_other,value='Other')   

df_test.native_country = df_test.native_country.replace(to_replace=replace_northA,value='North America')
df_test.native_country = df_test.native_country.replace(to_replace=replace_carib,value='Caribbean')
df_test.native_country = df_test.native_country.replace(to_replace=replace_asia,value='Asia')
df_test.native_country = df_test.native_country.replace(to_replace=replace_europe,value='Europe') 
df_test.native_country = df_test.native_country.replace(to_replace=replace_sa,value='South America')
df_test.native_country = df_test.native_country.replace(to_replace=replace_other,value='Other') 

#%%

df_training.drop(drop_cols,axis=1,inplace=True)
df_test.drop(drop_cols,axis=1,inplace=True)

#%%
df_training2 = df_training.copy()
df_test2 = df_test.copy()

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

X_train2 = df_training2.drop(columns=["income_bracket",target_col])
y_train2 = df_training2[target_col]
X_test2 = df_test2.drop(columns=["income_bracket",target_col])
y_test2 = df_test2[target_col]

#%%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score , classification_report, log_loss
from sklearn.svm import LinearSVC, SVC

#%%
preprocess = make_column_transformer(
    (cont_cols ,make_pipeline(SimpleImputer(), StandardScaler())),
    (cat_cols, OneHotEncoder()))

#%%
model1 = LogisticRegression(solver='liblinear')
model2 = make_pipeline(
    preprocess,
    LogisticRegression(solver='liblinear'))

svm1 = LinearSVC(C=0.007)
svm1.fit(X_train, y_train)
model1.fit(X_train,y_train)
model2.fit(X_train2,y_train2)

predictions1 = model1.predict(X_test)
predictions2 = model2.predict(X_test2)

svm_predictions = svm1.predict(X_test)
#%%
print(classification_report(y_test,predictions1))
print("Accuracy:",accuracy_score(y_test, predictions1))

print(classification_report(y_test2,predictions2))
print("Accuracy:",accuracy_score(y_test2, predictions2))

print(classification_report(y_test,svm_predictions))
print("Accuracy:",accuracy_score(y_test, svm_predictions))


#%%

logLoss = log_loss(y_test,predictions1)
print(
    "="*80,
    classification_report(y_test,predictions1),
    "Accuracy:    %f" %accuracy_score(y_test, predictions1),
    "Log Loss:    %f" % logLoss,
    "Continuous Columns:\n%a" % cont_cols,
    "Categorical Columns:\n%a" %cat_cols,
    "Drop Columns:\n%a" %drop_cols,
    sep="\n\n",
    end="\n\n"+("="*80))
#%%

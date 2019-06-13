# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs'))
	print(os.getcwd())
except:
	pass

# %%


# Add library references
import pandas as pd
import numpy as np
import seaborn as sns
import timeit
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score , classification_report, log_loss
from sklearn.svm import LinearSVC, SVC

# %%
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
def fetch_df():
    df_training = pd.read_csv("data/adult-training.csv",
        names=df_headers, 
        skipinitialspace = True)

    df_test = pd.read_csv("data/adult-test.csv",
        names = df_headers,
        skipinitialspace = True,
        skiprows=1)

    df = pd.concat([df_training,df_test],axis=0)
    # df.info()
    return df

def process_target(df,target_col=target_col):
    df[target_col] = (df["income_bracket"]
        .apply(lambda x: ">50K" in x)).astype(int)
    return df

def process_edu(df):
    replace_edu_no = ('1st-4th', '5th-6th','7th-8th','9th', '10th', '11th', '12th', 'Preschool')
    replace_edu_associate = ('Assoc-acdm', 'Assoc-voc')
    replace_edu_diploma = ('Some-college', 'HS-grad')

    df.education = df.education.replace(to_replace=replace_edu_no,value='No Diploma')
    df.education = df.education.replace(to_replace=replace_edu_associate,value='Associates')
    df.education = df.education.replace(to_replace=replace_edu_diploma,value='Diploma')
    return df['education'].str.strip()

def process_native(df):
    # Put countries in their native region continent
    replace_northA = ('United-States', 'Honduras', 'Mexico','Puerto-Rico','Canada', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Guatemala', 'El-Salvador')
    replace_carib = ('Cuba', 'Jamaica', 'Trinadad&Tobago', 'Haiti', 'Dominican-Republic')
    replace_asia = ('South', 'Cambodia','Thailand','Laos', 'Taiwan', 'China', 'Japan', 'India', 'Iran', 'Philippines', 'Vietnam', 'Hong')
    replace_europe = ('England', 'Germany', 'Portugal', 'Italy', 'Poland', 'France', 'Yugoslavia','Scotland', 'Greece', 'Ireland', 'Hungary', 'Holand-Netherlands')
    replace_sa = ('Columbia', 'Ecuador', 'Peru')
    replace_other = ('?')
    df.native_country = df.native_country.replace(to_replace=replace_northA,value='North America')
    df.native_country = df.native_country.replace(to_replace=replace_northA,value='North America')
    df.native_country = df.native_country.replace(to_replace=replace_carib,value='Caribbean')
    df.native_country = df.native_country.replace(to_replace=replace_asia,value='Asia')
    df.native_country = df.native_country.replace(to_replace=replace_europe,value='Europe') 
    df.native_country = df.native_country.replace(to_replace=replace_sa,value='South America')
    df.native_country = df.native_country.replace(to_replace=replace_other,value='Other')   
    return df

def process_drops(df, cols):
    return df.drop(cols,axis=1,inplace=True)

def build_preprocessor(cont_cols,cat_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_cols),
            ('cat', categorical_transformer, cat_cols)])
    return ('preprocessor',preprocessor)

def split_df(X,y,split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test

def build_df(drops=None):
    df = fetch_df()
    process_target(df, target_col=target_col)
    process_edu(df)
    process_native(df)
    process_drops(df,drops)
    X = df.drop(columns=["income_bracket",target_col])
    y = df[target_col]
    
    return X,y

def build_transform(cont_cols,cat_cols):
    return build_preprocessor(cont_cols,cat_cols)
#%%
X,y = build_df(drop_cols)
X_all,y = build_df(drops=[])
trans = build_transform(cont_cols,cat_cols)




#%%
# X_clean = X.dropna(axis=1, how='all')
trans = build_transform(cont_cols,cat_cols)
cat_cols2 = [
    "workclass",
    "marital_status", 
    "occupation",
    "race", 
    "gender",
    "relationship",
    "native_country",
    "education"]

cont_cols2 = [
    "age", 
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    'fnlwgt']
trans2 = build_transform(cont_cols2,cat_cols2)
X_processed = trans[1].fit_transform(X)
X_processed2 = trans2[1].fit_transform(X_all)
X_train, X_test, y_train, y_test = split_df(X_processed2,y)
X_processed_train, X_processed_test, y_train, y_test = split_df(X_processed,y,0.2)

#%%
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(
    lr.score(X_test, y_test)))
lr.fit(X_processed_train, y_train)
print("Score with only selected features: {:.3f}".format(
    lr.score(X_processed_test, y_test)))

#%%
svc1 = SVC(cache_size=5000,kernel='linear')
t_start = timeit.default_timer()
svc1.fit(X_processed_train,y_train)
print('linear Completed: %f'%(timeit.default_timer()-t_start))
print("Score with only selected features [svc-linear]: {:.3f}".format(
    svc1.score(X_processed_test, y_test)))

# svc1.kernel='linear'
# t_start = timeit.default_timer()
# svc1.fit(X_processed_train,y_train)
# print('linear Completed: %f'%(timeit.default_timer()-t_start))
# print("Score with only selected features [svc-linear]: {:.3f}".format(
#     svc1.score(X_processed_test, y_test)))

#%%
C_s = np.logspace(-10, 0, 10)

scores = list()
for C in C_s:
    t_start = timeit.default_timer()
    svc1.C = C
    this_fit = svc1.fit(X_processed_train,y_train)
    # this_pred = 
    this_scores = this_fit.score(X_processed_test, y_test)
    scores.append(this_scores)
    print('Rount Completed: %f'%(timeit.default_timer()-t_start))

max_index = np.argmax(scores)
max_C = C_s[max_index].round(8)
max_score = np.max(scores)

# %%

plt.figure()
plt.semilogx(C_s, scores)
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('Mean Prediction Accuracy')
plt.xlabel('Parameter C')
plt.annotate('Optimal C Value: %s'%max_C, xy=(max_C, max_score),  xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
plt.show()

#%%

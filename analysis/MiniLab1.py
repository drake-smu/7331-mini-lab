#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'labs'))
	print(os.getcwd())
except:
	pass

#%% [markdown]
# # Data Mining 7331 - Summer 2019
# ## MiniLab 1 - Logistic Regression and Support Vector Machines

# ### Carson Drake, Che Cobb, David Josephs, Andy Heroy

# ## Table of Contents
# We'll have to put one in tomorrow, i'm just going to start writing 

#%% [markdown] 
# supposed to be a table on contents, but i dont know how to make
# markdown do that yet. 
# ## Section 1:Business Understanding
#     ### Section 1a: Describe the purpose of the data set you selected

#%% [markdown] 
# ## Section 1: Business Understanding
# ### Section 1a: Describe the purpose of the data set you selected.  
# We chose this dataset from the UCI's machine learning repository for its categorical
# predictive attributes.  It contains 1994 Census data pulled from the US Census
# database.  The prediction task we've set forth is to predict if a persons
# salary range is >50k in a 1994, based on the various categorical/numerical
# attributes in the census database. The link to the data source is below:
#
# https://archive.ics.uci.edu/ml/datasets/census+income
#
# ### Section 1b: Describe how you would define and measure the outcomes from the dataset. 
# (That is, why is this data important and how do you know if you have mined
# useful knowledge from the dataset? How would you measure the effectiveness of
# a good prediction algorithm? Be specific.)
#
# The main benefit of this data is to be able to predict a persons salary range
# based on factors collected around each worker in 1994.  With that insight, we
# can look at a persons, age, education, marital status, occupation and begin to
# explore the relationships that most influence income.  We'd like to find:
#   * What factors are the strongest influence of a how much many they will
#     make. 
#   * What age groups show the largest amount of incomes over >50k?  aka, what
#     years of our life should we be working hardest in order to make the most
#     money. 
#   * Does where you come from influence your income? (native country)

#%% [markdown] 
# ## Section 2: Data Understanding
# ### Section 2a: Describe the meaning and type of data for each attribute
# Here we will dicuss each attribute and give some description about its ranges. 
#
#
# Category - Description
# #### Categorical Attributes 
# * workclass - Which business sector do they work in?
# * education - What level of education received?
# * marital_status - What is their marriage history
# * occupation - What do they do for a living
# * relationship - Family member relation
# * race - What is the subjects race
# * gender - What is the subjects gender
# * native_country - Where is the subject originally from
# * income_bracket - Do they make over or under 50k/year

# #### Continuous Attributes
# * age - How old is the subject?
# * fnlwgt - Sampling weight of observation  
# * education_num - numerical encoding of education variable
# * capital_gain - income from investment sources, seperate from wages/salary
# * capital_loss - losses from investment sources, seperate from wages/salary
# * hours_per_week - How many hours a week did they work? 
#
#
# ### Section 2b: Data Quality
# Verify data quality: Explain any missing values, duplicate data, and outliers.
# Are those mistakes? How do we deal with these problems?
#
# In the next code section we will import our libraries and data, then begin looking at
# missing data, duplicate data, and outliers. 

#%%
# Add library references
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
# Input in case we want to combine the dataframes. 
# df_test = pd.read_csv("data/adult-test.csv",names = df_headers,skiprows=1)
# df_census = pd.concat([df_test, df_census], axis=0)

df_census.head(10)

#%% [markdown] First, we'll start with looking at the head of the table to get a
# feel for overall structure and the variables that we're working with. Followed
# by a count of any missing values within the dataset.  We see that our data has
# no missing values which is great under most circumstances, but we also found
# that instead of marking the data with an NA, they did so with a "?.  Our first
# order of business is to replace those values.  We found counts of ? values in
# WorkClass, Occupation, and native country.  For now we'll replace them with
# "Other"



#%%
print("Structure of data:\n",df_census.shape,"\n")
print("Count of missing values:\n",df_census.isnull().sum().sort_values(ascending=False),"\n")
print("Count of ? values in workclass: " ,df_census.loc[df_census.workclass == ' ?', 'workclass'].count())
print("Count of ? values in occupation: ", df_census.loc[df_census.occupation == ' ?', 'occupation'].count())
print("Count of ? values in native_country: ", df_census.loc[df_census.native_country == ' ?', 'native_country'].count())

#%% [markdown] 
# While our missing values count is very low, we now must change
# all the ? entries to other in order not cause further errors.  We'll also be
# grouping each individual native country into their respective continent.  We
# feel that grouping as such will give us more insight into how U.S. immigrants
# fare in the job market.  We'll also introduce a pair plot to look in the
# visualization section to look for any outliers.  Which spoiler alert, it
# doesn't look like we have any that cause great concern. 

#%% Data Cleaning 
# Change income bracket values that have a . at end and remove space 
df_census = df_census.replace(to_replace=(' >50K.', ' >50K'),value='>50K')
df_census = df_census.replace(to_replace=(' <=50K.', ' <=50K'),value='<=50K')    
df_census = df_census.replace(to_replace=(' United-States', ' Honduras', ' Mexico',' Puerto-Rico',' Canada', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Guatemala', ' El-Salvador' ),value='North America')
df_census = df_census.replace(to_replace=(' Cuba', ' Jamaica', ' Trinadad&Tobago', ' Haiti', ' Dominican-Republic' ),value='Caribbean')
df_census = df_census.replace(to_replace=(' South', ' Cambodia',' Thailand',' Laos', ' Taiwan', ' China', ' Japan', ' India', ' Iran', ' Philippines', ' Vietnam', ' Hong'),value='Asia')
df_census = df_census.replace(to_replace=(' England', ' Germany', ' Portugal', ' Italy', ' Poland', ' France', ' Yugoslavia',' Scotland', ' Greece', ' Ireland', ' Hungary', ' Holand-Netherlands'),value='Europe') 
df_census = df_census.replace(to_replace=(' Columbia', ' Ecuador', ' Peru'),value='South America')
df_census = df_census.replace(to_replace=(' ?'),value='Other') 

# encoding into 1 and zero variables for income_bracket. 
df_census['income_bracket'] = df_census['income_bracket'].apply(lambda x: 1 if x=='>50K' else 0)

#%% [markdown]
# ### Section 2c: Simple Statistics
#
# #### Visualize appropriate statistics (e.g., range, mode, mean, median, variance, counts) for a subset of attributes. Describe anything meaningful you found from this or if you found something potentially interesting. 
#
# Now that our data has been cleansed of any obvious errors, its time to look at
# the statistics behind our continuous data in order to look for any other
# errors in the data we might have missed.  We also can get a look at how many
# variables each of our categorical attributes carry with them.  This will be
# useful down the line when we start grouping items for our basic EDA charts we
# would like to produce. 

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

#%% [markdown] As we can see from our stats, we've got normal ranges on each of
# the categories that we've analyzed.  One category of capital_gain has some
# very large numbers, but we might attribute that to massive investments made by
# one individual.  After exploring further, alot of the values are 99,999.
# Which we assume to be a cap on whats reported for capital gains.  We did find
# that most of the occupations showing such captial growth was mostly
# executives.  So we're not suprised to see the higher numbers here and won't
# change the data accordingly. 


#%% 
#Histogram charts
sns.set_style('whitegrid')
df_num = df_census.select_dtypes(include=['float64'])
df_census.hist(figsize =(14,12))
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
## boxplots of income by gender dist.
sns.set_style('whitegrid')
sns.countplot(x='income_bracket',
    hue='gender',
    data=df_census,
    palette='RdBu_r')
#%%
## by marital status
sns.set_style('whitegrid')
sns.countplot(x='income_bracket',
    hue='marital_status',
    data=df_census,
    palette='RdBu_r')


#%%
#Generate Correlation HeatMap
colormap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(10, 10))
corr = df_census.corr()

sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#%%
# Pairplot matrix.  
#%%
g = sns.PairGrid(df_census,vars=['age','fnlwgt',
                               'capital_gain','capital_loss', 
                               'hours_per_week'],
                               hue='income_bracket',palette = 'muted')
g.map(plt.scatter, alpha=0.8)
g.add_legend();

#%%
df_age = df_census.loc[:,['gender', 'age', 'income_bracket']]
conditions = [
    (df_age['age'] < 20),
    (df_age['age'] < 30),
    (df_age['age'] < 40),
    (df_age['age'] < 50),
    (df_age['age'] < 60),
    (df_age['age'] < 70),
    (df_age['age'] < 110)]
choices = ['10-20', '20-30', '30-40','40-50','50-60','60-70','70-110']
df_age['age_group'] = np.select(conditions, choices, default='70-110')

sns.set_style('whitegrid')
sns.countplot(x='age_group',
    hue='income_bracket',
    data=df_age,
    palette='RdBu_r',
    order=choices)


#%% Another style of the pairplot above with a few more details
g = sns.pairplot(df_census,kind="scatter",vars=['age','fnlwgt',
                               'capital_gain','capital_loss', 
                               'hours_per_week'],
                               hue='income_bracket',
                               plot_kws=dict(s=80, edgecolor="white", linewidth=2.5),
                               palette = 'muted')
g.add_legend();


#%% Crazy violin plot
# Plot
sns.catplot(x="age", y="native_country",
            hue="gender", col="income_bracket",
            data=df_census,
            orient="h", height=5, aspect=1, palette="tab10",
            kind="violin", dodge=True, cut=0, bw=.2)

#%%
# Assigning age group to the dataframe. 
df_census['age_group'] = np.select(conditions, choices, default='70-110')

#%%
## Box Plot of region by income bracket.
plt.figure(figsize=(10,8), dpi= 80)
sns.boxplot(x='age_group', y='hours_per_week', 
            data=df_census, hue='income_bracket',
            order=choices,palette="tab10")
# sns.stripplot(x='age_group', y='hours_per_week', data=df_census, color='black', size=3, jitter=1)

# for i in range(len(df_census['age_group'].unique())-1):
#     plt.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)

# Decoration
plt.title('Age Group Hours per week by income_bracket', fontsize=22)
plt.legend(title='Income_Bracket')
plt.show()

#%%

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

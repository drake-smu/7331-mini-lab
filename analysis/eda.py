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

## Table of Contents
# We'll have to put one in tomorrow, i'm just going to start writing 

#%% [markdown] 
# supposed to be a table on contents, but i dont know how to make
# markdown do that yet. 
# ## Section 1:Business Understanding
#     ### Section 1a: Describe the purpose of the data set you selected

#%% [markdown] 
# # Business Understanding
# ## Section 1a: Describe the purpose of the data set you selected.  
# We chose this dataset from the UCI's machine learning repository for its
# predictive attributes.  It contains 1994 Census data pulled from the US Census
# database.  The prediction task we've set forth is to predict if a persons
# salary range is >50k in a 1994, based on the various categorical/numerical
# attributes in the census database. The link to the data source is below:
#
# https://archive.ics.uci.edu/ml/datasets/census+income
#
# ## Section 1b: Describe how you would define and measure the outcomes from the dataset. 
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
#   *







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

#%% Data Cleaning (Mostlyche's)
# Change income bracket values that have a . at end and remove space 
df_census = df_census.replace(to_replace=(' >50K.', ' >50K'),value='>50K')
df_census = df_census.replace(to_replace=(' <=50K.', ' <=50K'),value='<=50K')    
df_census = df_census.replace(to_replace=(' United-States', ' Honduras', ' Mexico',' Puerto-Rico',' Canada', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Guatemala', ' El-Salvador' ),value='North America')
df_census = df_census.replace(to_replace=(' Cuba', ' Jamaica', ' Trinadad&Tobago', ' Haiti', ' Dominican-Republic' ),value='Caribbean')
df_census = df_census.replace(to_replace=(' South', ' Cambodia',' Thailand',' Laos', ' Taiwan', ' China', ' Japan', ' India', ' Iran', ' Philippines', ' Vietnam', ' Hong'),value='Asia')
df_census = df_census.replace(to_replace=(' England', ' Germany', ' Portugal', ' Italy', ' Poland', ' France', ' Yugoslavia',' Scotland', ' Greece', ' Ireland', ' Hungary', ' Holand-Netherlands'),value='Europe') 
df_census = df_census.replace(to_replace=(' Columbia', ' Ecuador', ' Peru'),value='South America')
df_census = df_census.replace(to_replace=(' ?'),value='Other')   
 

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

sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".1f",
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

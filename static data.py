#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing basic package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


#Read the file for indivudial assignment 

df_mentah = pd.read_csv("/Users/abdullahhasib/Documents/application file/NUS/courses/semester 2/SPH6004 Advanced Statistical Learning/group assignment/Assignment2_mimic dataset/MIMIC-IV-static(Group Assignment).csv")

df_mentah


# In[5]:


###Acknowledgement to https://github.com/galuhsahid/data-preparation-with-python/blob/master/02_DataPreparationWithPython.ipynb
#dropping irrelevant categorical data

df_mentah.dtypes.head(50)


# In[6]:


df_mentah.dtypes.head(100).index


# In[7]:


df_drop_cat = df_mentah.drop(['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit',
       'intime', 'outtime', 'deathtime','insurance',
       'language', 'marital_status', 'hospital_expire_flag',], axis=1)

df_drop_cat


# In[8]:


df_drop_cat.dtypes.head(100).index


# In[9]:


###Acknowledgement to pandas cheat sheet https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
#identification of column with missing values

df_drop_cat.isnull().sum().sort_values(ascending=False).head(50)


# In[10]:


#dropping columns with high missing values

df_drop_cat.isnull().sum().sort_values(ascending=False).head(50).index


# In[11]:


df_drop_head50 = df_drop_cat.isnull().sum().sort_values(ascending=False).head(50).index

df_drop_1 = df_drop_cat.drop(columns = df_drop_head50)

df_drop_1


# In[14]:


#second round for dropping column with missing values

# as the row is 76,943 rows, that means the treshold 50% will be >= 38471.5 so that any misisng values above this number will be removed

df_drop_1.isnull().sum().sort_values(ascending=False).head(50)


# In[16]:


df_drop_1.isnull().sum().sort_values(ascending=False).head(50).index


# In[17]:


df_drop_2 = df_drop_1.drop(['sofa2_respiration_24h_max', 'sofa2_liver_24h_max', 'alp_max',
       'alp_min', 'bilirubin_total_max', 'bilirubin_total_min', 'alt_min',
       'alt_max'], axis=1)

df_drop_2


# In[18]:


###Acknowledgement to https://github.com/galuhsahid/data-preparation-with-python/blob/master/01_DataPreparationWithPython.ipynb
#changing race into different categorical (one hot)

#identifying values in the column race
race = df_drop_2['race'].unique()

race


# In[19]:


#identifying how many values in the column race

race = df_drop_2['race'].nunique()

race

##Not to be run!! ## just for checking purposes!!
1) ASIAN = 
1.'ASIAN',
2.'ASIAN - CHINESE',
3.'ASIAN - KOREAN', 
4.'ASIAN - ASIAN INDIAN',
5.'ASIAN - SOUTH EAST ASIAN'

2) CAUCASIAN = 
6.'WHITE', 
7.'WHITE - OTHER EUROPEAN',
8.'WHITE - BRAZILIAN', 
9.'WHITE - EASTERN EUROPEAN',
10.'WHITE - RUSSIAN',
11.'HISPANIC/LATINO - MEXICAN', 
12.'HISPANIC OR LATINO',
13.'HISPANIC/LATINO - HONDURAN',
14.'HISPANIC/LATINO - COLUMBIAN',
15.HISPANIC/LATINO - CUBAN',
16.'HISPANIC/LATINO - DOMINICAN',
17.'HISPANIC/LATINO - PUERTO RICAN',
18.'HISPANIC/LATINO - SALVADORAN',
19.'HISPANIC/LATINO - CENTRAL AMERICAN',
20.'HISPANIC/LATINO - GUATEMALAN', 
21.'PORTUGUESE'

3) NEGROID =
22.'BLACK/AFRICAN AMERICAN', 
23.'BLACK/CAPE VERDEAN',
24.'BLACK/CARIBBEAN ISLAND',
25.'BLACK/AFRICAN'

4) OTHER =
26.'UNABLE TO OBTAIN', 
27.'MULTIPLE RACE/ETHNICITY',
28.'PATIENT DECLINED TO ANSWER',
29.'AMERICAN INDIAN/ALASKA NATIVE', 
30.'OTHER', 
31.'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
32.'SOUTH AMERICAN', 
33.'UNKNOWN'
# In[20]:


###https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html#pandas.DataFrame.replace
#Changing the race category

df_race = df_drop_2.replace(['ASIAN', 
                             'ASIAN - CHINESE',
                             'ASIAN - KOREAN', 
                             'ASIAN - ASIAN INDIAN',
                             'ASIAN - SOUTH EAST ASIAN'], 
                            'ASIAN').replace(['WHITE',
                            'WHITE - OTHER EUROPEAN',
                            'WHITE - BRAZILIAN',
                            'WHITE - EASTERN EUROPEAN',
                            'WHITE - RUSSIAN',
                            'HISPANIC/LATINO - MEXICAN', 
                            'HISPANIC OR LATINO',
                            'HISPANIC/LATINO - HONDURAN', 
                            'HISPANIC/LATINO - COLUMBIAN',
                            'HISPANIC/LATINO - CUBAN', 
                            'HISPANIC/LATINO - DOMINICAN',
                            'HISPANIC/LATINO - PUERTO RICAN', 
                            'HISPANIC/LATINO - SALVADORAN',
                            'HISPANIC/LATINO - CENTRAL AMERICAN', 
                            'HISPANIC/LATINO - GUATEMALAN',
                            'PORTUGUESE'], 
                            'CAUCASIAN').replace(['BLACK/AFRICAN AMERICAN', 
                            'BLACK/CAPE VERDEAN',
                            'BLACK/CARIBBEAN ISLAND',
                            'BLACK/AFRICAN'], 
                            'NEGROID').replace(['UNABLE TO OBTAIN', 
                            'MULTIPLE RACE/ETHNICITY', 
                            'PATIENT DECLINED TO ANSWER',
                            'AMERICAN INDIAN/ALASKA NATIVE', 
                            'OTHER', 
                            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                            'SOUTH AMERICAN', 
                            'UNKNOWN'], 
                            'OTHER')

df_race  


# In[21]:


#checking the value for race
df_race['race'].unique()


# In[22]:


###Acknowledgement to https://github.com/galuhsahid/data-preparation-with-python/blob/master/01_DataPreparationWithPython.ipynb

df_race.head().transpose()


# In[23]:


df_race.isnull().sum().sort_values(ascending=True)


# In[25]:


###Acknowledgement to https://www.youtube.com/watch?v=WWbyYFPHDH8&t=599s
#One Hot Encoding

onehot = df_race['race'].values.reshape(-1,1)
onehot


# In[26]:


from sklearn. preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()

onehot = onehot_encoder.fit_transform(onehot).toarray()
onehot


# In[27]:


onehot_encoder.categories_


# In[28]:


df_onehot = pd.DataFrame(onehot, columns=[str(i) for i in range(onehot.shape[1])])
df_onehot


# In[29]:


df_onehot1 = pd.concat([df_onehot, df_race], axis=1)
df_onehot1


# In[30]:


#Dropping race column

df_onehot2 = df_onehot1.drop(['race'], axis=1)

df_onehot2


# In[31]:


#Changing the column name
df = df_onehot2.rename(columns = {'0':'ASIAN', '1':'CAUCASIAN', '2':'NEGROID', '3':'OTHER'})

df


# In[32]:


###Acknowledgement to https://www.youtube.com/watch?v=WWbyYFPHDH8&t=539s
#Labelling gender

from sklearn.preprocessing import LabelEncoder

le_df = LabelEncoder()
df['gender'] = le_df.fit_transform(df['gender'])

df

The result shows gender M=1, F=0
# In[33]:


### Acknowledgement to https://www.youtube.com/shorts/-OiRzEyvah0

#changing 1 to 0 and vice versa in icu_death_flag to avoid my confusiion. 1 is the discharge one and 0 is the death 

df['icu_death_flag'] = df['icu_death_flag'].replace({1:0, 0:1})
df


# In[34]:


###Acknowledgement to https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html#pandas.DataFrame.hist
#Identification distribution of the data to see the skewness
df.hist(figsize=(15, 15))


# In[35]:


###Acknowledgement to pandas cheat sheet https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
#identification of column with missing values

df.isnull().sum().sort_values(ascending=False)


# In[36]:


#Filling the missing value with median

df = df.fillna(df.median())


# In[33]:


#Sorting the value with missing value to check them

df.isnull().sum().sort_values(ascending=False)


# In[37]:


#Deciding which one is the target and which one is the features

df.info()


# In[55]:


y = df.iloc[:, 6]
x = df.drop(columns=['icu_death_flag'])


# In[56]:


#checking the features 

x.describe().transpose()


# In[57]:


#checking the target

y.describe()


# In[58]:


# splitting the data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=12)


# In[59]:


#seeing my column

x_train.columns


# In[60]:


continuous_columns =['icu_los_hours', 'los', 'age', 'sofa2_cardiovascular_24h_max', 'sofa2_coagulation_24h_max',
       'sofa2_renal_24h_max', 'sofa2_cns_24h_max', 'heart_rate_min',
       'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean',
       'dbp_min', 'dbp_max', 'dbp_mean', 'mbp_min', 'mbp_max', 'mbp_mean',
       'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'temperature_min',
       'temperature_max', 'temperature_mean', 'spo2_min', 'spo2_max',
       'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean', 'gcs_min',
       'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable', 'hematocrit_min',
       'hematocrit_max', 'hemoglobin_min', 'hemoglobin_max', 'platelets_min',
       'platelets_max', 'wbc_min', 'wbc_max', 'aniongap_min', 'aniongap_max',
       'bicarbonate_min', 'bicarbonate_max', 'bun_min', 'bun_max',
       'calcium_min', 'calcium_max', 'chloride_min', 'chloride_max',
       'creatinine_min', 'creatinine_max', 'sodium_min', 'sodium_max',
       'potassium_min', 'potassium_max', 'thrombin_min', 'thrombin_max',
       'pt_min', 'pt_max', 'ptt_min', 'ptt_max', 'po2_min', 'po2_max',
       'radiology_note_count']

continuous_x_train = x_train[continuous_columns]

continuous_x_train


# In[61]:


categorical_column = ['ASIAN', 'CAUCASIAN', 'NEGROID', 'OTHER', 'gender']

categorical_x_train = x_train[categorical_column]

categorical_x_train


# In[62]:


#stand. the continuous data for the x train first

std_x_train_con = (continuous_x_train-continuous_x_train.mean())/continuous_x_train.std()
std_x_train_con.describe().transpose()


# In[63]:


#collapsing the cat and con into one for x_train

std_x_train = pd.concat([categorical_x_train, std_x_train_con], axis=1)

std_x_train.info()


# In[64]:


continuous_x_test = x_test[continuous_columns]

continuous_x_test


# In[65]:


categorical_x_test = x_test[categorical_column]

categorical_x_test


# In[66]:


###Acknowledgement to hands-on file sph6004
#stand. the continuous data for the x test now

std_x_test_con = (continuous_x_test-continuous_x_test.mean())/continuous_x_test.std()
std_x_test_con.describe().transpose()


# In[67]:


#collapsing the cat and con into one for x_test

std_x_test = pd.concat([categorical_x_test, std_x_test_con], axis=1)

std_x_test


# In[68]:


#checking the balance of the data

y_train.value_counts()


# In[69]:


###Acknowledgement to hands-on file sph6004

#SMOTE
from imblearn.over_sampling import SMOTE

smote_sampler = SMOTE(random_state=12,sampling_strategy='minority')
X_df_SMOTE, y_df_SMOTE = smote_sampler.fit_resample(std_x_train, y_train)


# In[70]:


y_df_SMOTE.value_counts()


# In[71]:


#Multicolinearity

###https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

X=X_df_SMOTE.copy() #untuk membuat duplikat supaya tidak terhubung dengan X_df_SMOTE_fw yg asal jika variabel asal dirubah

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()

print(X.columns)


# In[78]:


###Acknowledgment to https://github.com/tempse/sklearn-beginners-template/blob/8f4509631b8ef29bb10dc4a827c2043fa73aadd7/example%20analysis%20-%20random%20forest.ipynb

#Training the Model (RFC) with full features

from sklearn.ensemble import RandomForestClassifier

model_RFC_full = RandomForestClassifier(n_estimators=500,
                             criterion='gini',
                             max_depth=None,
                             min_samples_split=1000,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features='sqrt',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=-1,
                             random_state=None,
                             verbose=0,
                             warm_start=False,
                             class_weight=None)
model_RFC_full.fit(X_df_SMOTE, y_df_SMOTE)


# In[79]:


#Checking the features

importances = model_RFC_full.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_RFC_full.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

featureNames = X_df_SMOTE.columns

for f in range(X_df_SMOTE.shape[1]):
    print("\t%d. %s \t(%f)" % (f + 1,
                               featureNames[indices[f]],
                               importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_df_SMOTE.shape[1]),
        importances[indices],
        color="r",
        yerr=std[indices],
        align="center")
plt.xticks(range(X_df_SMOTE.shape[1]), featureNames[indices], rotation=90)
plt.xlim([-1, X_df_SMOTE.shape[1]])
plt.tight_layout()


# In[91]:


###Acknowledgement to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(max_depth=2, random_state=12)
selector = RFECV(estimator, step=1, cv=5, min_features_to_select=1)
selector = selector.fit(X_df_SMOTE, y_df_SMOTE)
selector.support_
selector.ranking_


# In[92]:


for i, col in zip(range(X_df_SMOTE.shape[1]), X_df_SMOTE.columns):
    print(f"{col} selected={selector.support_[i]} rank={selector.ranking_[i]}")


# In[93]:


import pandas as pd

# 1. Masukkan hasil ke dalam DataFrame
# Pastikan std_x_train.columns adalah kolom yang sama dengan X_df_SMOTE
ranking_df = pd.DataFrame({
    'Feature_Name': X_df_SMOTE.columns,
    'Selected': selector.support_,
    'Ranking': selector.ranking_
})

# 2. Urutkan dari Ranking 1 hingga yang terakhir (74)
# Ranking 1 adalah fitur-fitur terbaik yang dipilih model
ranking_df_sorted = ranking_df.sort_values(by='Ranking').reset_index(drop=True)

# 3. Tampilkan semua baris agar Anda bisa melihat urutan 1-74
pd.set_option('display.max_rows', None)
print(ranking_df_sorted)


# In[94]:


#training the model
from sklearn.linear_model import LogisticRegression
model_log= LogisticRegression(random_state=12, penalty='l1', C=0.01, solver='saga', max_iter=1000) 
model_log.fit(X_df_SMOTE, y_df_SMOTE)


# In[95]:


#Checking the features

import numpy as np
import matplotlib.pyplot as plt

# 1. Ambil koefisien (gunakan nilai absolut karena koefisien bisa negatif)
# .coef_[0] digunakan karena Logistic Regression mengembalikan array 2D
importances = np.abs(model_log.coef_[0])

# 2. Urutkan indeks dari yang terbesar ke terkecil
indices = np.argsort(importances)[::-1]

print("Feature ranking (Logistic Regression Coefficients):")

featureNames = X_df_SMOTE.columns

for f in range(X_df_SMOTE.shape[1]):
    print("\t%d. %s \t(%f)" % (f + 1, 
                               featureNames[indices[f]], 
                               importances[indices[f]]))

# 3. Visualisasi (Tanpa yerr/std karena model linear tidak memiliki variabilitas inter-tree)
plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Logistic Regression Coef)")
plt.bar(range(X_df_SMOTE.shape[1]), 
        importances[indices], 
        color="b", 
        align="center")
plt.xticks(range(X_df_SMOTE.shape[1]), featureNames[indices], rotation=90)
plt.xlim([-1, X_df_SMOTE.shape[1]])
plt.ylabel("Absolute Coefficient Value")
plt.tight_layout()
plt.show()


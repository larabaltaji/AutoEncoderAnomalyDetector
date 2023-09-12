#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

#Visualization libraries
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

#Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest

#Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, auc
from sklearn.metrics import recall_score, make_scorer
from pyod.utils.data import evaluate_print
from tensorflow.keras import layers, losses
from sklearn.metrics import mean_squared_error

# Feature Selection and Dimensionality Reduction
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
import kerastuner as kt
from kerastuner import Objective, RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

#saving the model for streamlit
import pickle

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# # Loading the Dataset

# In[4]:


df = pd.read_excel(r'C:\Users\l.baltaji\Desktop\Capstone Project\Methodology and Source Code\Tickets Table Plus Synthesized Anomalies.xlsx')


# In[5]:


df.head()


# ___
# # 1 Data Exploration

# In[6]:


df.info()


# In[7]:


df.describe().T


# ## Checking for missing values

# In[8]:


print('Number of missing values for each field:')
df.isnull().sum()


# No null values exist in the dataset

# ## Checking for duplicates

# In[9]:


# find duplicates
dups = df.duplicated()
print(f"There are {dups.sum()} duplicate rows.")


# In[10]:


# Let's remove the duplicated rows as they will add redundancy and noise to out data
df = df.drop_duplicates(keep = 'first')


# In[11]:


df.shape


# ## Plotting the target variable

# In[12]:


anomaly_counts = df['Anomaly'].value_counts()

custom_palette = ["lightblue" if count < max(anomaly_counts) else "darkblue" for count in anomaly_counts]

sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values,  palette=custom_palette)
plt.xlabel('Anomaly Label')
plt.ylabel('Frequency')
plt.title('Bar Plot of the Anomaly Label')
plt.show()


# In[13]:


anomaly_type_counts = df['Anomaly Type'].value_counts()
anomaly_type_counts = anomaly_type_counts[anomaly_type_counts.index!=0].sort_values(ascending = False)

custom_palette = ["lightblue" if count < max(anomaly_type_counts) else "darkblue" for count in anomaly_type_counts]
# Create a bar plot using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=anomaly_type_counts.index, y=anomaly_type_counts.values,  palette=custom_palette)
plt.xlabel('Anomaly Type')
plt.ylabel('Frequency')
plt.title('Bar Plot of the Anomaly Type (Excluding Type 0)')
sns.set_palette("coolwarm")
plt.show()


# In[14]:


anomaly_type_counts


# In[15]:


custom_palette = ["lightblue" if count == max(anomaly_counts) else "darkblue" for count in anomaly_counts]
sns.set_palette(custom_palette)
plt.pie(anomaly_counts, labels= ["Normal", "Anomaly"], autopct='%.00f%%')
plt.title('Pie Chart of the Anomaly Label')
plt.show()


# Our target variable is imbalanced with only 1% of the transactions being anomalous

# ### Checking the distribution of numerical variables

# In[16]:


sns.set_palette('coolwarm')


# In[17]:


numerical = ['Quantity', 'UnitPrice', 'AVGCost', 'ExchangeFees', 'BrokerageFees', 'TicketCharges','SettlementFees', 'OtherFees',
            'ValueAmount', 'LocalValueRate', 'LocalEquivalent']             
categorical = ['TypeId', 'SubTypeId', 'MethodId', 'HoldingId', 'ClientId', 'ProductId','BrokerId', 'TradingAccountId', 
               'ShareAccountId', 'CashBankAccountId','CashThirdPartyAccountId', 'ValueCurrencyId', 'SettlementCurrencyId']
datetime = ['CashSettlementDate', 'ShareSettlementDate', 'CreationDate']


# In[18]:


df[numerical].hist(bins=50, figsize=(20,15))
plt.show()


# None of the variables is normally distributed

# ## Analyzing Relationships

# In[19]:


corr = df.corr()


# In[20]:


corr["Anomaly"].sort_values(ascending=False)


# In[21]:


sorted_anomaly = corr["Anomaly"].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Numerical Category")
plt.title("Anomaly Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[22]:


corr['Anomaly Type'].sort_values(ascending = False)


# In[23]:


sorted_anomaly_type = corr["Anomaly Type"].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(sorted_anomaly_type.index, sorted_anomaly_type.values)
plt.xlabel("Correlation Value")
plt.ylabel("Numerical Category")
plt.title("Anomaly Type Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[24]:


# Create the correlation matrix
corr = df.corr()

# Create a heatmap plot of the correlation matrix
mask = np.triu(np.ones_like(corr,dtype = bool))
plt.figure(figsize=(7, 6))
ax = plt.axes()
sns.heatmap(corr,annot=False, mask=mask,lw=0,linecolor='white',fmt = "0.2f", cmap="RdBu_r")
plt.title('Correlation Analysis')
plt.axis('tight')
plt.show()


# We can notice that there exists a high correlation between:
# 
# - ClientId and MethodId 
# - ValueCurrencyId and BrokerId
# - ShareAccountId and TradingAccountId
# - CashBankAccountId and TradingAccountId
# - CashThirdPartyAccountId and TradingAccountId
# - CashThirdPartyAccountId and ShareAccountId
# - CashThirdPartyAccountId and CashBankAccountId
# - AVGCost and UnitPrice
# - LocalValueRate and ValueCurrencyId
# - LocalValueRate and SettlementCurrencyId
# - LocalEquivalent and ValueAmount
# - Anomaly and Anomaly Type
# 

# ___
# # 2 Baseline System
# 
# In this section, I will build a simple baseline system using Logistic Regression model and using only a subset of the features only (ignoring date and time features).
# 
# - Scaling numerical features using StandardScaler()
# - Deploying the Logistic Regression model

# In[25]:


df_subset = df.drop(datetime, axis = 1)


# In[26]:


# Splitting the dataset into training and validation datasets
train, valid = train_test_split(df_subset, test_size = 0.3, random_state =  1, shuffle = True)


# ## Preprocessing Numerical Features

# In[27]:


# Standardize the numerical data
scaler = StandardScaler()


# In[28]:


# fit the scaler on the numerical columns
scaler.fit(train[numerical])


# In[29]:


# Transform the numerical columns of the train and valid datasets
train[numerical] = scaler.transform(train[numerical])
valid[numerical] = scaler.transform(valid[numerical])


# In[30]:


train[numerical]


# In[31]:


# Extract features and target variable
X_train = train.drop(columns=['Anomaly', 'Anomaly Type'])  # Features (excluding the target variable and anomaly type)
y_train = train['Anomaly']  # Target variable
X_valid = valid.drop(columns=['Anomaly', 'Anomaly Type'])  
y_valid = valid['Anomaly']  


# ## Deploying the Logistic Regression model

# In[32]:


# Define and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[33]:


# predict probabilities on train and valid sets
y_train_scores = model.predict_proba(X_train)[:, 1]
y_valid_scores = model.predict_proba(X_valid)[:,1]

# calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# plot ROC curves
plt.plot(train_fpr, train_tpr,'-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr,'-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc = 'lower right')
plt.legend()
plt.show()


# In[34]:


# Not a valid metric since the data is imbalanced
print(f'Training Accuracy: {model.score(X_train, y_train)}')
print(f'Validation Accuracy: {model.score(X_valid, y_valid)}')


# In[35]:


# Confusion Matrix
y_pred_values = model.predict(X_valid)
cf_matrix = confusion_matrix(y_valid, y_pred_values)
sns.heatmap(cf_matrix, annot = True)
plt.title('Confusion Matrix for Baseline')


# In[36]:


print(classification_report(y_valid, y_pred_values))


# ##### Results:
# - Bad performance.
# - AUC on the training = 0.789 and AUC on the validation = 0.735.
# - The validation accuracy metric shows that the model is 0.987 accurate. However the data is imbalanced, which means that the accuracy cannot be a valid evaluation metric.
# - The confusion matrix and the classification report reveal that it wasn't able to capture any of the anomalies, classifying 48 normal values as anomalous and 4 anomalous entries as normal.
# - The classification matrix also shows that the macro-averaged precision is 0.49, but and the macro-averaged recall is 0.5.
# - The main reason for this bad performance is that is model uses only a subset of the features (diregarding the datetime features). In addition, no significant features has been extracted.
# 
# Let us improve the performance:
# ___

# # 3 Improved System
# 
# In this section, I will iteratively refine my preprocessing techniques and features (e.g., by including all features, engineering new ones, and then selecting the most powerful ones) to improve the performance of my model on the validation set.
# 1. Splitting Optimization (Chossing the most suitable splitting technique between random split and stratified split)
# 2. Feature Engineering (Feature Extraction and Feature Selection)
# 3. Data Preprocessing (Scaling numerical features and encoding categorical features)
# 4. Deploying the Logistic Regression model on all the features
# 5. Selecting the most important features based on their correlation with the anomaly types
# 6. Deploying the Logistic Regression on the selected features and comparing.

# ## Splitting Optimization

# In[37]:


df.head()


# In[38]:


train_rand, valid_rand = train_test_split(df, test_size=0.2, random_state=11, shuffle=True)
train_strat, valid_strat = train_test_split(df, test_size=0.2, random_state=11, shuffle=True, stratify=df.Anomaly)


# In[39]:


# Calculate the proportions for each category in each split
all_proportions = df['Anomaly'].value_counts(normalize = True)
train_rand_proportions = train_rand['Anomaly'].value_counts(normalize=True)
valid_rand_proportions = valid_rand['Anomaly'].value_counts(normalize=True)
train_strat_proportions = train_strat['Anomaly'].value_counts(normalize=True)
valid_strat_proportions = valid_strat['Anomaly'].value_counts(normalize=True)


# In[40]:


sns.set_palette('coolwarm')
# Create a DataFrame to hold the proportions
proportions_df = pd.DataFrame({
    'All Data':all_proportions,
    'Random Split (Train)': train_rand_proportions,
    'Random Split (Validation)': valid_rand_proportions,
    'Stratified Split (Train)': train_strat_proportions,
    'Stratified Split (Validation)': valid_strat_proportions
})

# Create a side-by-side bar plot
proportions_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Proportions in Splits')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.show()


# There is not much difference between random split and stratified split by the target variable. We will use the stratified split.

# In[41]:


train, valid = train_test_split(df, test_size=0.2, random_state=11, shuffle=True, stratify=df.Anomaly)


# ## Feature Extraction
# 
# Let's derive some useful features based on each anomaly type:
# 
# **Anomaly 1**: Transaction occurring outside market working days (all weekdays except Friday) and working hours (8AM - 6PM)
# - Working days and working hours 
# - Binary field showing being 1 if transactions are occuring outside of working days and working hours and 0 otherwise
# 
# **Anomaly 2**: Cash settlement date occurring before share settlement date
# - Difference between the share settlement date and the cash settlement date (if negative then anomolous)
# - Binary field being 1 if the difference is negative, and zero otherwise.
# 
# **Anomaly 3**: Unit price of a product deviating significantly from its average cost
# - Ratio between unit price and the average price (if the ratio is close to 1, then normal. If ratio is big, then anomalous)
# 
# **Anomaly 4**: Ticket charge deviating significantly from the usual ticket charge pattern
# - *No need to extract a new feature. Normalization will do the job.*
# 
# **Anomaly 5**: Local settlement rate deviating significantly from the usual local exchange rate pattern for the same currency
# - Binary field being 1 if the local settlement rate is an outlier relative to the usual local settlement rate for the same currency and 0 otherwise.
# 
# **Anomaly 6**: Product quantity purchase/sale deviating significantly from the usual product quantity purchase/sale pattern
# - *No need to extract a new feature. Normalization will do the job.*
# 
# **Anomaly 7**: Negative quantity or value amount which is a classic mistake
# - Binary feature being 1 when quantity or value amount is negative, and 1 otherwise.

# In[42]:


df.head()


# In[43]:


# Extract hour feature
def add_hour_feature(df):
    hour = df['CreationDate'].dt.hour
    return hour

# Extract weekday feature (0: Monday, 1: Tuesday, 2: Wednesday, 3: Thursday....)
def add_weekday_feature(df):
    weekday = df['CreationDate'].dt.weekday
    return weekday

# Extract difference between the cash settlement date and the share settlement date 
def date_difference(df):
    date_difference = df['CashSettlementDate'] - df['ShareSettlementDate']
    num_days = date_difference.dt.days
    return num_days


# In[44]:


# Creating a copy of our dataframe
df_copy = df.copy()


# In[45]:


df_copy['hour'] = add_hour_feature(df_copy)
df_copy['weekday'] = add_weekday_feature(df_copy)
df_copy['date_difference'] = date_difference(df_copy)
df_copy


# In[46]:


# Exract binary field being 1 if transactions are occuring outside of working days and working hours and 0 otherwise
def detect_outside_working_time(df):
    # Define working hours
    start_working_hour = 8
    end_working_hour = 18
    # Define non-working day (Friday)
    non_working_day = 4

    def is_outside_working_time(row):
        if row['weekday'] != non_working_day:
            if start_working_hour <= row['hour'] < end_working_hour:
                return 0
        return 1
    
    outside_working_time = df.apply(is_outside_working_time, axis=1)
    return outside_working_time


# Extract binary field being 1 if the date difference is negative and 0 otherwise
def detect_negative_date_difference(df):
    negative_date_difference = (df['date_difference'] < 0).astype(int)
    return negative_date_difference

# Extract ratio between unit price and the average price (If ratio is big, then anomalous)
# Added 1 in the denominator to handle values that are 0
def unitprice_avgcost_ratio(df):
    unitprice_avgcost_ratio = df['UnitPrice']/(df['AVGCost']+1)
    return unitprice_avgcost_ratio

# Extract binary field being 1 if the local value rate is an outlier relative to the usual local value rate for the same currency and 0 otherwise.
def detect_localvaluerate_outliers(df):
    def is_outlier(s):
        z_scores = (s - s.mean()) / s.std()
        return np.abs(z_scores) > 3  # Adjust the threshold as needed
    
    grouped = df.groupby('ValueCurrencyId')['LocalValueRate']
    outliers = grouped.transform(is_outlier)
    return outliers.astype(int)

def detect_negative_quantity_or_valueamount(df):
    negative_quantity_or_valueamount = ((df['Quantity'] <= 0) | (df['ValueAmount'] <= 0)).astype(int)
    return negative_quantity_or_valueamount


# In[47]:


def add_extra_features(df):
    outside_working_time = detect_outside_working_time(df)
    negative_date_difference = detect_negative_date_difference(df)
    unitprice_avgcost_ratio_val = unitprice_avgcost_ratio(df)
    localvaluerate_outliers = detect_localvaluerate_outliers(df)
    negative_quantity_or_valueamount = detect_negative_quantity_or_valueamount(df)
    
    # Create a new DataFrame with the additional features
    new_columns = ['outside_working_time', 'negative_date_difference','unitprice_avgcost_ratio', 'localvaluerate_outliers', 'negative_quantity_valueamount']
    
    extra_features_df = pd.DataFrame({
        'outside_working_time' : outside_working_time,
        'negative_date_difference': negative_date_difference,
        'unitprice_avgcost_ratio': unitprice_avgcost_ratio_val,
        'localvaluerate_outliers': localvaluerate_outliers,
        'negative_quantity_valueamount': negative_quantity_or_valueamount
    }, columns=new_columns)
    
    # Concatenate the new DataFrame with the original DataFrame
    combined_df = pd.concat([df, extra_features_df], axis=1)
    
    return combined_df


# ##### Testing the function:

# In[48]:


df_copy_new = add_extra_features(df_copy)
df_copy_new[df_copy_new['Anomaly Type']==5]


# #### Analyzing Relationships with the Anomaly feature and Anomaly Type Feature

# In[49]:


corr = df_copy_new.corr()


# In[50]:


corr["Anomaly"].sort_values(ascending=False)


# In[51]:


sorted_anomaly = corr["Anomaly"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# Perfect! The new extracted features have a higher correlation with the Anomaly feature than most of the existing!

# In[52]:


anomaly1 = df_copy_new[(df_copy_new['Anomaly Type'] == 1) | (df_copy_new['Anomaly Type'] == 0)]
corr1 = anomaly1.corr()
sorted_anomaly = corr1["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 1 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[53]:


anomaly2 = df_copy_new[(df_copy_new['Anomaly Type'] == 2) | (df_copy_new['Anomaly Type'] == 0)]
corr2 = anomaly2.corr()
sorted_anomaly = corr2["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 2 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[54]:


anomaly3 = df_copy_new[(df_copy_new['Anomaly Type'] == 3) | (df_copy_new['Anomaly Type'] == 0)]
corr3 = anomaly3.corr()
sorted_anomaly = corr3["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 3 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[55]:


anomaly4 = df_copy_new[(df_copy_new['Anomaly Type'] == 4) | (df_copy_new['Anomaly Type'] == 0)]
corr4 = anomaly4.corr()
sorted_anomaly = corr4["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 4 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[56]:


anomaly5 = df_copy_new[(df_copy_new['Anomaly Type'] == 5) | (df_copy_new['Anomaly Type'] == 0)]
corr5 = anomaly5.corr()
sorted_anomaly = corr5["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 5  Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[57]:


anomaly6 = df_copy_new[(df_copy_new['Anomaly Type'] == 6) | (df_copy_new['Anomaly Type'] == 0)]
corr6 = anomaly6.corr()
sorted_anomaly = corr6["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 6 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[58]:


anomaly7 = df_copy_new[(df_copy_new['Anomaly Type'] == 7) | (df_copy_new['Anomaly Type'] == 0)]
corr7 = anomaly7.corr()
sorted_anomaly = corr7["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
plt.figure(figsize=(10, 8))
plt.barh(sorted_anomaly.index, sorted_anomaly.values)
plt.xlabel("Correlation Value")
plt.ylabel("Feature")
plt.title("Anomaly Type 7 Correlation with Different Variables")
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()


# In[59]:


anomaly_types = [1, 2, 3, 4, 5, 6, 7]
num_anomaly_types = len(anomaly_types)

num_rows = 4
num_cols = 2
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 20))

for i, anomaly_type in enumerate(anomaly_types):
    row = i // num_cols
    col = i % num_cols
    
    ax = axes[row, col]
    anomaly_data = df_copy_new[(df_copy_new['Anomaly Type'] == anomaly_type) | (df_copy_new['Anomaly Type'] == 0)]
    corr = anomaly_data.corr()
    sorted_anomaly = corr["Anomaly Type"].sort_values(ascending=False).drop(index=['Anomaly', 'Anomaly Type'])
    
    ax.barh(sorted_anomaly.index, sorted_anomaly.values)
    ax.set_xlabel("Correlation Value")
    ax.set_ylabel("Feature")
    ax.set_title(f"Anomaly Type {anomaly_type} Correlation")
    ax.tick_params(axis='x', rotation=90)
    
    if i == num_anomaly_types:
        ax.axis('off')

plt.tight_layout()
plt.show()


# By using FunctionTransformer, we ensure that the custom transformations seamlessly integrate into the scikit-learn workflow and can be easily combined with other transformers and models within pipelines.

# In[60]:


# Make add_extra_features() a FunctionFransformer
feature_adder = FunctionTransformer(add_extra_features, validate = False)


# In[61]:


# Extracting hour, weekday and date_difference features from original dataset then dropping the features of type datetime
df['hour'] = add_hour_feature(df)
df['weekday'] = add_weekday_feature(df)
df['date_difference'] = date_difference(df)
df = df.drop(['CashSettlementDate','ShareSettlementDate','CreationDate'], axis = 1)
df


# In[62]:


# Stratified Split
train, valid = train_test_split(df, test_size=0.2, random_state=11, shuffle=True, stratify=df.Anomaly)


# In[63]:


y_type_train = train['Anomaly Type']
y_type_valid = valid['Anomaly Type']


# In[64]:


# Extract X and y from the train dataset
X_train = train.drop(['Anomaly', 'Anomaly Type'], axis=1)
y_train = train[['Anomaly']]

# Extract X and y from the valid dataset
X_valid = valid.drop(['Anomaly', 'Anomaly Type'], axis=1)
y_valid = valid[['Anomaly']]


# ##### Testing the feature engineering transformer

# In[65]:


X_train_sample = X_train.sample(n=400, random_state = 0)
X_valid_sample = X_valid.sample(n=200, random_state = 0)


# In[66]:


X_train_sample_tr = feature_adder.fit_transform(X_train_sample)


# In[67]:


X_train_sample_tr


# In[68]:


X_valid_sample_tr = feature_adder.transform(X_valid_sample)


# In[69]:


X_valid_sample_tr


# ##### Deploying the feature engineering transformer

# In[70]:


X_train = feature_adder.fit_transform(X_train)
X_valid = feature_adder.transform(X_valid)


# ## Preprocessing

# In[71]:


numerical_cols = ['Quantity','UnitPrice','AVGCost','ExchangeFees', 'BrokerageFees','TicketCharges', 'SettlementFees', 'OtherFees',
                  'ValueAmount', 'LocalValueRate', 'LocalEquivalent', 'hour', 'weekday', 'date_difference', 
                  'unitprice_avgcost_ratio']
categorical_cols = ['TypeId', 'SubTypeId', 'MethodId', 'HoldingId','ClientId', 'ProductId', 'BrokerId', 'TradingAccountId', 'ShareAccountId', 'CashBankAccountId', 'CashThirdPartyAccountId', 'ValueCurrencyId']
binary_cols = ['outside_working_time', 'negative_date_difference', 'localvaluerate_outliers', 'negative_quantity_valueamount']


# In[72]:


len(numerical_cols) + len(categorical_cols) + len(binary_cols)


# In[73]:


# Standardize the numerical data
scaler = StandardScaler()


# In[74]:


# fit the scaler on the numerical columns of X_train
scaler.fit(X_train[numerical_cols])


# In[75]:


# Transform the numerical columns of the X_train and X_valid
X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_valid[numerical_cols] = scaler.transform(X_valid[numerical_cols])


# In[76]:


X_train[numerical_cols]


# ## Evaluating the Logistic Regression using all features

# In[77]:


model = LogisticRegression()


# In[78]:


model.fit(X_train,  y_train)


# In[79]:


# predict probabilities on train and valid sets
y_train_scores = model.predict_proba(X_train)[:, 1]
y_valid_scores = model.predict_proba(X_valid)[:,1]

# calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# plot ROC curves
plt.plot(train_fpr, train_tpr,'-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr,'-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves before Feature Selection')
plt.legend(loc = 'lower right')
plt.legend()
plt.show()


# In[80]:


# Confusion Matrix
y_pred_values = model.predict(X_valid)
cf_matrix = confusion_matrix(y_valid, y_pred_values)
sns.heatmap(cf_matrix, annot = True)
plt.title('Confusion Matrix for Baseline')


# In[81]:


print(classification_report(y_valid, y_pred_values))


# ## Feature Selection
# 
# Selecting features based on feature importance and correlation with the anomaly each anomaly type:
# 
# - Anomaly Type 1: 'outside_working_time'
# - Anomaly Type 2: 'negative_date_difference'
# - Anomaly Type 3: 'unitprice_avgcost_ratio', 'ValueAmount', 'LocalEquivalent'
# - Anomaly Type 4: 'TicketCharges'
# - Anomaly Type 5: 'localvaluerate_outliers', 'LocalValueRate'
# - Anomaly Type 6: 'Quantity'
# - Anomaly Type 7: 'negative_quantity_valueamount'

# In[82]:


features_to_select = ['Quantity', 'TicketCharges', 'ValueAmount', 'LocalValueRate', 'LocalEquivalent', 
                        'outside_working_time', 'negative_date_difference', 'unitprice_avgcost_ratio', 
                      'localvaluerate_outliers','negative_quantity_valueamount']


# In[83]:


X_train_new = X_train[features_to_select]
X_valid_new = X_valid[features_to_select]


# ## Evaluating the Logistic Regression using selected features

# In[84]:


model = LogisticRegression()


# In[85]:


model.fit(X_train_new,  y_train)


# In[88]:


# predict probabilities on train and valid sets
y_train_scores = model.predict_proba(X_train_new)[:, 1]
y_valid_scores = model.predict_proba(X_valid_new)[:,1]

# calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# plot ROC curves
plt.plot(train_fpr, train_tpr,'-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr,'-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves after Feature Selection')
plt.legend(loc = 'lower right')
plt.legend()
plt.show()


# In[89]:


# Confusion Matrix
y_pred_values = model.predict(X_valid_new)
cf_matrix = confusion_matrix(y_valid, y_pred_values)
sns.heatmap(cf_matrix, annot = True)
plt.title('Confusion Matrix for Baseline')


# In[90]:


print(classification_report(y_valid, y_pred_values))


# In[94]:


df_copy_new.to_csv("Data with Extracted Features.csv", index = False)


# _**Results**_:
# 
# The model's performance improved after extracting new features and selecting the most important features among all.

# ### t-SNE Scatterplot 

# In[93]:


y_train_new = y_train.reset_index(drop = True)


# In[94]:


# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_train_new)
df_tsne = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
df_tsne['Anomaly'] = y_train_new 


# In[95]:


df_tsne


# In[96]:


### Create a scatter plot
sns.scatterplot(x='TSNE1',y='TSNE2', hue = 'Anomaly', palette={0: 'blue', 1: 'red'}, data=df_tsne)
plt.title('t-SNE Scatter Plot')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()


# ____
# # 4 Model Optimization and Selection

# ## Supervised Machine Learning
# 
# In this section, I will train and optimize the hyperparameters of Logistic Regression, KNN, Gaussian Naïve Bayes, SVM, Decision Tree, Random Forest, and XGBoost to further improve the AUC and macro-averaged Recall on validation datasets.

# In[97]:


# Function for training and evaluating a model

def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    # Fit the model
    model.fit(X_train, y_train.to_numpy().ravel())

    # Predict probabilities on train and validation sets
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_valid_scores = model.predict_proba(X_valid)[:, 1]

    # Calculate AUC scores on train and validation sets
    train_auc = roc_auc_score(y_train, y_train_scores)
    valid_auc = roc_auc_score(y_valid, y_valid_scores)

    # Calculate macro-averaged recall for validation data
    y_valid_pred_labels = model.predict(X_valid)
    valid_recall = recall_score(y_valid, y_valid_pred_labels, average='macro')

    # Display Train AUC, Valid AUC, and Macro-Averaged Recall
    print(f'Train AUC: {train_auc:.3f}')
    print(f'Valid AUC: {valid_auc:.3f}')
    print(f'Macro-Averaged Recall: {valid_recall:.3f}')


# In[98]:


# Function for tuning the hyperparameters of a model. The metric used for scoring is the ROC-AUC

def tune_hyperparameters(model, param_grid, X_train, y_train, X_valid, y_valid):
    
    # Perform grid search with 5-fold cross-validation
    gs = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5)
    
    # Fit the model on the training data
    gs.fit(X_train, y_train.to_numpy().ravel())
    
    # Predict on the training and validation data
    y_train_pred = gs.predict_proba(X_train)[:, 1]
    y_valid_pred = gs.predict_proba(X_valid)[:, 1]
    
    # Calculate the AUC score for the training and validation data
    train_auc = roc_auc_score(y_train, y_train_pred)
    valid_auc = roc_auc_score(y_valid, y_valid_pred)
    
    # Calculate macro-averaged recall for validation data
    y_valid_pred_labels = gs.predict(X_valid)
    valid_recall = recall_score(y_valid, y_valid_pred_labels, average='macro')
    
    # Print the best hyperparameters, AUC scores, and macro-averaged recall for the model
    print(f'Best Hyperparameters for {model.__class__.__name__}:\n{gs.best_params_}')
    print(f'Train AUC: {train_auc:.3f}')
    print(f'Valid AUC: {valid_auc:.3f}')
    print(f'Macro-Averaged Recall: {valid_recall:.3f}')


# In[99]:


def evaluate_best_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train_new, y_train)

    # predict probabilities on train and valid sets
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_valid_scores = model.predict_proba(X_valid)[:,1]

    # calculate AUC scores on train and valid sets
    train_auc = roc_auc_score(y_train, y_train_scores)
    valid_auc = roc_auc_score(y_valid, y_valid_scores)

    # calculate false positive rates and true positive rates for ROC curves
    train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
    valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

    # plot ROC curves
    plt.plot(train_fpr, train_tpr,'-r', label=f"Train AUC = {train_auc:.3f}")
    plt.plot(valid_fpr, valid_tpr,'-b',  label=f"Valid AUC = {valid_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc = 'lower right')
    plt.legend()
    plt.show()

    # Confusion Matrix
    y_pred_values = model.predict(X_valid)
    cf_matrix = confusion_matrix(y_valid, y_pred_values)
    print(cf_matrix)
    sns.heatmap(cf_matrix, annot = True)
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_valid, y_pred_values))


# ### Logistic Regression

# In[94]:


lr = LogisticRegression()


# #### Evaluating the model with default parameters

# In[95]:


# Call the evaluate_model function
evaluate_model(lr, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[96]:


# Define the parameter grid for hyperparameter tuning
param_grid_lr = {
    'class_weight': ['balanced'],
    'C': [0.1, 1, 10, 100],
    'max_iter': [100, 200, 500]
}


# In[97]:


tune_hyperparameters(lr, param_grid_lr, X_train_new, y_train, X_valid_new, y_valid)


# In[98]:


lr_best = LogisticRegression(C = 1, class_weight ='balanced', max_iter = 100)
evaluate_best_model(lr_best, X_train_new, y_train, X_valid_new, y_valid)


# ### K-Nearest Neighbors

# In[99]:


knn = KNeighborsClassifier()


# #### Evaluating the model with default parameters

# In[100]:


# Call the evaluate_model function
evaluate_model(knn, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[101]:


# Define the parameter grid for hyperparameter tuning
param_grid_knn = {
    'weights': ['uniform'],
    'n_neighbors': [3, 5, 7, 9]
}


# In[102]:


tune_hyperparameters(knn, param_grid_knn, X_train_new, y_train, X_valid_new, y_valid)


# In[103]:


knn_best = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')
evaluate_best_model(knn_best, X_train_new, y_train, X_valid_new, y_valid)


# ### Gaussian Naive Bayes

# In[104]:


nb = GaussianNB()


# #### Evaluating the model with default parameters

# In[105]:


# Call the evaluate_model function
evaluate_model(nb, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[106]:


# Define the parameter grid for hyperparameter tuning
param_grid_nb = {
    'priors': [None, [0.1, 0.9]], # to account for the imbalanced target
    'var_smoothing': [1e-9, 1e-7, 1e-5]
}


# In[107]:


tune_hyperparameters(nb, param_grid_nb, X_train_new, y_train, X_valid_new, y_valid)


# In[108]:


nb_best = GaussianNB(priors = None, var_smoothing = 1e-9)
evaluate_best_model(nb_best, X_train_new, y_train, X_valid_new, y_valid)


# ### Support Vector Machine 

# In[109]:


svm = SVC(kernel='rbf', probability=True)


# #### Evaluating the model with default parameters

# In[110]:


# Call the evaluate_model function
evaluate_model(svm, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[111]:


# Define the parameter grid for hyperparameter tuning
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'class_weight': [None,'balanced'],
    'kernel': ['rgb', 'linear', 'sigmoid']
}


# In[112]:


tune_hyperparameters(svm, param_grid_svm, X_train_new, y_train, X_valid_new, y_valid)


# In[113]:


svm_best = SVC(C = 0.1, class_weight = 'balanced', gamma = 'scale', kernel = 'linear', probability = True)
evaluate_best_model(svm_best, X_train_new, y_train, X_valid_new, y_valid)


# ### Decision Tree

# In[114]:


dt = DecisionTreeClassifier()


# #### Evaluating the model with default parameters

# In[115]:


# Call the evaluate_model function
evaluate_model(dt, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[116]:


# Define the parameter grid for hyperparameter tuning
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}


# In[117]:


tune_hyperparameters(dt, param_grid_dt, X_train_new, y_train, X_valid_new, y_valid)


# In[118]:


dt_best = DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = None, min_samples_leaf = 1, min_samples_split = 10, splitter = 'random')
evaluate_best_model(dt_best, X_train_new, y_train, X_valid_new, y_valid)


# ### Random Forest

# In[119]:


rf = RandomForestClassifier()


# #### Evaluating the model with default parameters

# In[120]:


# Call the evaluate_model function
evaluate_model(rf, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[121]:


# Define the parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'class_weight': ['balanced', None]
}


# In[122]:


tune_hyperparameters(rf, param_grid_rf, X_train_new, y_train, X_valid_new, y_valid)


# In[123]:


rf_best = RandomForestClassifier(class_weight = None, max_depth = 10, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 100)
evaluate_best_model(rf_best, X_train_new, y_train, X_valid_new, y_valid)


# ### XGBoost

# In[124]:


xgb = XGBClassifier()


# #### Evaluating the model with default parameters

# In[125]:


# Call the evaluate_model function
evaluate_model(xgb, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter tuning

# In[126]:


# Define the parameter grid for hyperparameter tuning
param_grid_xgb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 5]
}


# In[127]:


tune_hyperparameters(xgb, param_grid_xgb, X_train_new, y_train, X_valid_new, y_valid)


# In[128]:


xgb_best = XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.1, max_depth = 3, n_estimators = 100, scale_pos_weight = 5, subsample = 0.8)
evaluate_best_model(xgb_best, X_train_new, y_train, X_valid_new, y_valid)


# ## Supervised Deep Learning
# 
# In this section, I will train and optimize the hyperparameters of a Deep Neural Network. I tried three types, with one, two and three hidden layers respectively. 

# ### 1 hidden layer

# In[100]:


X_train_new.shape


# In[101]:


# Define the DNN model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_new.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[102]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


# In[103]:


# Define a function to build the model with hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train_new.shape[1],)))

    # Tune the number of units in the dense layer
    hp_units_layer = hp.Int('units_layer', min_value=2, max_value=8, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer, activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy', metrics=[keras.metrics.AUC()])

    return model


# In[104]:


# Initialize the tuner 
tuner = kt.RandomSearch(build_model,
                        objective = Objective('val_auc', direction='max'),
                        max_trials= 10,
                        directory=r'C:\Users\l.baltaji\Desktop\Capstone Project\Methodology and Source Code',  # Change to a directory where you want to save the results
                        project_name='hyperparameter_tuning1',
                       overwrite = True)


# In[105]:


# Perform the hyperparameter search
tuner.search(X_train_new, y_train, epochs=10, validation_data=(X_valid_new, y_valid))


# In[106]:


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# In[107]:


print("Best Hyperparameters:")
print("Units in the first dense layer:", best_hps.get('units_layer'))
print("Learning rate:", best_hps.get('learning_rate'))


# In[108]:


# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)


# In[109]:


# Train the final model and get the training history
history = final_model.fit(X_train_new, y_train, epochs=50, validation_data=(X_valid_new, y_valid))


# In[110]:


# Plot the AUC-epoch graph
plt.plot(history.history['auc_1'], label='train_auc', color='blue')
plt.plot(history.history['val_auc_1'], label='val_auc', color='red')
plt.title('AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Plot the loss-epoch graph
plt.plot(history.history['loss'], label='train_loss', color='blue')
plt.plot(history.history['val_loss'], label='val_loss', color='red')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[111]:


# Predict probabilities on train and valid sets
y_train_scores = final_model.predict(X_train_new)
y_valid_scores = final_model.predict(X_valid_new)

# Extract the probability for the positive class (class 1)
y_train_scores = y_train_scores[:, 0]
y_valid_scores = y_valid_scores[:, 0]

# Calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# Calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# Plot ROC curves
plt.plot(train_fpr, train_tpr, '-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr, '-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()


# In[112]:


# Confusion Matrix
y_pred_values = final_model.predict(X_valid_new)
y_pred_classes = (y_pred_values > 0.5).astype(int)
cf_matrix = confusion_matrix(y_valid, y_pred_classes)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title('Confusion Matrix for Chosen Deep Learning Model')
plt.show()


# In[113]:


# Calculate the classification report
print(classification_report(y_valid, y_pred_classes))


# ### 2 hidden layers

# In[114]:


# Define the DNN model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_new.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[115]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


# In[116]:


# Define a function to build the model with hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train_new.shape[1],)))

    # Tune the number of units in the first dense layer
    hp_units_layer1 = hp.Int('units_layer1', min_value=4, max_value=8, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer1, activation='relu'))

    # Tune the number of units in the second dense layer
    hp_units_layer2 = hp.Int('units_layer2', min_value=2, max_value=4, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer2, activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy', metrics=[keras.metrics.AUC()])

    return model


# In[117]:


# Initialize the tuner 
tuner = kt.RandomSearch(build_model,
                        objective = Objective('val_auc', direction='max'),
                        max_trials= 10,
                        directory=r'C:\Users\l.baltaji\Desktop\Capstone Project\Methodology and Source Code',  
                        project_name='hyperparameter_tuning2',
                       overwrite = True)


# In[118]:


# Perform the hyperparameter search
tuner.search(X_train_new, y_train, epochs=10, validation_data=(X_valid_new, y_valid))


# In[119]:


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# In[120]:


print("Best Hyperparameters:")
print("Units in the first dense layer:", best_hps.get('units_layer1'))
print("Units in the first dense layer:", best_hps.get('units_layer2'))
print("Learning rate:", best_hps.get('learning_rate'))


# In[121]:


# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)


# In[122]:


# Train the final model and get the training history
history2 = final_model.fit(X_train_new, y_train, epochs=50, validation_data=(X_valid_new, y_valid))


# In[123]:


# Plot the AUC-epoch graph
plt.plot(history2.history['auc_1'], label='train_auc', color='blue')
plt.plot(history2.history['val_auc_1'], label='val_auc', color='red')
plt.title('AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Plot the loss-epoch graph
plt.plot(history2.history['loss'], label='train_loss', color='blue')
plt.plot(history2.history['val_loss'], label='val_loss', color='red')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[124]:


# Predict probabilities on train and valid sets
y_train_scores = final_model.predict(X_train_new)
y_valid_scores = final_model.predict(X_valid_new)

# Extract the probability for the positive class (class 1)
y_train_scores = y_train_scores[:, 0]
y_valid_scores = y_valid_scores[:, 0]

# Calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# Calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# Plot ROC curves
plt.plot(train_fpr, train_tpr, '-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr, '-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()


# In[125]:


# Confusion Matrix
y_pred_values = final_model.predict(X_valid_new)
y_pred_classes = (y_pred_values > 0.5).astype(int)
cf_matrix = confusion_matrix(y_valid, y_pred_classes)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title('Confusion Matrix for Chosen Deep Learning Model')
plt.show()


# In[126]:


# Calculate the classification report
print(classification_report(y_valid, y_pred_classes))


# ### 3 hidden layers

# In[127]:


# Define the DNN model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_new.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[128]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])


# In[129]:


# Define a function to build the model with hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train_new.shape[1],)))

    # Tune the number of units in the first hidden layer
    hp_units_layer1 = hp.Int('units_layer1', min_value=6, max_value=8, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer1, activation='relu'))

    # Tune the number of units in the second hidden layer
    hp_units_layer2 = hp.Int('units_layer2', min_value=4, max_value=6, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer2, activation='relu'))

    # Tune the number of units in the third hidden layer
    hp_units_layer3 = hp.Int('units_layer3', min_value=2, max_value=4, step=2)
    model.add(keras.layers.Dense(units=hp_units_layer3, activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy', metrics=[keras.metrics.AUC()])

    return model


# In[130]:


# Initialize the tuner 
tuner = kt.RandomSearch(build_model,
                        objective = Objective('val_auc', direction='max'),
                        max_trials= 10,
                        directory=r'C:\Users\l.baltaji\Desktop\Capstone Project\Methodology and Source Code',  # Change to a directory where you want to save the results
                        project_name='hyperparameter_tuning3',
                       overwrite = True)


# In[131]:


# Perform the hyperparameter search
tuner.search(X_train_new, y_train, epochs=10, validation_data=(X_valid_new, y_valid))


# In[132]:


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# In[133]:


print("Best Hyperparameters:")
print("Units in the first dense layer:", best_hps.get('units_layer1'))
print("Units in the second dense layer:", best_hps.get('units_layer2'))
print("Units in the third dense layer:", best_hps.get('units_layer3'))
print("Learning rate:", best_hps.get('learning_rate'))


# In[134]:


# Build the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hps)


# In[135]:


# Train the final model and get the training history
history3 = final_model.fit(X_train_new, y_train, epochs=50, validation_data=(X_valid_new, y_valid))


# In[136]:


# Plot the AUC-epoch graph
plt.plot(history3.history['auc_1'], label='train_auc', color='blue')
plt.plot(history3.history['val_auc_1'], label='val_auc', color='red')
plt.title('AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Plot the loss-epoch graph
plt.plot(history3.history['loss'], label='train_loss', color='blue')
plt.plot(history3.history['val_loss'], label='val_loss', color='red')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[137]:


# Predict probabilities on train and valid sets
y_train_scores = final_model.predict(X_train_new)
y_valid_scores = final_model.predict(X_valid_new)

# Extract the probability for the positive class (class 1)
y_train_scores = y_train_scores[:, 0]
y_valid_scores = y_valid_scores[:, 0]

# Calculate AUC scores on train and valid sets
train_auc = roc_auc_score(y_train, y_train_scores)
valid_auc = roc_auc_score(y_valid, y_valid_scores)

# Calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)

# Plot ROC curves
plt.plot(train_fpr, train_tpr, '-r', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr, '-b',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()


# In[138]:


# Confusion Matrix
y_pred_values = final_model.predict(X_valid_new)
y_pred_classes = (y_pred_values > 0.5).astype(int)
cf_matrix = confusion_matrix(y_valid, y_pred_classes)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.title('Confusion Matrix for Chosen Deep Learning Model')
plt.show()


# In[139]:


# Calculate the classification report
print(classification_report(y_valid, y_pred_classes))


# ## Unsupervised Machine Learning

# In[88]:


def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    # Fit the model
    model.fit(X_train)
    
    # Binary Labels of the training data: 0 for normal, 1 for anomalies
    y_train_pred = model.labels_
    
    # the outlier scores of the training data
    y_train_scores = model.decision_scores_
    
    # Predict if a particular sample is an outlier
    y_valid_pred = model.predict(X_valid)
    
    # Predict raw anomaly score of the transaction
    y_valid_scores = model.decision_function(X_valid)
    
    # Predict the probability of a sample being an outlier
    y_valid_proba = model.predict_proba(X_valid)

    # Calculate AUC scores on train and validation sets
    train_auc = roc_auc_score(y_train, y_train_scores)
    valid_auc = roc_auc_score(y_valid, y_valid_scores)

    # Calculate macro-averaged recall for validation data
    y_valid_pred_labels = model.predict(X_valid)
    valid_recall = recall_score(y_valid, y_valid_pred_labels, average='macro')

    # Display Train AUC, Valid AUC, and Macro-Averaged Recall
    print(f'Train AUC: {train_auc:.3f}')
    print(f'Valid AUC: {valid_auc:.3f}')
    print(f'Macro-Averaged Recall: {valid_recall:.3f}')


# In[89]:


# Function for tuning the hyperparameters of a model. The metric used for scoring is the ROC-AUC
def tune_hyperparameters(model, param_grid, X_train, y_train, X_valid, y_valid):
    gs = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
    
    gs.fit(X_train)

    best_model = gs.best_estimator_
    
    # Predict if a particular sample is an outlier
    y_train_pred = best_model.predict(X_train)
    
    # The outlier scores of the training data
    y_train_scores = best_model.decision_scores_
    
    # Predict if a particular sample is an outlier
    y_valid_pred = best_model.predict(X_valid)
    
    # Predict raw anomaly score of the transaction
    y_valid_scores = best_model.decision_function(X_valid)
    
    # Calculate AUC scores on train and validation sets
    train_auc = roc_auc_score(y_train, y_train_scores)
    valid_auc = roc_auc_score(y_valid, y_valid_scores)
    
    # Calculate macro-averaged recall for validation data
    valid_recall = recall_score(y_valid, y_valid_pred, average='macro')

    # Print the best hyperparameters, AUC scores, and macro-averaged recall for the model
    print(f'Best Hyperparameters for {model.__class__.__name__}:\n{gs.best_params_}')
    print(f'Train AUC: {train_auc:.3f}')
    print(f'Valid AUC: {valid_auc:.3f}')
    print(f'Macro-Averaged Recall: {valid_recall:.3f}')


# In[90]:


def evaluate_best_model2(model, X_train, y_train, X_valid, y_valid):
    # Fit the model
    model.fit(X_train)
    
    # Binary Labels of the training data: 0 for normal, 1 for anomalies
    y_train_pred = model.labels_
    
    # the outlier scores of the training data
    y_train_scores = model.decision_scores_
    
    # Predict if a particular sample is an outlier
    y_valid_pred = model.predict(X_valid)
    
    # Predict raw anomaly score of the transaction
    y_valid_scores = model.decision_function(X_valid)
    
    # Predict the probability of a sample being an outlier
    y_valid_proba = model.predict_proba(X_valid)

    # calculate AUC scores on train and valid sets
    train_auc = roc_auc_score(y_train, y_train_scores)
    valid_auc = roc_auc_score(y_valid, y_valid_scores)

    # calculate false positive rates and true positive rates for ROC curves
    train_fpr, train_tpr, _ = roc_curve(y_train, y_train_scores)
    valid_fpr, valid_tpr, _ = roc_curve(y_valid, y_valid_scores)
    
    sns.set_style('white')
    sns.color_palette()
    # plot ROC curves
    plt.plot(train_fpr, train_tpr,'red', label=f"Train AUC = {train_auc:.3f}")
    plt.plot(valid_fpr, valid_tpr,'blue',  label=f"Valid AUC = {valid_auc:.3f}")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc = 'lower right')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cf_matrix = confusion_matrix(y_valid, y_valid_pred)
    print(cf_matrix)
    sns.heatmap(cf_matrix, annot = True)
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_valid, y_valid_pred))


# ### Isolation Forest

# In[171]:


isf = IForest(random_state = 42)


# #### Evaluating the model with default parameters

# In[172]:


evaluate_model(isf, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter Tuning

# In[173]:


param_grid_isf = {
    'n_estimators': [100, 200, 300],
    'max_samples': [0.1, 0.2, 0.3],
    'contamination': [0.01, 0.05, 0.1]
}


# In[174]:


tune_hyperparameters(isf, param_grid_isf, X_train_new, y_train, X_valid_new, y_valid)


# In[91]:


isf_best = IForest(random_state = 42, contamination = 0.01, max_samples = 0.1, n_estimators = 100)
evaluate_best_model2(isf_best, X_train_new, y_train, X_valid_new, y_valid)


# In[92]:


# Calculate the anomaly scores for X_valid_new
anomaly_scores = -isf_best.decision_function(X_valid_new)


# In[93]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_valid, pd.DataFrame):
    y_valid = y_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Label': y_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
sns.boxplot(x='Anomaly_Label', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Label')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# In[94]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_type_valid, pd.DataFrame):
    y_type_valid = y_type_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Type': y_type_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Anomaly_Type', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Type')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# ### Local Outlier Factor

# In[208]:


lof = LOF()


# In[209]:


# Print model parameters
for key, value in lof.get_params().items():
    print(f"{key}: {value}")


# #### Evaluating the model with default parameters

# In[210]:


evaluate_model(lof, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter Tuning

# In[211]:


param_grid_lof = {
        'contamination': [0.1, 0.01, 0.001],
        'n_neighbors':[20]
    }


# In[212]:


tune_hyperparameters(lof, param_grid_lof, X_train_new, y_train, X_valid_new, y_valid)


# In[95]:


lof_best = LOF(contamination = 0.1, n_neighbors = 20)
evaluate_best_model2(lof_best, X_train_new, y_train, X_valid_new, y_valid)


# In[96]:


# Calculate the anomaly scores for X_valid_new
anomaly_scores = -lof_best.decision_function(X_valid_new)/1000

# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_valid, pd.DataFrame):
    y_valid = y_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Label': y_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
sns.boxplot(x='Anomaly_Label', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Label')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# In[97]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_type_valid, pd.DataFrame):
    y_type_valid = y_type_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Type': y_type_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Anomaly_Type', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Type')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# ### One Class SVM

# In[222]:


oc_svm = OCSVM()


# In[223]:


# Print model parameters
for key, value in oc_svm.get_params().items():
    print(f"{key}: {value}")


# #### Evaluating the model with default parameters

# In[224]:


evaluate_model(oc_svm, X_train_new, y_train, X_valid_new, y_valid)


# #### Hyperparameter Tuning

# In[225]:


param_grid_oc_svm = {
        'contamination' :[0.1, 0.01],
        'gamma': ['auto', 'scale']
    }


# In[226]:


tune_hyperparameters(oc_svm, param_grid_oc_svm, X_train_new, y_train, X_valid_new, y_valid)


# In[261]:


ocsvm_best = OCSVM(contamination = 0.1,gamma = 'auto')
sns.set_style('white')

evaluate_best_model2(ocsvm_best, X_train_new, y_train, X_valid_new, y_valid)


# In[228]:


# Calculate the anomaly scores for X_valid_new
anomaly_scores = -ocsvm_best.decision_function(X_valid_new)/1000

# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_valid, pd.DataFrame):
    y_valid = y_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Label': y_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
sns.boxplot(x='Anomaly_Label', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Label')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# In[229]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_type_valid, pd.DataFrame):
    y_type_valid = y_type_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores,
    'Anomaly_Type': y_type_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Anomaly_Type', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Type')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# ## Unsupervised Deep Learning: Auto-Encoders

# In[140]:


# Convert y_train to a pandas Series if it's a DataFrame column
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]


# In[188]:


# Keep only the normal data for the training of the autoencoder
mask = (y_train == 0)

X_train_normal = X_train_new[mask]


# In[189]:


# Input Layer
input = tf.keras.layers.Input(shape=(X_train_normal.shape[1],))

# Encoder Layers
encoder = tf.keras.Sequential([
    layers.Dense(8, activation = 'relu'),
    layers.Dense(6, activation = 'relu'),
    layers.Dense(4, activation = 'relu')
])(input)

# Decoder Layers
decoder = tf.keras.Sequential([
    layers.Dense(6, activation = 'relu'),
    layers.Dense(8, activation = 'relu'),
    layers.Dense(X_train_normal.shape[1], activation = 'sigmoid')
])(encoder)

# Create the autoencoder
autoencoder = keras.Model(inputs = input, outputs = decoder)


# In[190]:


# Compile the autoencoder
autoencoder.compile(optimizer = 'adam', loss = 'mae')


# In[191]:


# Fit the autoencoder
history = autoencoder.fit(X_train_normal, X_train_normal,
                         epochs = 20, 
                         batch_size = 64,
                         validation_data = (X_valid_new, X_valid_new),
                         shuffle = True)


# In[205]:


sns.set(style="white")
# Loss-Epochs Plot
plt.plot(history.history['loss'], label = 'Training Loss', color = 'blue')
plt.plot(history.history['val_loss'], label = 'Validation Loss', color = 'red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[195]:


# Predict anomalies in the training dataset
prediction = autoencoder.predict(X_valid_new)

# Get the mean absolute error between actual and reconstruction/prediction
prediction_loss = tf.keras.losses.mae(prediction, X_valid_new)

# Check the prediction loss threshold for 3% of outliers
loss_threshold = np.percentile(prediction_loss, 97)
print(f'The prediction loss threshold for 3% of anomalies is {loss_threshold: .2f}')


# In[196]:


# Check the model performance at 2% threshold
threshold_prediction = [0 if i<loss_threshold else 1 for i in prediction_loss]


# In[197]:


sns.set(style="white")
# Predict anomalies in the training dataset
prediction2 = autoencoder.predict(X_train_new)

# Get the mean absolute error between actual and reconstruction/prediction
prediction_loss2 = tf.keras.losses.mae(prediction2, X_train_new)

# Check the prediction loss threshold for 2% of outliers
loss_threshold = np.percentile(prediction_loss2, 98)

# Calculate false positive rates and true positive rates for ROC curves
train_fpr, train_tpr, _ = roc_curve(y_train, prediction_loss2)
valid_fpr, valid_tpr, _ = roc_curve(y_valid, prediction_loss)

# Plot ROC curves
plt.plot(train_fpr, train_tpr, 'red', label=f"Train AUC = {train_auc:.3f}")
plt.plot(valid_fpr, valid_tpr, 'blue',  label=f"Valid AUC = {valid_auc:.3f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()


# In[198]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_valid, threshold_prediction)
print(conf_matrix)
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion Matrix for Autoencoder')
plt.show()


# In[199]:


# Check the prediction performance
print(classification_report(y_valid, threshold_prediction))


# In[200]:


anomaly_scores_autoencoder = (loss_threshold) / (prediction_loss*27)


# In[201]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_valid, pd.DataFrame):
    y_valid = y_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores_autoencoder,
    'Anomaly_Label': y_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
sns.boxplot(x='Anomaly_Label', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Label')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Label')

# Show the plot
plt.show()


# In[202]:


# Convert y_valid to a pandas Series if it's a DataFrame column
if isinstance(y_type_valid, pd.DataFrame):
    y_type_valid = y_type_valid.iloc[:, 0]

# Create a dataframe for plotting
data = {
    'Anomaly_Score': anomaly_scores_autoencoder,
    'Anomaly_Type': y_type_valid,
}
df2 = pd.DataFrame(data)

# Set up the plotting style
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Anomaly_Type', y='Anomaly_Score', data=df2)

# Add labels and title
plt.xlabel('Anomaly Type')
plt.ylabel('Anomaly Score')
plt.title('Boxplot of Anomaly Scores by Anomaly Type')

# Show the plot
plt.show()


# In[203]:


pickle_out1 = open('autoencoder.pkl', 'wb')
pickle.dump(autoencoder, pickle_out1)
pickle_out1.close()


# In[204]:


pickle_out2 = open('scaler.pkl', 'wb')
pickle.dump(scaler, pickle_out2)
pickle_out2.close()


# In[ ]:





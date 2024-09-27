import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset
df = pd.read_csv('Credit_Card_Defaulter_Prediction.csv')

# viewing data
df

# viewing the first few rows of data (by default it is 5)
df.head()

# viewing the last few rows of data (here 9)
df.tail(9)

# counts rows and column
df.shape

# Check data info
df.info()

# Describes data
df.describe().T
# .T shows it in Transpose form

# checks for any duplicates
len(df[df.duplicated()])

# Taking sum of null values in columns of the data
df.isnull().sum()

# columns of dataset
df.columns



##### Data Preprocessing

## Renaming
# Changing names of column for better understanding and simplicity (using inplace = True to retain the changes throughtout the process)
df.rename(columns = {"default" : "IsDefaulter"}, inplace = True)
df.rename(columns = {"PAY_1" : "PAY_SEPT", "PAY_2" : "PAY_AUG", "PAY_3" : "PAY_JULY", "PAY_4" : "PAY_JUNE", "PAY_5" : "PAY_MAY", "PAY_6" : "PAY_APR"}, inplace = True)
df.rename(columns = {"BILL_AMT1" : "BILL_AMT_SEPT", "BILL_AMT2" : "BILL_AMT_AUG", "BILL_AMT3" : "BILL_AMT_JULY", "BILL_AMT4" : "BILL_AMT_JUNE", "BILL_AMT5" : "BILL_AMT_MAY", "BILL_AMT6" : "BILL_AMT_APR"}, inplace = True)
df.rename(columns = {"PAY_AMT1" : "PAY_AMT_SEPT", "PAY_AMT2" : "PAY_AMT_AUG", "PAY_AMT3" : "PAY_AMT_JULY", "PAY_AMT4" : "PAY_AMT_JUNE", "PAY_AMT5" : "PAY_AMT_MAY", "PAY_AMT6" : "PAY_AMT_APR"}, inplace = True)

## Replacing
# Replacing values with their labels
df.replace({"SEX" : {1 : "Male", 2 : "Female"}}, inplace = True)
df.replace({"EDUCATION" : {1 : "Graduate School", 2 : "University", 3 : "High School", 4 : "Others"}}, inplace = True)
df.replace({"MARRIAGE" : {1 : "Married", 2 : "Single", 3 : "Others"}}, inplace = True)
df.replace({"IsDefaulter" : {"Y" : "Yes", "N" : "No"}}, inplace = True)

df.info()

df.head()


##### Exploratory Data Analysis (EDA)

## Finding Defaulters
# Counting defaulters
df['IsDefaulter'].value_counts()
# value count plot for defaulters
plt.figure(figsize = (6,5))
sns.countplot(x = "IsDefaulter", data = df)

### Categorical Features
## Sex
# Value count for Sex Category
df['SEX'].value_counts()
# count plot for Sex and with respect to Defaulters
fig, axes = plt.subplots(ncols = 2, figsize = (12,5))
sns.countplot(x = 'SEX', ax = axes[0], data = df)
sns.countplot(x = 'SEX', hue = 'IsDefaulter', ax = axes[1], data = df)

## Education
# Value count for Education Category
df['EDUCATION'].value_counts()
# count plot for Education and with respect to Defaulters
fig, axes = plt.subplots(ncols = 2, figsize = (12,7))
sns.countplot(x = 'EDUCATION', ax = axes[0], data = df)
sns.countplot(x = 'EDUCATION', hue = 'IsDefaulter', ax = axes[1], data = df)

## Marriage
# Value count for Education Category
df['MARRIAGE'].value_counts()
# count plot for Education and with respect to Defaulters
fig, axes = plt.subplots(ncols = 2, figsize = (12,7))
sns.countplot(x = 'MARRIAGE', ax = axes[0], data = df)
sns.countplot(x = 'MARRIAGE', hue = 'IsDefaulter', ax = axes[1], data = df)

## Age
# Age wise distribution 
df['AGE'].value_counts()
# value count for Age
plt.figure(figsize = (20,10))
sns.countplot(x = 'AGE', data = df)
# value count for Age with respect to IsDefaulter
plt.figure(figsize = (20,10))
sns.countplot(x = 'AGE', hue = 'IsDefaulter', data = df)


### Checking correlation
df.info()

df.corr(numeric_only = True)

plt.figure(figsize = (20,18))
sns.heatmap(df.corr(numeric_only = True), cmap = 'coolwarm')
plt.show()


### One Hot Encoding
# label encoding
encode_num = {"SEX" : {'Female' : 0, 'Male' : 1}, 'IsDefaulter' : {'No' : 0, 'Yes' : 1}}
df = df.replace(encode_num)
# check for changed labels
df.head()

## creating dummy variables
df = pd.get_dummies(df, columns = ['EDUCATION', 'MARRIAGE'])

df.shape

df.head()

## Removing education and marriage others columns
df.drop(['EDUCATION_Others', 'MARRIAGE_Others'], axis = 1, inplace = True)

df.shape

df.head()


##### Handling Class Imbalance Using SMOTE
### SMOTE ( Synthetic Minority Oversampling Technique)

# importing SMOTE to handle class imbalance
from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(df[(i for i in list(df.describe(include = 'all').columns) if i != 'IsDefaulter')], df['IsDefaulter'])

print('Original unbalanced dataset shape', len(df))
print('Resampled balanced dataset shape', len(y_smote))

# Creating new Data Frame from balanced dataset after using SMOTE
balanced_df = pd.DataFrame(x_smote, columns = [i for i in list(df.describe(include = 'all').columns) if i != 'IsDefaulter'])

# adding target variable to new created Data Frame
balanced_df['IsDefaulter'] = y_smote

balanced_df['IsDefaulter'].value_counts()

balanced_df.shape

# removing feature ID from dataset
balanced_df.drop(['ID'], axis = 1, inplace = True)

# final dataset
balanced_df

# Separating dependent and independent variables
x = balanced_df[(list(i for i in list(balanced_df.describe(include = 'all').columns) if i != 'IsDefaulter'))]
y = balanced_df['IsDefaulter']

x.shape
y.shape

x
y



##### Data Transformation

# importing libraries for data transformation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x = scaler.fit_transform(x)

### Train Test Split
# importing libraries for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42, stratify = y)

x_train.shape

x_test.shape



##### Model Implementation
### Logistic Regression Model

# importing logistic regression and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# fiting data into logistic regression model
logi = LogisticRegression(fit_intercept = True, max_iter = 10000)
logi.fit(x_train, y_train)

# class precision of y
y_pred_logi = logi.predict(x_test)
y_train_pred_logi = logi.predict(x_train)

# getting all score for logistic regression
train_accuracy_logi = round(accuracy_score(y_train_pred_logi, y_train), 3)
accuracy_logi = round(accuracy_score(y_pred_logi, y_test), 3)
precision_score_logi = round(precision_score(y_pred_logi, y_test), 3)
recall_score_logi = round(recall_score(y_pred_logi, y_test), 3)
f1_score_logi = round(f1_score(y_pred_logi, y_test), 3)

print('The accuracy on train data is : ', train_accuracy_logi)
print('The accuracy on test data is : ', accuracy_logi)
print('The precision on test data is : ', precision_score_logi)
print('The recall on test data is : ', recall_score_logi)
print('The f1 score on test data is : ', f1_score_logi)

# Confusion Matrix
labels = ['Not Defaulter', 'Defaulter']
cm_logi = confusion_matrix(y_test, y_pred_logi)
print(cm_logi)

# plotting confusion matrix
ax = plt.subplot()
sns.heatmap(cm_logi, annot = True, ax = ax)

# Labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

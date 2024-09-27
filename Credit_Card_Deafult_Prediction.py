import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Credit_Card_Defaulter_Prediction.csv')


### Data Preprocessing

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


### Exploratory Data Analysis (EDA)

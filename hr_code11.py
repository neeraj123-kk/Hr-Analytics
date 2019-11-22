# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 06:45:06 2019

@author: Niran
"""

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

train = pd.read_csv("train_HR.csv")
train.shape
is_promoted_Count = pd.crosstab(index=train["is_promoted"],columns="count") 
is_promoted_Count
train = train.drop(["employee_id"],axis=1)

#Removing Outliers
####################################
sns.boxplot(x=train['age'])
sns.boxplot(x=train['length_of_service'])
sns.boxplot(x=train['avg_training_score'])

#Training Data- Age outlier removal
#continously need to run this code and box plot, until the outliers are removed
train_Age_Q1 = train['age'].quantile(0.25)
train_Age_Q3 = train['age'].quantile(0.75)
train_Age_IQR = train_Age_Q3 - train_Age_Q1
print(train_Age_Q1,train_Age_Q3,train_Age_IQR)
train = train[~((train.age<(train_Age_Q1-1.5*train_Age_IQR))|(train.age>(train_Age_Q3+1.5*train_Age_IQR)))]
sns.boxplot(x=train['age'])

#Training Data - length_of_service outlier removal
#length_of_service
train_LengthOfService_Q1 = train['length_of_service'].quantile(0.25)
train_LengthOfService_Q3 = train['length_of_service'].quantile(0.75)
train_LengthOfService_IQR = train_LengthOfService_Q3 - train_LengthOfService_Q1
print(train_LengthOfService_Q1,train_LengthOfService_Q3,train_LengthOfService_IQR)
train = train[~((train.length_of_service<(train_LengthOfService_Q1-1.5*train_LengthOfService_IQR))|(train.length_of_service>(train_LengthOfService_Q3+1.5*train_LengthOfService_IQR)))]
sns.boxplot(x=train['length_of_service'])

#Missing value imputation
train.isna().sum()
train.dropna(axis=0,subset=['education'],inplace=True)
train['previous_year_rating'] = train['previous_year_rating'].fillna(0) 
train.isna().sum()

#Checking Skewness
sns.distplot(train['age'])
print(round(skew(train['age']),2)) #0.5

sns.distplot(train['length_of_service'].values)
print(round(skew(train['length_of_service']),2)) #0.59

sns.distplot(train['avg_training_score'].values)
print(round(skew(train['avg_training_score']),2)) #0.42

#Normalizing the training data
#sns.distplot(train['age'])
print("Before log transformation:",skew(train['age']))
print("After log transformation:",skew(np.log10(train['age'])))
sns.distplot(np.log10(train['age']))

print("Before sqrt transformation:",skew(train['length_of_service']))
print("After sqrt transformation:",skew(np.sqrt(train['length_of_service'])))
sns.distplot(np.sqrt(train['length_of_service']))

print("Before log transformation:",skew(train['avg_training_score']))
print("After log transformation:",skew(np.log10(train['avg_training_score'])))
sns.distplot(np.log10(train['avg_training_score']))

train['age'] = np.log10(train['age'])
train['length_of_service'] = np.sqrt(train['length_of_service'])
train['avg_training_score'] = np.log10(train['avg_training_score'])

#One hot encoding
train = pd.get_dummies(train)

#Spliting the data
y = train['is_promoted']
x = train.drop(['is_promoted'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)

#Model Building and predicting
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x,y)
#regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#Validating model
from sklearn.metrics import confusion_matrix 
results = confusion_matrix(y_test, y_pred)
results

from sklearn.metrics import accuracy_score
print("Accuracy score is :",round(accuracy_score(y_test, y_pred),2))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#########################Preprocessing Test Data####################################
test = pd.read_csv("test_HR.csv")
#test = test.drop(["employee_id"],axis=1)

#Removing Outliers
####################################
sns.boxplot(x=test['age'])
sns.boxplot(x=test['length_of_service'])
sns.boxplot(x=test['avg_training_score'])

#Training Data- Age outlier removal
#continously need to run this code and box plot, until the outliers are removed
test_Age_Q1 = test['age'].quantile(0.25)
test_Age_Q3 = test['age'].quantile(0.75)
test_Age_IQR = test_Age_Q3 - test_Age_Q1
print(test_Age_Q1,test_Age_Q3,test_Age_IQR)
test = test[~((test.age<(test_Age_Q1-1.5*test_Age_IQR))|(test.age>(test_Age_Q3+1.5*test_Age_IQR)))]
sns.boxplot(x=test['age'])

#Training Data - length_of_service outlier removal
#length_of_service
test_LengthOfService_Q1 = test['length_of_service'].quantile(0.25)
test_LengthOfService_Q3 = test['length_of_service'].quantile(0.75)
test_LengthOfService_IQR = test_LengthOfService_Q3 - test_LengthOfService_Q1
print(test_LengthOfService_Q1,test_LengthOfService_Q3,test_LengthOfService_IQR)
test = test[~((test.length_of_service<(test_LengthOfService_Q1-1.5*test_LengthOfService_IQR))|(test.length_of_service>(test_LengthOfService_Q3+1.5*test_LengthOfService_IQR)))]
sns.boxplot(x=test['length_of_service'])

#Missing value imputation
test.isna().sum()
test.dropna(axis=0,subset=['education'],inplace=True)
test['previous_year_rating'] = test['previous_year_rating'].fillna(0) 
test.isna().sum()

#Checking Skewness
sns.distplot(test['age'])
print(round(skew(test['age']),2)) #0.51

sns.distplot(test['length_of_service'].values)
print(round(skew(test['length_of_service']),2)) #0.6

sns.distplot(test['avg_training_score'].values)
print(round(skew(test['avg_training_score']),2)) #0.43

#Normalizing the training data
#sns.distplot(train['age'])
print("Before log transformation:",skew(test['age']))
print("After log transformation:",skew(np.log10(test['age'])))
sns.distplot(np.log10(test['age']))

print("Before sqrt transformation:",skew(test['length_of_service']))
print("After sqrt transformation:",skew(np.sqrt(test['length_of_service'])))
sns.distplot(np.sqrt(test['length_of_service']))

print("Before sqrt transformation:",skew(test['avg_training_score']))
print("After sqrt transformation:",skew(np.log10(test['avg_training_score'])))
sns.distplot(np.log10(test['avg_training_score']))

test['age'] = np.log10(test['age'])
test['length_of_service'] = np.sqrt(test['length_of_service'])
test['avg_training_score'] = np.log10(test['avg_training_score'])

#One hot encoding
test = pd.get_dummies(test)

employeeID = pd.DataFrame(test['employee_id'],columns=['employee_id'])
test = test.drop(["employee_id"],axis=1)

#Predicting the value
y_pred_test = regressor.predict(test)

#Converting test data predictions to dataframe
isPromoted = pd.DataFrame(y_pred_test,columns=['IsPromoted'])

is_promoted_Count_Test = pd.crosstab(index=isPromoted["IsPromoted"],columns="count") 
is_promoted_Count_Test

employeeID.reset_index(drop = True,inplace=True)
isPromoted.reset_index(drop = True,inplace=True)

#Concatenating the dataframes
finalData = pd.concat([employeeID,isPromoted],axis=1)

#Writing the dataframe to file
finalData.to_csv("FinalData.csv",index = False)
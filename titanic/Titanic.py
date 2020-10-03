import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import svm,tree
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

## fill mising value
#Train_data
imputer_train = KNNImputer(n_neighbors=5)
imputer_train.fit(train_data.select_dtypes('float64'))
column_float = list(train_data.select_dtypes('float64').columns)
train_data[column_float] = imputer_train.transform(train_data.select_dtypes('float64'))

# Test_data
imputer_test = KNNImputer(n_neighbors=5)
imputer_test.fit(test_data.select_dtypes('float64'))
column_float = list(test_data.select_dtypes('float64').columns)
test_data[column_float] = imputer_test.transform(test_data.select_dtypes('float64'))

train_data.dropna(how = 'all', inplace=True,subset=['Embarked'])
#test_data.dropna(how = 'all', inplace=True,subset=['Fare'])
test_data.Fare.fillna(7.0,inplace = True)

#drop cabin
train_data = train_data.drop(['Cabin'],axis=1)
test_data = test_data.drop(['Cabin'],axis = 1)

# DATA PREPROCESSING
train_df = train_data.drop(['PassengerId','Name','Ticket'],axis = 1)
test_df = test_data.drop(['PassengerId','Name','Ticket'],axis = 1)

sex = {'male' : 1.0, 'female' : 0.0}
train_df['Sex'] = train_df['Sex'].map(sex)
test_df['Sex'] = test_df['Sex'].map(sex)

train_df['Embarked'].value_counts()
embarked = {'S': 1.0, 'C' : 2.0, 'Q': 3.0}
train_df['Embarked'] = train_df['Embarked'].map(embarked)
test_df['Embarked'] = test_df['Embarked'].map(embarked)

# CONVERT STRING TO INT
intType = ['Pclass','SibSp','Parch']
for col in intType:
    train_df[col] = train_df[col].astype(float)
    test_data[col] = test_data[col].astype(float)

# drop label
label = train_df.loc[:,'Survived']
train_df = train_df.drop('Survived',axis = 1)

X, y = plt.subplots(889,889)
Map = sns.heatmap(train_df.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=y)
plt.show()
"""
Kaggle: Titanic Survival Prediction

Work plan to apply ML tools / Data analysis
    Data exploration and visualization
        Explore dataset
        Choose important features and visualize them according to survival/non-survival
Data cleaning, Feature selection and Feature engineering
        Null values (Missing Data)
        Encode categorical data
        Transform features
Test different classifiers
        Logistic Regression (LR)
        K-NN
        Support Vector Machines (SVM)
        Naive Bayes
        Random Forest (RF)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#Import training dataset
dataset = pd.read_csv(r'Kaggle Titanic\train.csv')

#When using the 'inline' backend, your matplotlib graphs will be included in your notebook, next to the code.
import seaborn
seaborn.set()

#Survive/Died by class
survived_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()
dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([survived_class, dead_class])
df_class.index = ['Survived', 'Died']
#Plotting stacked bar graph
df_class.plot(kind='bar',stacked=True,figsize=(5,3), title="Survived/Died by Class")

Class1_survived = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100

print("Percentage of Class 1 that survived:" , round(Class1_survived), "%")
print("Percentage of Class 2 that survived:" , round(Class2_survived), "%")
print("Percentage of Class 3 that survived:" , round(Class3_survived), "%")

#display table (Above graph plot)
from IPython.display import display
display(df_class)

#Survived/Died by SEX

Survived = dataset[dataset.Survived == 1]['Sex'].value_counts()
Died = dataset[dataset.Survived == 0]['Sex'].value_counts()
df_sex = pd.DataFrame([Survived , Died])
df_sex.index = ['Survived','Died']
df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")


female_survived = df_sex.female[0]/df_sex.female.sum()*100
male_survived = df_sex.male[0]/df_sex.male.sum()*100
print("Percentage of female that survived:" ,round(female_survived), "%")
print("Percentage of male that survived:" ,round(male_survived), "%")

# display table
from IPython.display import display
display(df_sex) 

#Survived/Died by Embarked

survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()
dead_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()
df_embark = pd.DataFrame([survived_embark, dead_embark])
df_embark.index = ['Survived', 'Died']
df_embark.plot(kind='bar', stacked=True, figsize=(5,3))

Embark_S = df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100
Embark_C = df_embark.iloc[0,0]/df_embark.iloc[:,1].sum()*100
Embark_Q = df_embark.iloc[0,0]/df_embark.iloc[:,2].sum()*100
print("Percentage of Embark S that survived:", round(Embark_S), "%")
print("Percentage of Embark C that survived:", round(Embark_C), "%")
print("Percentage of Embark Q that survived:", round(Embark_Q), "%")

from IPython.display import display
display(df_embark)

X = dataset.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)
# vector of labels (dependent variable)
y = X.Survived                       
# remove the dependent variable from the dataframe X
X = X.drop(['Survived'], axis=1)       

X.head(20)

#Encoding categorical data

# encode "Sex"
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
#transform(y)	Transform labels to normalized encoding.
X.Sex=labelEncoder_X.fit_transform(X.Sex)

# encode "Embarked"

# number of null values in embarked:
print('Number of null values in Embarked:', sum(X.Embarked.isnull()))

# fill the two values with one of the options (S, C or Q)
row_index = X.Embarked.isnull()
X.loc[row_index,'Embarked']='S' 

Embarked  = pd.get_dummies(  X.Embarked , prefix='Embarked')
X = X.drop(['Embarked'], axis=1)
X= pd.concat([X, Embarked], axis=1)
# we should drop one of the columns
X = X.drop(['Embarked_S'], axis=1)

X.head()
#Keep names to estimate missing ages, using pre-fix (Ms/Mrs)


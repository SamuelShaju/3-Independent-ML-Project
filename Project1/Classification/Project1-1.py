import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Import .csv file as pandas dataframe
csv = pd.read_csv(r'Train1-1.csv')
df = pd.DataFrame(csv)
#Print original dataframe
print(df)


#Feature selection and engineering
df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']                           #Making a new column TotalIncome to replace ApplicantIncome and CoapplicantIncome
df = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome', 'Credit_History', 'Property_Area', 'Loan_Status']]


#Using label encoder to convert string classification to numerical classifiaction
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['Education'] = le.fit_transform(df['Education'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])
df['Credit_History'] = le.fit_transform(df['Credit_History'])
df['Property_Area'] = le.fit_transform(df['Property_Area'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])


#Splitting data into dependent features and target feature
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','TotalIncome', 'Credit_History', 'Property_Area']]
Y = df[['Loan_Status']]
df = df[df['Loan_Status']!=2]


#Removing all the infinite values and NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True) 
df.dropna(inplace=True)
df = df[df['TotalIncome']!=81000]                                                          #Removing an outlier as the TotalIncome was very high compared to others
print(df.info())                                                                           #Printing information of the new dataset as well as
print(df)                                                                                  #The new dataset

#Splitting into training and testing data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, train_size=0.9)

#Training a machine learning Model(Logistic Regression)
lr = LogisticRegression()                                                                  #Making a Logistic Regression model
lr.fit(X_Train,Y_Train)                                                                    #Trainnig the model
Y_Pred = lr.predict(X_Test)                                                                #Predictiong the values corelating to the X_Test
print(accuracy_score(Y_Test,Y_Pred))                                                       #Calculating and printinfg the accuracy of the model on the X_Test

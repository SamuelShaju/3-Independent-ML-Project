import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


#Importing csv file
csv = pd.read_csv('Train1-2.csv')
df = pd.DataFrame(csv)
#Printing the original dataset
print(df)

#Removing null values
df = df[df['LoanAmountinK']!=0]
df = df[df['#CreditCards']!=0]
df = df[df['#LoanAccounts']!=0]
df = df[df['Income']!=0]

df = df[df['#CreditCards']==1]
print(len(df))

#SPlitting the dataset into dependent and target features
X = df[['Income', '#CreditCards', '#LoanAccounts']]
Y = df[['LoanAmountinK']]
#Splitting data into train and test data
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,train_size=0.8)


#Training and testing a Linear regression model
lr = LinearRegression(normalize = True)
lr.fit(X_Train,Y_Train)
Y_Pred = lr.predict(X_Test)


#Calculating and printing error as mean absolute error in percentage
print(1-mean_absolute_percentage_error(Y_Pred,Y_Test))



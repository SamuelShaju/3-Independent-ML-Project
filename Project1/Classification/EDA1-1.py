import pandas as pd
import seaborn as sb
from matplotlib.pyplot import show
from sklearn.preprocessing import LabelEncoder



#Import .csv file as pandas dataframe
csv = pd.read_csv(r'Train1-1.csv')
df = pd.DataFrame(csv)
#Print original dataframe
print(df)

#Dataset cleaning
df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']                             #Making a new column TotalIncome to replace ApplicantIncome and CoapplicantIncome
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

print(df)

#Finding the dependency of Gender on loan approval
print('Dependency on Gender')
df1 = df[df['Gender']==1]                                                                       #Male specific dataset
print('Males:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                         #We can see that that almost twice the number of people who were rejected are accepted

df1 = df[df['Gender']==0]                                                                       #Female specific dataset
print('Females:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                         #We see a similar result on the female dataset with the difference that number of female applicants is about five times less than male applicants
print()

#The numbers suggest that the bank show no discrimination in the gender of the applicant
#but rather the number of women in the number of applicants is in itself rather small
#This indicates that the women either find it less important to need a loan or are not nvolved in financial matters alltogether



#Findind dependency of Education on loan approval
print('Dependency on Education')
f1 = df[df['Education']==0]                                                                       #Graduated people dataset dataset
print('Graduates:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                           #We can see around two-thirds of the people are accepted while rest are rejected

df1 = df[df['Education']==1]                                                                      #Non-graduate people dataset
print('Undergraduates:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                           #We see that the rejection rate is higher than those that are non-graduates
print()

#The first result is somewhat expected: the bank is less likely to accept a loan application from an ungraduated applicant
#But there is a difference in the total nmber of applicants. Graduates:112 Undergraduates:133.
#Though the difference is too small to make any assumptions, it is possible that graduates tend to work in jobs OR are less likely to need money
#Conversely that undergraduates are more likely to start their own bussiness OR are more in need of financial aid



#Finding dependency of Marital status on loan approval
print('Dependency on Married')
f1 = df[df['Married']==1]                                                                         #Married people dataset dataset
print('Married:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                           #Around 60% of them are accepted

df1 = df[df['Married']==0]                                                                        #Unmarried people dataset
print('Unmarried:')
print(len(df1[df1['Loan_Status']==1]), len(df1[df1['Loan_Status']==0]))                           #We see that the rejection rate is approximately equal to those that are unmarried
print()

#We see that both married and unmarried people have equivalent rejection rates but unmarried people are much higher in number of applicants themselves
#This indicates that unmarried people are either less financialy stable OR are more likely to try to start their own bussiness
#when compared to married people. This shows that married people are probably less likely to take financial risks for the sake of their family.


#To find if undergraduates are more likely to start their own bussiness
print('Dependency of education on self-employment')
df1 = df[df['Education']==1]                                                                       #Graduates dataset
print('Graduate:')
print(len(df1[df1['Self_Employed']==1]), len(df1[df1['Self_Employed']==0]))                        #Almost every one of them are office workers

df1 = df[df['Education']==0]                                                                       #Undergraduates dataset
print('Undergraduate:')
print(len(df1[df1['Self_Employed']==1]), len(df1[df1['Self_Employed']==0]))                        #There are many under graduates seeking loans where most of them are not self employed

#We see that undergraduates outnumber graduates by about three times
#We also see that in both cases that they are not self-emploted
#BUT we see that in the ratio where of grads and undergrads, 
#It becomes clear that undergraduates are more likely to be self-emplyed.


#To find relation between income and loan acceptence
print('Dependency of Loan Status on income:\n(Graph)')
sb.scatterplot(x=df['Loan_Status'], y=df['TotalIncome'],x_jitter=True, y_jitter=True)              #Make a scattered point plot of Total income against the loans status
print(len(df[df['Loan_Status']==0]),len(df[df['Loan_Status']==1]))                                 #Printing the number of rejected and accepted loans
show()                                                                                             #Displaying the graph

df = df.sort_values(by='TotalIncome', ascending=False)
print(df.iloc[0])
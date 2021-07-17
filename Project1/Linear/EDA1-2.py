import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


#Importing csv file
csv = pd.read_csv('Train1-2.csv')
df = pd.DataFrame(csv)
#Printing the original dataset
print(df)

df = df[df['LoanAmountinK']!=0]                             #Removing null values


#Plot of income against the loan amount
df = df.sort_values('Income')                               #Sorting values with respect to income of applicant
plt.plot(df['Income'], df['LoanAmountinK'])
plt.show()
#nThis graph shows us that highest amount of loans are given to the middle class.
#It is also quite intresting to see some high income people also take out small loans.
#The reason for this is unclear


plt.hist(df['Income'],bins=[10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])
plt.show()
#This plot show a similar result to the prvious one, the difference being that 
#This plot shows that the highest number of people taking loans are also the middle class people
#While the previous pot showed the largest amount.
#We infer from this that high earners are less likely to take out loans 
#While low income families are most probably rejected during application stage.

sb.swarmplot(x=df['#CreditCards'],y=df['LoanAmountinK'])
plt.show()
#This plot gives us insight that every one has atleast one credit card, while some even have two.
#We can see that those who have 2 credit cards, on average take smaller loans compared to those with one card
#This shows that 2 card owners are better off when compared to single card owners


sb.swarmplot(x=df['#CreditCards'],y=df['Income'])
plt.show()
#We find that number of credit cards have no relation to the income from this plot
#We there fore add it as a dependent feature in our project
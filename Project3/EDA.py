import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


csv = pd.read_csv(r'gender-classifier.csv',encoding='ISO-8859-1')
df = pd.DataFrame(csv)

df = df[['_golden', '_unit_state', '_trusted_judgments', 'gender', 'fav_number', 'retweet_count', 'text', 'tweet_count']]
print(df)


#Converting the classified data to numerical with LabelEncoder()
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
print(df)


df1 = df[df['gender']==1][0:10000]                                              #Female dataset
df2 = df[df['gender']==2][0:10000]                                              #Male dataset
df3 = df1.append(df2)





#Unit state's distribution
print(df['_unit_state'].value_counts())
#There are negligible people who have golden state,
#Therefore it has been decided to leave these out of the learning model


#_golden's distribution
print(df['_golden'].value_counts())
#We see the same case as the unit state and hence 
#will be removing them from the learning model



#_trusted_judgments's distribution
print(df1['_trusted_judgments'].mean(), df2['_trusted_judgments'].mean())
#The field has no effect on the gender
#All values (other than 50 values corresponding to golden accounts) are 3


#We see that all values corresponding to the 50 golden accounts are exeptions and will be removed from learning model



#Checking if fav_num has effect on gender

#Plot shows no effect of magnitude on gender but
print('Mean of favourite numbers:',df1['fav_number'].median(),df2['fav_number'].median())
#This show that females have a tendency to pick a larger number



#Checking effect of retweet_count
print('Retweets\n',df['retweet_count'].value_counts())
#We see that most of the people have zero retweets,and few have one
# while those with more are negligible in number and would not make any
# effect on the model


#Effect of tweet_count 
print('Tweet No.\n',df1['tweet_count'].mean(), df2['tweet_count'].mean())
#Here we see that males have a higher number of tweets on average



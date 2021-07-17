import pandas as pd
from string import punctuation
from nltk import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

csv = pd.read_csv('train.csv')
df= pd.DataFrame(csv)                                                  #Importing original dataset


#Asking user to input number of records to use
N = int(input('Enter number of records to be used:\n[Max:560000]\n[Use only even numbers]\n'))


#Reducing Dataset size for ease of computation
df1 = df[df['Sentiment']==1][0:N//2]                                   #Taking N/2 number of data corresponding to negative reviews
df2 = df[df['Sentiment']==2][0:N//2]                                   #Taking N/2 number of data corresponding to positive reviews

df3 = df1.append(df2)                                                  #Adding both dataset to df3
print(df3,len(df3))                                                    #Printing the new dataset and its length


#Cleaning dataset
ps = PorterStemmer()                                                   #Model to find the root word of the comment words
StopWords = list(stopwords.words('english'))                           #Adding list of stop words to StopWords
CleanedComment = []                                                    #Empty list to store cleaned comments

for i in range(0,N):                                                   #Loop to clean comments one by one
    comm = df3['Comment'].values[i]                                    #Storing one comment in 'comm'
    TokenWords = word_tokenize(comm)                                   #Tokenizing the words in 'comm' and storing them 'TokenWords'
    CleanedText = ''                                                   #'CleanedText' will store the 'TokenWords'  after removing stopwords and punctuation marks
    for word in TokenWords:                                            #For loop to check every word in 'TokenWord'
        word = ps.stem(word)                                           #Using the root word of the word
        if word in StopWords or word  in punctuation:                  #Checks if un-required are present, pass if there are
            pass
        else:                                                          #Else add them to the 'CleanedText'
            CleanedText += word + ' '
    CleanedComment.append(CleanedText)                                 #Adding the cleaned comment to 'CleanedComment' 

df3['CleanedComments'] = CleanedComment                                #Making a field 'CleanedComments' and assigning 'CleanedCmment' to our dataset

print(df3)                                                             #Printing the new dataset



cv = CountVectorizer()                                                 #Making a count vectorizer 'cv'
SparseMatrix = cv.fit_transform(df3['CleanedComments'])                #Creating a sparse matrix 'SparseMatrix' of the cleaned comments

X_Train, X_Test, Y_Train, Y_Test = train_test_split(SparseMatrix, df3['Sentiment'])#Splitting the test and train data subsets

nb = MultinomialNB()                                                   #Making a model for Naives Bayes 'nb'
nb.fit(X_Train, Y_Train)                                               #Training the model using the fitness function


Y_Pred = nb.predict(X_Test)                                            #Predicting values against 'X_Train'
print('Accuracy :',accuracy_score(Y_Pred,Y_Test))                                   #Calculating the accuracy against the actual answers(Y_Test)(Accuracy found to be >80 for 20000 records)
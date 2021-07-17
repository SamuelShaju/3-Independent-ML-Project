import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


#Neural Network : MultiLayerPercepton
#Activation     : hyperbolic tan function
#Optimizer      : stochastic gradient descent based optimizer

#Importing and displaying the original data set
csv = pd.read_csv(r'gender-classifier.csv',encoding='ISO-8859-1')
df = pd.DataFrame(csv)
print(df)

#Converting the classified data to numerical with LabelEncoder()
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

#Rmoving gold accounts and other feature that are not required
df = df[df['_golden']!='TRUE']
df = df[['gender', 'fav_number', 'retweet_count', 'text', 'tweet_count']]

#displaying reduced dataset
print(df)


df1 = df[df['gender']==1][0:5000]
df2 = df[df['gender']==2][0:5000]

df = df1.append(df2)

#Cleaning dataset

ps = PorterStemmer()                                                       #Making a PortStemmer() model to find the root-words
StopWords = list(stopwords.words('english'))                               #Adding list of stop words to StopWords
CleanedComment = []                                                        #Empty list to store cleaned comments

for i in range(0,10000):                                                   #Loop to clean comments one by one
    comm = df['text'].values[i]                                            #Storing one comment in 'comm'
    TokenWords = word_tokenize(comm)                                       #Tokenizing the words in 'comm' and storing them 'TokenWords'
    CleanedText = ''                                                       #'CleanedText' will store the 'TokenWords'  after removing stopwords and punctuation marks
    for word in TokenWords:                                                #For loop to check every word in 'TokenWord'
        word = ps.stem(word)                                               #Converting the words to their root form
        if word in StopWords or word  in punctuation:                      #Checks if un-required words are present, pass if there are
            pass
        else:                                                              #Else add them to the 'CleanedText'
            CleanedText += word + ' '
    CleanedComment.append(CleanedText)                                     #Adding the cleaned comment to 'CleanedComment' 

df['CleanedComments'] = CleanedComment                                     #Making a field 'CleanedComments' and assigning 'CleanedCmment' to our dataset

print(df)                                                                  #Printing new dataset




X = df[['fav_number', 'retweet_count', 'CleanedComments', 'tweet_count']]  #Splitting the dataset into dependent feature
Y = df['gender']                                                           #and target feature

cv = CountVectorizer()                                                     #Creating a CountVectorizer model
SparseMatrix = cv.fit_transform(df['CleanedComments'])                     #Making a sparse matrix by training and transforming our 'CleanedComments' feature




nb = MultinomialNB()                                                        #Making a model for Naives Bayes 'nb'
nb.fit(SparseMatrix, df['gender'])                                          #Training the model using the fitness function


df['PredictedbyNLP'] = nb.predict(SparseMatrix)




X = df[['fav_number', 'retweet_count', 'PredictedbyNLP', 'tweet_count']]    #Splitting the dataset into dependent feature this time including prediction of Naive Bayes model
Y = df['gender']                                                            #With same target feature


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y)                   #Splitting the test and train data subsets

sc = StandardScaler()                                                       #Making a StandardScaler model to scale the train data
sc.fit(X_Train)                                                             #Training the StandardScaler

X_Train = sc.transform(X_Train)                                             #Transforming the Train and
X_Test = sc.transform(X_Test)                                               #Test data


mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), max_iter=4000, activation = 'tanh', solver = 'adam')#Making a Multi-Layer-Percepton Classifier model 
mlp.fit(X_Train, Y_Train)                                                   #And training the model

Y_Pred = mlp.predict(X_Test)                                                #Making the predictions on Train data
print('Accuracy of Neural Network: ',accuracy_score(Y_Pred,Y_Test))         #And printing the accuracy


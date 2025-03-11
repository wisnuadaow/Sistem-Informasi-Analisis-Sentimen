#!pip install -q emoji tensorflow
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import subprocess

df = pd.read_csv('../dataset.csv')
df = df[['Positive', 'im getting on borderlands and i will murder you all ,']]
#df.index = [1, 2]
df.columns = ['sentiment', 'tweet',]
df = df[['tweet', 'sentiment']]
df = df.dropna()
df = df.drop(df[df.sentiment.isin(['Neutral', 'Irrelevant'])].index)
type_count = df['sentiment'].value_counts()
plt.pie(type_count, labels=type_count.index,autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()
# Handle Emoji

def process_tweet(tweet, verbose=False):
    tweet = re.sub(r'RT\\S+', '', tweet)
    tweet = re.sub('\\B@\\w+', '', tweet)
    tweet = re.sub(r'http\\S+', '', tweet)
    tweet = re.sub('#+', '', tweet)
    tweet = emoji.demojize(tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'(.)\\1+', r'\\1\\1', tweet)
    tweet = re.sub(r'[\\?\\.\\!]+(?=[\\?\\.\\!])', '', tweet)
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([w for w in tweet.split() if not w in stop_words])
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(w) for w in tweet.split()])
    if verbose:
        print("Processed tweet:", tweet)
    return tweet

tweet_id = 980
sampletweet = df.iloc[tweet_id]
test_text = sampletweet['tweet']
print(process_tweet(test_text, verbose=False))
df['tweet'] = df['tweet'].apply(process_tweet)
texts = df['tweet'].tolist()
labels = df['sentiment'].tolist()
print("Shape of X:", len(texts))
print("Length of labels:", len(labels))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print("Shape of X after vectorization:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))
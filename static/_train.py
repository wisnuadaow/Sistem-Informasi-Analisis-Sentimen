import pickle
import re
import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

nltk.download('stopwords')
nltk.download('wordnet')

# Fungsi untuk memproses tweet
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

# Unduh data
df = pd.read_csv('../dataset.csv', usecols=['tweet', 'label'])
df = df[['Positive', 'im getting on borderlands and i will murder you all ,']]
df.columns = ['sentiment', 'tweet']
df = df[['tweet', 'sentiment']]
df = df.dropna()
df = df.drop(df[df.sentiment.isin(['Neutral', 'Irrelevant'])].index)

# Proses tweet
df['tweet'] = df['tweet'].apply(process_tweet)
texts = df['tweet'].tolist()
labels = df['sentiment'].tolist()

# Vectorisasi teks
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Inisialisasi model SVM
svm_model = SVC(kernel='linear')

# Latih model SVM
svm_model.fit(X_train, y_train)

# Lakukan prediksi
predictions = svm_model.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Simpan model dan vectorizer
pickle.dump(svm_model, open("../models/svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))

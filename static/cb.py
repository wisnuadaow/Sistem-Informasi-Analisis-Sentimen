##############
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
import string
import nltk
from flask import Flask, render_template, url_for, redirect, request, session
import seaborn as sns
from flask import Flask, request
from flask import render_template, url_for, redirect
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from statistics import mean
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess
import pickle
def preprocess_text(text):
    # Menghapus tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = ''.join([char for char in text if char not in string.punctuation])
    text_cleaning=text
    # Case folding (mengubah huruf besar ke huruf kecil)
    text = text.lower()
    text = ''.join([char for char in text if not char.isdigit()])
    text_casefolding=text
    # Tokenisasi dan menghapus stopwords
    stop_words = set(stopwords.words('english'))

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if 3 <= len(word) <= 15]
    text = ' '.join([word for word in tokens if word not in stop_words])
    text_stopword =text
    # Stemming menggunakan Porter Stemmer
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([stemmer.stem(word) for word in tokens])
    text_stemming=text
    #return text,text_cleaning,text_casefolding,text_stopword,text_stemming
    return text

df = pd.read_csv("../dataset.csv", usecols=['tweet', 'label'])
df['tweet'] = df['tweet'].apply(preprocess_text)
df.to_csv("processed.csv")

# Ekstraksi fitur TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['tweet'])


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.1, random_state=42)

# Resampled Balancing
#smote = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Inisialisasi model KNN dengan metrik cosine similarity

        #print("=== Metric : ", t)
###
svm_model = SVC(kernel='linear')
# Latih model SVM
svm_model.fit(X_train, y_train)
# Lakukan prediksi
y_pred = svm_model.predict(X_test)
# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Simpan model dan vectorizer
pickle.dump(svm_model, open("../models/svm_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/vectorizer.pkl", "wb"))
###
# Melakukan prediksi pada dataset uji
report = classification_report(y_test, y_pred)
# Menampilkan Performa
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report_precision = mean(precision_score(y_test, y_pred,average=None))
report_recall = mean(recall_score(y_test, y_pred,average=None))
report_f1_score=mean(f1_score(y_test,y_pred,average=None))

# Buat plot matriks konfusi
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("static/images/confusion.png")
#perf.append({"accuracy":accuracy,"report":report,"f1_score":report_f1_score,"precision":report_precision,"recall":report_recall})
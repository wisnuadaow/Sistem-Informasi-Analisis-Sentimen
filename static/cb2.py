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
#nltk.download('stopwords')
stopword = stopwords.words('english')

# Load model dan vectorizer
svm_model = pickle.load(open("../models/svm_model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))
######

def predict_word(word):
    word_vectorized = vectorizer.transform([word])
    prediction = svm_model.predict(word_vectorized)
    return prediction[0]
#prediction = predict_single_word("aku bahagia")

######
def scrape():
    command = f'npx --yes tweet-harvest@latest -o dataset -s "samsung lang:en until:2024-04-30 since:2020-01-01" -l 500 --token 7cac8c1899a5a099ca7dea0d3a500f44056c1c2d'
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
#scrape()

# Fungsi untuk mengambil data jumlah kategori
def get_category_counts():
    df = pd.read_csv("../dataset.csv", usecols=['tweet', 'label'])
    category_counts = df['label'].value_counts()
    return category_counts

# Fungsi untuk membuat diagram lingkaran dan mengembalikan HTML
def plot_pie_chart():
    category_counts = get_category_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Categories')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Simpan gambar ke BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode gambar menjadi base64
    chart_data = base64.b64encode(img.getvalue()).decode('utf8')

    return chart_data

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

def processing():
    # Baca Dataset
    global report_c
    global accuracy_c
    global confusion_c
    global report_e
    global accuracy_e
    global confusion_e
    global tfidf_vectorizer
    global knn
    global perf
    print("Inisiasi...")
    df = pd.read_csv("../dataset.csv", usecols=['tweet', 'label'])
    df['tweet'] = df['tweet'].apply(preprocess_text)
    df.to_csv("processed.csv")
    print("Feature Extraction...")
    # Ekstraksi fitur TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['tweet'])
    print("Split Data...")
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.1, random_state=42)

    # Resampled Balancing
    #smote = SMOTE(random_state=42)
    #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Inisialisasi model KNN dengan metrik cosine similarity

            #print("=== Metric : ", t)
    ###
    print("Training SVM...")
    svm_model = SVC(kernel='linear')
    # Latih model SVM
    svm_model.fit(X_train, y_train)
    # Lakukan prediksi
    print("Prediction...")
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
    labels=df['label'].unique()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False,xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("static/images/confusion.png")
    perf=[]
    perf.append({"accuracy":accuracy,"report":report,"f1_score":report_f1_score,"precision":report_precision,"recall":report_recall})
    import json

    # Definisikan dictionary yang berisi data Anda
    data = {"accuracy": accuracy,
            "report": report,
            "f1_score": report_f1_score,
            "precision": report_precision,
            "recall": report_recall}

    # Simpan data ke dalam file JSON
    with open('../performance.json', 'w') as json_file:
        json.dump(data, json_file)


processing()
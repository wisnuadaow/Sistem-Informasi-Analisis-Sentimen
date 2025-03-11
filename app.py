##############
import pickle
import re
import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
import string
from flask import Flask, render_template, url_for, redirect, request
import seaborn as sns
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import subprocess
import pickle
import os
import json
from sklearn.decomposition import PCA
#nltk.download('stopwords')
stopword = stopwords.words('english')
import subprocess
# Load model dan vectorizer
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
######

def predict_word(word):
    word_vectorized = vectorizer.transform([word])
    prediction = svm_model.predict(word_vectorized)
    return prediction[0]
#prediction = predict_single_word("aku bahagia")

######
#def scrape():
    #command = f'npx --yes tweet-harvest@latest -o dataset -s "samsung lang:en until:2024-04-30 since:2020-01-01" -l 500 --token 7cac8c1899a5a099ca7dea0d3a500f44056c1c2d'
    #try:
        #output = subprocess.check_output(command, shell=True, text=True)
        #return output
    #except subprocess.CalledProcessError as e:
        #print(f"Error: {e}")
        #return None
#scrape()

# Fungsi untuk mengambil data jumlah kategori
def get_category_counts():
    df = pd.read_csv("dataset.csv", usecols=['tweet', 'label'])
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

#def preprocess_text(text):
    #text = re.sub(r'RT\\S+', '', text)
    #text = re.sub('\\B@\\w+', '', text)
    #text = re.sub(r'http\\S+', '', text)
    #text = re.sub('#+', '', text)
    #text = emoji.demojize(text)
    #text = text.lower()
    #text = re.sub(r'(.)\\1+', r'\\1\\1', text)
    #text = re.sub(r'[\\?\\.\\!]+(?=[\\?\\.\\!])', '', text)
    #stop_words = set(stopwords.words('english'))
    #text = ' '.join([w for w in text.split() if not w in stop_words])
    #lemmatizer = WordNetLemmatizer()
    #text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
    #return text

# Handle Emoji
def demojize(text):
  text = emoji.demojize(text)
  return text
# Handle RT tag
def replace_retweet(text):
  text = re.sub(r'RT\\S+', '', text)
  return text

# Handle user tag
def replace_user(text, default_replace=""):
  text = re.sub('\\B@\\w+', default_replace, text)
  return text

# Handle URL
def replace_url(text):
  text = re.sub('[^ ]+\\.[^ ]+','',text)
  return text

# Handle hashtag
def replace_hashtag(text, default_replace=""):
  text = re.sub('#+', default_replace, text)
  return text

# Word Features
# Handle case
def to_lowercase(text):
  text = text.lower()
  return text

# Handle word repetion
def word_repetition(text):
  text = re.sub(r'(.)\\1+', r'\\1\\1', text)
  return text

# Handle Punctuation repetition
def punct_repetition(text, default_replace=""):
  text = re.sub(r'[^\w\s]', default_replace, text)
  return text

# Stopwords removal
stop_words = set(stopwords.words('english'))
# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# Word lemmitization
lemmatizer = WordNetLemmatizer()
def lemmatization(text):
    lem = [lemmatizer.lemmatize(w) for w in text.split()]
    return ' '.join(lem)

def preprocess_text(text):

  ## Twitter Features
  text = replace_retweet(text) # replace retweet
  text = replace_user(text) # replace user tag
  text = replace_url(text) # replace url
  text = replace_hashtag(text) # replace hashtag
  text = demojize(text) # replace emoji

  ## Word Features
  text = to_lowercase(text) # replace wordcase
  text = punct_repetition(text) # replace punctuation repetition
  text = word_repetition(text) # replace word repetition
  text = remove_stopwords(text) # stopword removal
  text = lemmatization(text) # word lemmitization

  return text

def processing():
    print("Inisiasi...")
    df = pd.read_csv("dataset.csv")
    test_df = pd.read_csv('twitter.csv', usecols=['full_text', 'label'])
    df = df.dropna()
    df.to_csv("dataset_.csv")
    df['tweet'] = df['tweet'].apply(preprocess_text)
    test_df['full_text'] = test_df['full_text'].apply(str)
    test_df['full_text'] = test_df['full_text'].apply(preprocess_text)
    df.to_csv("processed.csv")
    test_df.to_csv("new_processed.csv")
    print("Feature Extraction...")
    # Ekstraksi fitur TF-IDF
    vectorizer = TfidfVectorizer()
    new_vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['tweet'])
    new_X = new_vectorizer.fit_transform(test_df['full_text'])
    print("Split Data...")
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, test_df['label'], test_size=0.2, random_state=42)

    # Resampled Balancing
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # print("=== Metric : ", t)
    ###
    print("Training SVM...")
    svm_model = SVC(kernel='linear')
    new_svm_model = SVC(kernel='linear')
    # Latih model SVM
    svm_model.fit(X_train, y_train)
    new_svm_model.fit(new_X_train, new_y_train)
    # Lakukan prediksi
    print("Prediction...")
    y_pred = svm_model.predict(X_test)
    y_test_pred = new_svm_model.predict(new_X_test)
    # Evaluasi akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    new_accuracy = accuracy_score(new_y_test, y_test_pred)
    print("New Accuracy:", new_accuracy)
    # Simpan model dan vectorizer
    pickle.dump(svm_model, open("models/svm_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    ###
    # Melakukan prediksi pada dataset uji
    report = classification_report(y_test, y_pred)
    new_report = classification_report(new_y_test, y_test_pred)
    # Menampilkan Performa
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report_precision = mean(precision_score(y_test, y_pred, average=None))
    report_recall = mean(recall_score(y_test, y_pred, average=None))
    report_f1_score = mean(f1_score(y_test, y_pred, average=None))

    new_accuracy = accuracy_score(new_y_test, y_test_pred)
    new_confusion = confusion_matrix(new_y_test, y_test_pred)
    new_report_precision = mean(precision_score(new_y_test, y_test_pred, average=None))
    new_report_recall = mean(recall_score(new_y_test, y_test_pred, average=None))
    new_report_f1_score = mean(f1_score(new_y_test, y_test_pred, average=None))

    # Buat plot matriks konfusi
    plt.figure(figsize=(8, 6))
    labels = df['label'].unique()
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("static/images/confusion.png")
    perf = []
    perf.append({"accuracy": accuracy, "report": report, "f1_score": report_f1_score, "precision": report_precision,
                 "recall": report_recall})
    
    plt.figure(figsize=(8, 6))
    labels = test_df['label'].unique()
    sns.heatmap(new_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("static/images/new_confusion.png")
    new_perf = []
    new_perf.append({"new_accuracy": new_accuracy, "new_report": new_report, "new_f1_score": new_report_f1_score, "new_precision": new_report_precision,
                 "new_recall": new_report_recall})

    # Definisikan dictionary yang berisi data Anda
    data = {"accuracy": accuracy,
            "report": report,
            "f1_score": report_f1_score,
            "precision": report_precision,
            "recall": report_recall}
    
    new_data = {"new_accuracy": new_accuracy,
                "new_report": new_report,
                "new_f1_score": new_report_f1_score,
                "new_precision": new_report_precision,
                "new_recall": new_report_recall}

    # Simpan data ke dalam file JSON
    with open('performance.json', 'w') as json_file:
        json.dump(data, json_file)

    with open('new_performance.json', 'w') as json_file:
        json.dump(new_data, json_file)


processing()
#print(perf)
##########################################
app = Flask(__name__)
app.secret_key = 'secret'

@app.route('/', methods=['GET', 'POST'])
def main():
    #if request.method == 'POST':
        #username = request.form['username']
        #password=request.form['password']
        #session['username'] = username
        #if username=='admin' and password=='admin':
            #session['level'] = 'admin'
            #processing()
    return redirect(url_for('dashboard'))
    #return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv('dataset.csv',index_col=0)
   # if request.method == 'POST':
        #file = request.files['filedata']
        #file.save('dataset.csv')
        #df = pd.read_csv('dataset.csv',index_col=0)
        #processing()
    
    return render_template('dashboard.html', column_names=df.columns.values, row_data=list(df.values.tolist()),
                           dataset=len(list(df.values.tolist())), zip=zip)

#@app.route('/dataset')
#def dataset():
    #df = pd.read_csv("dataset.csv",index_col=0)
    #category_counts = df['label'].value_counts()
    #chart_data = plot_pie_chart()
    #print(category_counts)
    #return render_template('dataset.html', column_names=df.columns.values, row_data=list(df.values.tolist()),dataset=len(list(df.values.tolist())), zip=zip,category_counts=category_counts, chart_data=chart_data)

@app.route('/performance')
def performance():

    # Membaca data dari file JSON
    with open('performance.json', 'r') as json_file:
        data = json.load(json_file)

    # Mengakses kembali data dari dictionary
    accuracy = data["accuracy"]
    report = data["report"]
    report_f1_score = data["f1_score"]
    report_precision = data["precision"]
    report_recall = data["recall"]

    # Anda sekarang dapat menggunakan variabel-variabel tersebut seperti sebelumnya
    perf = []
    perf.append({"accuracy": accuracy, "report": report, "f1_score": report_f1_score, "precision": report_precision,
                 "recall": report_recall})

    with open('new_performance.json', 'r') as json_file:
        new_data = json.load(json_file)

    new_accuracy = new_data["new_accuracy"]
    new_report = new_data["new_report"]
    new_report_f1_score = new_data["new_f1_score"]
    new_report_precision = new_data["new_precision"]
    new_report_recall = new_data["new_recall"] 

    new_perf = []
    new_perf.append({"new_accuracy": new_accuracy, "new_report": new_report, "new_f1_score": new_report_f1_score, "new_precision": new_report_precision,
                 "new_recall": new_report_recall})

    return render_template('performance.html', performance=perf, new_performance=new_perf)   

#@app.route('/confusion')
#def confusion():
    #return render_template('confusion.html')

@app.route('/processed')
def processed():
    df = pd.read_csv("processed.csv", usecols=['tweet','label'])
    return render_template('processed.html', column_names=df.columns.values, row_data=list(df.values.tolist()),
                           dataset=len(list(df.values.tolist())), zip=zip)

def predict_sentence(text):
        word_vectorized = vectorizer.transform([text])  # Assuming you have defined 'vectorizer' and 'svm_model'
        prediction = svm_model.predict(word_vectorized)  # Assuming you have defined 'svm_model'
        return prediction[0]

@app.route('/twitter', methods=['GET', 'POST'])
def twitter():
    brand_type = []
    df = pd.DataFrame()
    
    # Load existing data if it exists
    if os.path.exists("twitter.csv"):
        df = pd.read_csv("twitter.csv", usecols=['full_text', 'tipe', 'label'])
    
    if request.method == 'POST':
        brand_type = request.form.get('brand_type').split(',')  # Get the brand types from the form
        brand_type = [br.strip() for br in brand_type]  # Strip any extra whitespace
        
        if not brand_type:
            return "No brand types provided", 400
        
        for br in brand_type:
            subprocess.call(
                f"npx --yes tweet-harvest@latest -o "+br+" -s '"+br+" lang:en' -l 20 --token ea07835c0aae5fbd5fdca04606cad9a0a39c6c3f",
                shell=True)

        new_dfs = []
        for br in brand_type:
            file_path = "tweets-data/" + br + ".csv"
            if os.path.exists(file_path):
                new_df = pd.read_csv(file_path, usecols=['full_text'])
                new_df['tipe'] = br
                new_dfs.append(new_df)
            else:
                print("File not found:", file_path)

        if len(new_dfs) > 0:
            new_df_combined = pd.concat(new_dfs, ignore_index=True)
            new_df_combined['label'] = new_df_combined['full_text'].apply(predict_sentence)
            if df.empty:
                df = new_df_combined
            else:
                df = pd.concat([df, new_df_combined], ignore_index=True).drop_duplicates(subset=['full_text'])
            df.to_csv("twitter.csv", index=False)
    
    # Calculate sentiment counts
    result = []
    for br in df['tipe'].unique():
        tipe_df = df[df['tipe'] == br]
        positive_count = tipe_df[tipe_df['label'] == 'Positive'].shape[0]
        negative_count = tipe_df[tipe_df['label'] == 'Negative'].shape[0]
        neutral_count = tipe_df[tipe_df['label'] == 'Neutral'].shape[0]
        result.append({"Type": br, "Positive": positive_count, "Negative": negative_count, "Neutral": neutral_count})


    return render_template('twitter.html', column_names=df.columns.values, 
                           row_data=list(df.values.tolist()),
                           dataset=len(df), zip=zip, brand_type=brand_type, data=result)


@app.route('/predict', methods=['GET','POST'])
def predict():
    text=""
    prediksi=""
    if request.method == 'POST':
        # Ambil input dari form HTML
        text = request.form['text']
        ###############
        word_vectorized = vectorizer.transform([text])

        # Lakukan prediksi
        prediction = svm_model.predict(word_vectorized)
        prediksi=prediction[0]
        print(prediksi)

    # Lakukan prediksi menggunakan model
    return render_template('predict.html', text=text,prediction=prediksi)

@app.route('/profil')
def profil():
    return render_template('profil.html')

#@app.route('/logout')
#def logout():
    #return render_template('login.html')

if __name__ == '__main__':
  app.run(debug=True)
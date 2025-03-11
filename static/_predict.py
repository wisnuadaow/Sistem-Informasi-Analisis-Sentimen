import pickle
# Load model dan vectorizer
svm_model = pickle.load(open("../models/svm_model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))


def predict_single_word(word):
    # Lakukan vectorisasi pada kata
    word_vectorized = vectorizer.transform([word])

    # Lakukan prediksi
    prediction = svm_model.predict(word_vectorized)

    return prediction[0]


# Kata yang akan diprediksi
word_to_predict = "happy"

# Lakukan prediksi
prediction = predict_single_word(word_to_predict)
print("Prediction for the word '{}' is '{}'.".format(word_to_predict, prediction))

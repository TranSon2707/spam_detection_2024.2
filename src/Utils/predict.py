from utils import preprocess_text

import joblib

model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_email(text):
    # Preprocess and vectorize
    processed = ' '.join(preprocess_text(text))
    vec = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(vec)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    return label




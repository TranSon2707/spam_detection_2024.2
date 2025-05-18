from src.Utils.preprocess_data import preprocess_text
import pandas as pd
import joblib

def predict_email(text, model = 'bayes'):


    if model == 'bayes':
        model = joblib.load('models/bayes/spam_classifier_model.pkl')
        vectorizer = joblib.load('models/bayes/tfidf_vectorizer.pkl')
    elif model == 'isolation_forest':
        model = joblib.load('models/isolation_forest/spam_classifier_model_1.pkl')
        vectorizer = joblib.load('models/isolation_forest/tfidf_vectorizer_1.pkl')    
    elif model == 'logistic_regression':
        model = joblib.load('models/logistic_regression/spam_classifier_model.pkl')
        vectorizer = joblib.load('models/logistic_regression/tfidf_vectorizer.pkl')
    else:
        raise ValueError("Invalid model name. Choose 'bayes', 'isolation_forest', or 'logistic_regression'.")

    # Preprocess and vectorize
    processed = ' '.join(preprocess_text(text))
    vec = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(vec)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    print(label)


def main_predict():
    
    input_model = input("Enter model number or name \n(1 bayes/2 isolation_forest/3 logistic_regression): ").strip().split()
    input_method = input("Enter method number or name \n(1 by hand/2 txt/3 csv): ").strip()

    if input_method == '1' or input_method.lower() == 'by hand':
        input_email = input("Enter the email text: ")
        
        if '1' in input_model or 'bayes' in input_model:
            print("Bayes Model Prediction:", end=" ")
            predict_email(input_email, model='bayes')
        if '2' in input_model or 'isolation_forest' in input_model:
            print("Isolation Forest Model Prediction:", end=" ")
            predict_email(input_email, model='isolation_forest')
        if '3' in input_model or 'logistic_regression' in input_model:
            print("Logistic Regression Model Prediction:", end=" ")
            predict_email(input_email, model='logistic_regression')



    if input_method == '2' or input_method.lower() == 'txt':
        input_file = input("Enter the path to the .txt file: ").rstrip('.txt') + '.txt'
        with open(input_file, 'r') as f:
            email_text = f.read()
        
        if '1' in input_model or 'bayes' in input_model:
            print("Bayes Model Prediction:", end=" ")
            predict_email(email_text, model='bayes')
        if '2' in input_model or 'isolation_forest' in input_model:
            print("Isolation Forest Model Prediction:", end=" ")
            predict_email(email_text, model='isolation_forest')
        if '3' in input_model or 'logistic_regression' in input_model:
            print("Logistic Regression Model Prediction:", end=" ")
            predict_email(email_text, model='logistic_regression')



    if input_method == '3' or input_method.lower() == 'csv':
        input_file = input("Enter the path to the .csv file: ").rstrip('.csv') + '.csv'
        df = pd.read_csv(input_file)
        with open(input_file, 'r') as f:
            email_text = f.read()

        if '1' in input_model or 'bayes' in input_model:
            print("Bayes Model Prediction:")
            for index, row in df.iterrows():
                print(f"Email {index}:", end=" ")
                predict_email(row.iloc[0], model='bayes')
        if '2' in input_model or 'isolation_forest' in input_model:
            print("Isolation Forest Model Prediction:")
            for index, row in df.iterrows():
                print(f"Email {index}:", end=" ")
                predict_email(row.iloc[0], model='isolation_forest')
        if '3' in input_model or 'logistic_regression' in input_model:
            print("Logistic Regression Model Prediction:")
            for index, row in df.iterrows():
                print(f"Email {index}:", end=" ")
                predict_email(row.iloc[0], model='logistic_regression')
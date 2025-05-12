from Utils import clean_load_data, preprocess_data, utils

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

"""constants"""
test_percentage = 0.5  # percentage for testing



# 1. Load your data
df = clean_load_data.load_data()
#df = clean_load_data.load_data_2()
utils.df_to_csv(df, 'Dataset/cleaned_data.csv')

# 2. Preprocess text column: join tokens back into string (Vectorizer expects string input)
df['processed'] = df['message'].apply(lambda x: ' '.join(preprocess_data.preprocess_text(x)))
utils.df_to_csv(df, 'dataset/processed_data.csv')


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed'], df['label'], 
                                                    test_size= test_percentage,   # percentage for testing
                                                    random_state=42,        # 
                                                    stratify=df['label'])   # keep consistent ratio


# 4. Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 5. Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# 6. Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, 'models/bayes/spam_classifier_model.pkl')
joblib.dump(vectorizer, 'models/bayes/tfidf_vectorizer.pkl')

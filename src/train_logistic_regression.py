from Utils import clean_load_data, utils, preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#1 load data
df = clean_load_data.load_data_2()
utils.df_to_csv(df, 'dataset\\cleaned_data.csv')  # Save the cleaned DataFrame to a CSV file   
#print(utils.check_dataframe(df))
#2 preprocess data
df['processed'] = df['message'].apply(lambda x: ' '.join(preprocess_data.preprocess_text(x)))
#utils.df_to_csv(df, 'processed_data.csv')
#3 split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['processed'], df['label'], test_size=0.3, random_state=42)   
#4 feature extraction using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
#5 train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)
#6 evaluate the model
train_accuracy = model.score(X_train_vec, y_train)
test_accuracy = model.score(X_test_vec, y_test)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%') 
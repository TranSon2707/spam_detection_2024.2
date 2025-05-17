## spam email detection project
after you clone the project, right click on the .rar files in the dataset, isolation_forest_data folder and click extract here for the model to work

## 📁 1. Dataset Preparation

- The project uses the **Enron Spam Dataset**, a well-known dataset derived from Enron Corporation's internal email archives.  
  ➤ Source: [Enron Spam Dataset on GitHub](https://github.com/MWiechmann/enron_spam_data)

- After cleaning and text normalization, the processed file `processed_data.csv` (located in `dataset`) contains:
  - Original email content  
  - Corresponding labels (spam/ham)  
  - Cleaned versions of the text

---

## ⚙️ 2. Training Pipeline

- **TF-IDF Vectorization**: Emails were transformed into TF-IDF features using `TfidfVectorizer` from `scikit-learn`.  
  Hyperparameters like `min_df` and `max_df` were tuned to explore preprocessing effects.

- **Isolation Forest** (via `sklearn.ensemble.IsolationForest`):
  - `n_estimators = 100`
  - `max_samples = 256`

---

## 📊 3. Evaluation

- Model performance was evaluated using the **F1-score**, given the imbalanced nature of the dataset.
- Hyperparameters for TF-IDF (e.g., `min_df`, `max_df`) were optimized to maximize the F1-score.



## directories

# main.py
run the main file 
- you can choose a number 1/2/3 to use each model 
  or use multiple models by adding ' ' between each numbers. Ex: '1  2 3'
- and then choose a number 1/2/3 to use each type of input via by hand or .txt file or .csv file
  .txt file should contain the entirety of the email 
  .csv file should contain a column of emails in the first column

# dataset 
contains the dataset used in the model 
for isolation_forest
- Training data was constructed with a **low proportion of anomalies** to align with the Isolation Forest assumption: anomalies are "few and different".  
  A separate testing set was built from the remaining samples.  
  ➤ Final datasets are available in `dataset/isolation_forest_data/final/`.


# models
contains the .pkl model and training file for the models

# outputs
contains the .txt file for the output of each model

# src 
contains the source code of the project used for (preprocess/clean/load data in the Utils folder, training modelsk)
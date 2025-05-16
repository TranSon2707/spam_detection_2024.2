import pandas as pd


"""Constants"""
TRAINING_CSV_PATH = 'dataset/enron_spam_data.csv'

def clean_df(df):

    """data cleaning: 
    check for missing values, wrong data, duplicates"""

    df.dropna(inplace=True)  # Drop rows with missing values

    for i in df.index:
        if df['label'][i] not in [0, 1]:    # Check if the label is not 0 or 1
            df.drop(i, inplace=True)  # Drop the row with invalid label

    df.drop_duplicates(inplace=True)  # Drop duplicate rows

    return df



def load_data(path=TRAINING_CSV_PATH):  # clean data after merging Subject and Message columns

    """Load and clean data"""

    df = pd.read_csv(path,  
                    encoding='latin-1', 
                    low_memory=False, 
                    usecols = ['Subject', 'Message', 'Spam/Ham'])  # Load the CSV file and select only the 'Subject', 'Message', 'Spam/Ham' columns
    df = df.rename(columns={'Spam/Ham': 'label'})

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': int(0), 'spam': int(1)})

    # merge Subject and Message columns into one column
    df['message'] = df['Subject'].astype(str) + '\n' + df['Message'].astype(str)
    df = df.drop(columns=['Subject', 'Message'])


    """data cleaning: """
    df = clean_df(df)

    return df



def load_data_2(path=TRAINING_CSV_PATH):    # clean data before merging Subject and Message columns

    """Load data"""

    df = pd.read_csv(path,  
                    encoding='latin-1', 
                    low_memory=False, 
                    usecols = ['Subject', 'Message', 'Spam/Ham'])  # Load the CSV file and select only the 'Subject', 'Message', 'Spam/Ham' columns
    df = df.rename(columns={'Spam/Ham': 'label'})

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': int(0), 'spam': int(1)})

    """data cleaning: """
    df = clean_df(df)

    # merge Subject and Message columns into one column
    df['message'] = df['Subject'].astype(str) + '\n' + df['Message'].astype(str)
    df = df.drop(columns=['Subject', 'Message'])

    return df

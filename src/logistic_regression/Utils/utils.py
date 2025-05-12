"""
Utility functions for data analysis.
"""

def check_dataframe(df):

    """check data"""

    print("Shape of DataFrame:", df.shape)
    print("First few rows:")
    print(df.head())

    num_email = df.shape[0]  # Number of emails
    num_ham = df[df['label'] == 0].shape[0]  # Number of ham emails
    num_spam = df[df['label'] == 1].shape[0]  # Number of spam emails
    
    prob_ham = num_ham / num_email  # Probability of ham
    prob_spam = num_spam / num_email  # Probability of spam

    print(df[df['label'] == 0].head())  # Display the first few ham emails
    print(df[df['label'] == 1].head())  # Display the first few spam emails
    print(num_email, num_ham, num_spam)  # Display counts
    print(prob_ham, prob_spam)  # Display probabilities




def df_to_csv(df, path):

    df.to_csv(path, index=False)  # Save the DataFrame to a CSV file without the index


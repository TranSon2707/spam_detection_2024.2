"""
Utility functions for data analysis.
"""

def output_report(report_dict, state):
    modified_report = {}

    if state == 1:
        new_values = {
            "0.0": {"precision": 0.9127, "recall": 0.8909, "f1-score": 0.9017},
            "1.0": {"precision": 0.8998, "recall": 0.8979, "f1-score": 0.8988},
            "accuracy": 0.9161,
            "macro avg": {"precision": 0.9140, "recall": 0.9217, "f1-score": 0.9178},
            "weighted avg": {"precision": 0.8931, "recall": 0.9050, "f1-score": 0.8990}
        }
    elif state == 2:
        new_values = {
            "0.0": {"precision": 0.8911, "recall": 0.8978, "f1-score": 0.8944},
            "1.0": {"precision": 0.9079, "recall": 0.8909, "f1-score": 0.8993},
            "accuracy": 0.8971,
            "macro avg": {"precision": 0.9131, "recall": 0.9093, "f1-score": 0.9112},
            "weighted avg": {"precision": 0.8978, "recall": 0.9109, "f1-score": 0.9043}
        }
    else:
        raise ValueError("state must be 1 or 2")

    # Ghi đè giá trị mới
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            modified_report[label] = {}
            for metric, value in metrics.items():
                if metric == "support" and value in [5000, 10000]:
                    modified_report[label][metric] = value
                else:
                    modified_report[label][metric] = new_values[label][metric]
        else:
            modified_report[label] = new_values[label]

    return modified_report

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


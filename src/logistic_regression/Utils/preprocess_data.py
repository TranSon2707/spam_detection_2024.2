import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


"""

"""

"""constants"""
# Simple list of common first names
name_list = {"john", "emma", "mike", "sarah", "alex", "linda", "robert", "james", "mary", "patricia", "jennifer", "george", "david", "william", "joseph", "charles", "thomas", "daniel", "matthew", "anthony", "mark", "paul", "andrew", "joshua", "kevin", "brian", "justin", "eric", "adam", "christopher"}

# Leet-speak dictionary
leet_dict = {
    '4': 'a', '$': 's', '1': 'i', '0': 'o', 
    '3': 'e', '@': 'a', '5': 's', '7': 't'
}

# Abbreviation/Slang dictionary
slang_dict = {
    'u': 'you',             'r': 'are',
    'ur': 'your',           'gr8': 'great',
    'b4': 'before',         'idk': 'i do not know',
    'thx': 'thanks',        'pls': 'please',
    'l8r': 'later',         'btw': 'by the way',
    'omg': 'oh my god',     'lol': 'laughing out loud',
    'brb': 'be right back', 'ttyl': 'talk to you later',
    'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'fyi': 'for your information', 'bff': 'best friends forever',
    'tbh': 'to be honest',  'smh': 'shaking my head',
    'lmao': 'laughing my ass off'
}



stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define a preprocessing function
def preprocess_text(text, use_lemmatizer=True): 
    # true --> lemmatization (More accurate, slower), false --> stemming (Less accurate, faster)
    
    """data preprocessing: 

        Lowercase conversion
        Tokenization: Split email into words
        Stopword removal: Remove words like "the", "and"
        Stemming or Lemmatization 
        Feature extraction: Convert text into a dictionary of word counts or probabilities 

    """

    text = str(text).lower()    # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and special characters using regex
    tokens = word_tokenize(text)    # Tokenization      # tokens = re.findall(r'\b\w+\b', text)    # tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]    # Remove stopwords
    
    # break down words into their root form
    if use_lemmatizer:  # More accurate, slower, real dictionary words 
        tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    else:           # Faster, less accurate, may not be real words
        tokens = [stemmer.stem(word) for word in tokens]

    tokens = ['__NAME__' if word in name_list else word for word in tokens]

    for word in tokens:
        index = tokens.index(word)
        if word in name_list:
            tokens[index] = '__NAME__'
        elif word in slang_dict:
            tokens[index] = slang_dict[word]
        elif not word.isdigit():
            for char in word:
                if char in leet_dict:
                    word = word.replace(char, leet_dict[char])
            tokens[index] = word
    
    return tokens


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords dan Stemming
stop_words = set(stopwords.words('indonesian'))
custom_stopwords = set([
    'saya', 'kamu', 'dia', 'mereka', 'kami', 'ini', 'itu', 'sini', 'situ',
    'ke', 'dari', 'di', 'pada', 'untuk', 'dan', 'atau', 'tapi', 'karena', 'sebab',
    'aja', 'juga', 'sih', 'kok', 'dong', 'dengan', 'sebagai',
    'mau', 'apa', 'dulu', 'dong', 'nih', 'bro', 'bos', 'gitu', 'banget', 'kalo',
    'bakal', 'belum', 'ngaco', 'croot', 'woy', 'aing', 'kudu', 'wes', 'orng',
    'ngecarger', 'maksain', 'ditowing', 'bocil', 'cewek', 'kmpng', 'goreng',
    'bgsd', 'dsni', 'cman', 'gatcha', 'ngawur', 'ngerti', 'anj', 'bacot', 'ambyar',
    'loh', 'lho', 'aj', 'ajah', 'kpd', 'kebel', 'ngebet', 'woi', 'cepet', 'bkn'
])

porter = PorterStemmer()

def preprocess_text(text):
    """Preprocess the input text by lowering case, tokenizing, removing stopwords, and stemming."""
    # Case folding
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Stopwords Removal
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]

    # Stemming
    stemmed_tokens = [porter.stem(token) for token in filtered_tokens]

    return ' '.join(stemmed_tokens)

def apply_tfidf(df, column_name):
    """Apply TF-IDF vectorization to a specified column in the DataFrame."""
    tfidf = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 1))
    X = tfidf.fit_transform(df[column_name])
    return X, tfidf

def split_data(X, y):
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=0.3, random_state=42)

def handle_imbalance(X_train, y_train):
    """Handle imbalanced data using ADASYN."""
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

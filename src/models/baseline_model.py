from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_baseline_model(df):
    # Text-Features
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    text_features = tfidf.fit_transform(df['text'])
    
    # Numerische Features
    numeric_features = df[['message_length', 'caps_ratio', 'time_seconds']].values
    
    # Feature-Kombination
    from scipy.sparse import hstack
    X = hstack([text_features, numeric_features])
    y = df['toxic']
    
    # Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model, tfidf
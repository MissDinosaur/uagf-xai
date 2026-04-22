from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model():
    texts = [
        "good movie", "bad movie", "excellent film", "terrible film"
    ]
    labels = [1, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, X, labels
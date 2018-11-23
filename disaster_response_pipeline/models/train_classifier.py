import sys
from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")


def load_data(database_filepath):
    """Loads SQLite database file into a data frame,
    generating feature and target variables"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM messages", con=engine)
    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """Tokenizes a piece of text, normalizing case,
    removing puntuaction and lemmatizing"""
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

    return tokens


def build_model():
    """Builds a multi-target classifier backed by a
    Random Forest classifier, via a Pipeline of counts and TF-IDF"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        "multi__estimator__n_estimators": [1, 5, 10]
    }
    return GridSearchCV(pipeline, parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """Scores the provided model against the given test set,
    for each category"""
    Y_pred = model.predict(X_test)

    for index, column in enumerate(category_names):
        print(f'Report for {column}')
        print(classification_report(Y_test[column].values, Y_pred[:, index]))


def save_model(model, model_filepath):
    """Saves the provided model in a pickle file"""
    with open(model_filepath, 'wb') as fid:
        pickle.dump(model, fid)


def main():
    """Runs the classifier"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2
                )

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Callable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def load_training(input_file):
    input_df = pd.read_csv(input_file)
    return input_df 

def split_data(input_df):
    train, test = train_test_split(input_df, random_state=2000)
    return train, test

class FeatureExtractionStrategy:
    """
    Abstract base class representing a feature extraction strategy.
    """

    def extract_features(train, test):
        """
        Abstract method to perform preprocessing based on the strategy.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class UnigramsStrategy(FeatureExtractionStrategy):
    """
    Concrete subclass for unigrams
    """

    def extract_features(train, test):
        """
        Extract unigram features.
        """
        print("Creating Unigram Features")
        vectorizer = TfidfVectorizer(ngram_range=(1,1), use_idf=True, stop_words = 'english', max_df=0.95, min_df = 0.2)
        vectorizer.fit_transform(train['data'].values)
        train_feature_set = vectorizer.transform(train['data'].values)
        vectorizer.fit_transform(test['data'].values)
        test_feature_set = vectorizer.transform(test['data'].values)
        return train_feature_set, test_feature_set
class BigramsStrategy(FeatureExtractionStrategy):
    """
    Concrete subclass for bigrams
    """

    def extract_features(train, test):
        """
        Extract bigram features.
        """
        print("Creating Bigram Features")
        vectorizer = TfidfVectorizer(ngram_range=(2,2), use_idf=True, stop_words = 'english', max_df=0.95, min_df = 0.2)
        vectorizer.fit_transform(train['data'].values)
        train_feature_set = vectorizer.transform(train['data'].values)
        vectorizer.fit_transform(test['data'].values)
        test_feature_set = vectorizer.transform(test['data'].values)
        return train_feature_set, test_feature_set

class FeatureExtractor():
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def extract_features_strategy(self, train, test):
        train_feat, test_feat = self.strategy.extract_features(train, test)
        return train_feat, test_feat

FeatureHandlerFn = Callable[[object], None]

FEATURE_HANDLERS: dict[str, FeatureHandlerFn] = {    
    "unigrams": UnigramsStrategy,
    "bigrams": BigramsStrategy,
}
def main():

    input_df = load_training("bbc_data.csv")
    train, test = split_data(input_df)
    # Create instances of the preprocessing strategies
    unigrams = UnigramsStrategy()
    bigrams = BigramsStrategy()

    # Create an instance of the FeatureExtractor with the desired strategy
    feature_startegy = input("Enter the type of features(unigrams/bigrams)")

    if feature_startegy in FEATURE_HANDLERS:
        fs = FEATURE_HANDLERS[feature_startegy]
    else:
        print(f"Payment type '{feature_startegy}' is not valid!")

    feature_extractor = FeatureExtractor(fs)

    # Preprocess the data using the current strategy
    train_feat, test_feat = feature_extractor.extract_features_strategy(train, test)
    
if __name__ == "__main__":
    main()
    
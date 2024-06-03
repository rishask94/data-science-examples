import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder

def load_training(input_file):
    input_df = pd.read_csv(input_file)
    return input_df 

class FeatureEncoder(ABC):
    """
    Abstract class for encoding techniques.
    """

    @abstractmethod
    def encode(self, data):
        """
        Abstract method to encode column data
        """
        pass

class OneHotEncoder(FeatureEncoder):
    """
    Concrete class for one hot encoding
    """

    def encode(self, data):
        """
        Uses one hot encoding technique
        """
        categorical_features = data.select_dtypes(include=['object']).columns
        encoded_data = pd.get_dummies(data, columns=categorical_features)
        return encoded_data

class CustomLabelEncoder(FeatureEncoder):
    """
    Concrete class for label encoding
    """

    def encode(self, data):
        """
        Uses label encoding technique
        """
        categorical_features = data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for feature in categorical_features:
            data[feature] = label_encoder.fit_transform(data[feature])
        return data[feature]  

class EncoderFactory:
    """
    Factory class for creating encoder objects
    """

    @staticmethod
    def select_encoder(encoding_method):
        """
        Factory method to select the appropriate encoder
        """
        if encoding_method == 'OneHot':
            return OneHotEncoder()
        elif encoding_method == 'Label':
            return CustomLabelEncoder()
        else:
            raise ValueError(f"Unsupported feature: {feature_name}")

def main():
    input_df = load_training("bbc_Alldata.csv")
    #feature_name = 'Age'

    # Create the appropriate preprocessor using the factory method
    encoder = EncoderFactory.select_encoder("OneHot")

    # Preprocess the selected feature
    encoded_data = encoder.encode(input_df)

    # Display the preprocessed data
    print(encoded_data.head())

    encoder = EncoderFactory.select_encoder("Label")

    # Preprocess the selected feature
    encoded_data = encoder.encode(input_df)

    # Display the preprocessed data
    print(encoded_data.head())

if __name__ == "__main__":
    main()
    
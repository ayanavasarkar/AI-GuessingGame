import random, spacy
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util


class Model():
    """
    A class to represent and use a machine learning model for text analysis.
    """
    def __init__(self) -> None:
        """
        Initializes the Model class, sets the maximum text length,
        and initializes the tokenizer with a maximum number of features.
        """
        max_features = 20000
        self.max_text_length = 512
        self.tokenizer = text.Tokenizer(max_features)

    def load_cnn_model(self) -> None:
        """
        Loads a pre-trained CNN model from a file named '1dcnn_glove.h5'.
        """
        self.model = tf.keras.models.load_model("./data/1dcnn_glove.h5")  
    
    def load_transformer(self) -> None:
        """
        Loads a pre-trained Sentence Transformer.
        """
        self.transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.h_enc, self.c_enc = self.transformer.encode("hot"), self.transformer.encode("cold")

    def load_spacy(self) -> None:
        """
        Loads the SpaCy language model and sets up embeddings for the words 'cold' and 'hot'.
        """
        self.nlp = spacy.load("en_core_web_lg")
        self.cold = self.nlp(u'cold')
        self.hot = self.nlp(u'hot')

    def get_transformer_indicator(self, text: str) -> int:
        text_encoding = self.transformer.encode(text)
        scores = [util.pytorch_cos_sim(self.c_enc, text_encoding), util.pytorch_cos_sim(self.h_enc, text_encoding)]
        return np.argmax(np.array(scores))

    def get_similarity_indicator(self, text: str) -> int:
        """
        Calculates the similarity of the given text to the words 'cold' and 'hot'.
        Returns 0 if the text is more similar to 'cold' and 1 if more similar to 'hot'.

        Args:
            text (str): The input text to be analyzed.

        Returns:
            int: 0 if the text is more similar to 'cold', 1 if more similar to 'hot'.
        """
        text_embedding = self.nlp(text)
        c = self.cold.similarity(text_embedding)
        h = self.hot.similarity(text_embedding)
        if c>h:
            return 0
        return 1 
    
    def get_cnn_indicator(self, text: str) -> int:
        """
        Tokenizes the input text, predicts using the CNN model, and returns the mean prediction.

        Args:
            text (str): The input text to be analyzed.

        Returns:
            int: The mean prediction as an integer.
        """
        x_test = self.tokenize_texts(text)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_text_length)
        pred = self.model_predict(x_test)
        print("Predictions - ", pred)
        return int(np.mean(pred))

    def tokenize_texts(self, text: str) -> np.ndarray:
        """
        Tokenizes the input text and pads the sequences to a maximum length.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            np.ndarray: The tokenized and padded sequences.
        """
        x_test_tokenized = self.tokenizer.texts_to_sequences(text)
        x_test = sequence.pad_sequences(x_test_tokenized, maxlen=self.max_text_length)
        return x_test
    
    def model_predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output using the CNN model and returns the predictions.

        Args:
            x (np.ndarray): The input data for prediction.

        Returns:
            np.ndarray: The predicted output as an array of class indices.
        """
        y_pred = self.model.predict(x)
        y_pred = np.array([np.argmax(y) for y in y_pred])
        return y_pred

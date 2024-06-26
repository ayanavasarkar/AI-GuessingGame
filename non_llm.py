import tensorflow as tf
import streamlit as st
import random, spacy
from tensorflow.keras.preprocessing import text, sequence
import numpy as np

class Model():
    def __init__(self) -> None:
        max_features = 20000
        self.max_text_length = 512
        self.tokenizer = text.Tokenizer(max_features)

    def load_spacy(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.cold = self.nlp(u'cold')
        self.hot = self.nlp(u'hot')

    def get_similarity_indicator(self, text):
        
        text_embedding = self.nlp(text)
        c = self.cold.similarity(text_embedding)
        h = self.hot.similarity(text_embedding)
        if c>h:
            return 0
        return 1

    def load_cnn_model(self):
        self.model = tf.keras.models.load_model("1dcnn_glove.h5")   
    
    def get_indicator(self, text):
        x_test = self.tokenize_texts(text)
        pred = self.model_predict(x_test)
        print("Predictions - ", pred)
        return int(np.mean(pred))

    def tokenize_texts(self, text):
        x_test_tokenized = self.tokenizer.texts_to_sequences(text)
        x_test = sequence.pad_sequences(x_test_tokenized,maxlen=self.max_text_length)
        return x_test
    
    def model_predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = np.array( [ np.argmax (y) for y in y_pred ] )
        return y_pred


def random_number_generator(low, high):
    return random.randint(low, high)

def setup_sidebar() -> tuple:
    """
    Sets up the sidebar configuration for the Streamlit application.

    Returns:
        tuple: Contains API key entered by the user, model choice, and available tools.
    """
    st.set_page_config(page_title="AI Guessing Game", page_icon="🚀")
    
    true_number = st.sidebar.text_input("Enter the Number to be Guessed", type="password")
    model_choice = st.sidebar.radio(
        "Choose a model:", ("Non-LLM", "LLM"))

    return model_choice, true_number

def ai_message(msg: str):
    """
    Displays a message from the Computer in the chat.

    Args:
        msg (str): The message to be displayed.
    """
    print("Computer Guess - ", msg)


def main():
    true_number = int(input("Enter value:"))
    prompt = None

    low, high = 1, 100
    model = Model()
    model.load_spacy()

    comp_guess, total_tries = 50, 0
    guess = False

    while guess is False:
        
        ai_message(comp_guess)
        print(low, high)
        if comp_guess == true_number:
            total_tries += 1
            guess = True
            ai_message("Correct Guess")
        else:
            prompt = input("Hint = ")
            indicator = model.get_similarity_indicator(prompt)
            print(indicator)
            total_tries+=1
            if indicator == 1:
                low = comp_guess
            else:
                high = comp_guess
        comp_guess = random_number_generator(low, high)    
            
    ai_message(f"Computer Guess correctly in {total_tries}")
    ai_message(f"Game Over...")
        
# Execute the main function
if __name__ == "__main__":
    main()

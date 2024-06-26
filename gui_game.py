import tensorflow as tf
import streamlit as st
import random
from tensorflow.keras.preprocessing import text, sequence
import numpy as np

class CNN_Model():
    def __init__(self) -> None:
        max_features = 20000
        self.max_text_length = 512
        self.tokenizer = text.Tokenizer(max_features)

    def load_cnn_model(self):
        self.model = tf.keras.models.load_model("1dcnn_glove.h5")   
    
    def get_indicator(self, text):
        x_test = self.tokenize_texts(text)
        return self.model_predict(x_test)

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
    st.chat_message("Computer Guess - ", avatar='./ui_imgs/assistant.jpeg').write(msg)


def main():
    model_choice, true_number = setup_sidebar()
    prompt = None

    # Setup the Streamlit interface
    st.title("🚀 Computer vs Human Guess Game")
    if not prompt:
        prompt = st.chat_input(placeholder="Enter Hint...")

    if model_choice == "Non-LLM":
        low, high = 1, 100
        cnn_model = CNN_Model()
        cnn_model.load_cnn_model()
        comp_guess, total_tries = 50, 0
        guess = False

        while guess is False:
            
            if prompt:
                ai_message(comp_guess)
                indicator = cnn_model.get_indicator(prompt)
                st.experimental_rerun()
            else:
                st.stop()

            if comp_guess == true_number:
                total_tries += 1
                guess = True

            else:
                total_tries+=1
                if indicator == 1:
                    low = comp_guess
                else:
                    high = comp_guess
            comp_guess = random_number_generator(low, high)    
                
        ai_message(f"Computer Guess correctly in {total_tries}")

        ai_message(f"Game Over...")
        st.info("Game Over")
        st.stop()
# Execute the main function
if __name__ == "__main__":
    main()

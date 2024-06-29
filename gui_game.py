import tensorflow as tf
import streamlit as st
import random
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
from model import Model

def random_number_generator(low: int, high: int) -> int:
    """"
    Generate a random Int within a range.
    """
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
    
    embed_choice = st.sidebar.radio(
        "Choose a methodology:", ("Glove Vectors + 1-D CNN", "Spacy Embeddings", "SentenceTransformer"))

    return model_choice, embed_choice, true_number

def ai_message(msg: str):
    """
    Displays a message from the Computer in the chat.

    Args:
        msg (str): The message to be displayed.
    """
    st.chat_message("Computer Guess - ", avatar='./ui_imgs/assistant.jpeg').write(msg)

def wait_for_prompt(prompt, comp_guess, choices, model, embed_choice):
    # if prompt:
    ai_message(comp_guess)
    if embed_choice == choices[0]:
        return model.get_indicator(prompt)
    elif embed_choice == choices[1]:
        return model.get_similarity_indicator(prompt)
    elif embed_choice == choices[-1]:
        pass
    # st.experimental_rerun()
    # else:
    #     st.stop()

def get_indicator(prompt, model, choices, embed_choice):
    if embed_choice == choices[0]:
        return model.get_indicator(prompt)
    elif embed_choice == choices[1]:
        return model.get_similarity_indicator(prompt)
    elif embed_choice == choices[-1]:
        return "NOT IMPLEMENTED"

def main():
    model_choice, embed_choice, true_number = setup_sidebar()
    prompt = None
    choices = ["Glove Vectors + 1-D CNN", "Spacy Embeddings", "SentenceTransformer"]
    # Setup the Streamlit interface
    st.title("🚀 Computer vs Human Guess Game")
    if not prompt:
        prompt = st.chat_input(placeholder="Enter Hint...")

    if model_choice == "Non-LLM":
        low, high = 1, 100
        model = Model()
        if embed_choice == choices[0]:
            model.load_cnn_model()
        elif embed_choice == choices[1]:
            model.load_spacy()
        elif embed_choice == choices[-1]:
            pass

        comp_guess, total_tries = 50, 0
        guess = False

        while guess is False:
            ai_message(comp_guess)
            # Maximum of 100 tries
            if total_tries > 100:
                ai_message(f"Computer Guess not within 100 tries; {total_tries} taken")
                ai_message(f"Game Over...")
                break

            # wait_for_prompt(prompt, comp_guess, choices, model, embed_choice)
            # Check if the guess is the correct number
            if comp_guess == true_number:
                total_tries += 1
                guess = True
                ai_message("Correct Guess")
                # Success Messages
                ai_message(f"Computer Guess correctly in {total_tries}")
                ai_message(f"Game Over...")

            # Else give hints and update the variables
            else:
                prompt = input("Hint = ")
                # Get the Similarity or the Model results
                indicator = get_indicator(prompt, model, choices, embed_choice)
                total_tries+=1
                # If the model predicted indicator is "HOT"
                if indicator == 1:
                    # Update the lower value of the range for the computer to guess
                    low = comp_guess
                # If the model predicted indicator is "COLD"
                else:
                    # Update the higher boundary for the computer guess
                    high = comp_guess
            # Generate random number between [low, high] as the computer guess
            comp_guess = random_number_generator(low, high)    

        st.info("Game Over")
        st.stop()

# Execute the main function
if __name__ == "__main__":
    main()

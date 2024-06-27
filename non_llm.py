import tensorflow as tf
import streamlit as st
import random, spacy
from tensorflow.keras.preprocessing import text, sequence
import numpy as np
from model import Model

def random_number_generator(low: int, high: int) -> int:
    """"
    Generate a random Int within a range.
    """
    return random.randint(low, high)

def ai_message(msg: str):
    """
    Displays a message from the Computer in the chat.

    Args:
        msg (str): The message to be displayed.
    """
    print("Computer Guess - ", msg)

def load_model(embed_choice, model):
    # Load the Model Based on user choice
    if embed_choice == 1:
        model.load_cnn_model()
    elif embed_choice == 2:
        model.load_spacy()
    elif embed_choice == 3:
        model.load_transformer()

def get_indicator(embed_choice: int, model: Model, prompt: text) -> int:
    # Load the Model Based on user choice
    if embed_choice == 1:
        return model.get_cnn_indicator(prompt)
    elif embed_choice == 2:
        return model.get_similarity_indicator(prompt)
    elif embed_choice == 3:
        return model.get_transformer_indicator(prompt)


def main():
    # Human Number
    true_number = int(input("Enter value:"))
    prompt = None

    # Define the Variables and the models
    low, high = 1, 100
    model = Model()
    embed_choice = int(input("Please enter embedding choice (1= CNN; 2=Spacy; 3=SentenceTransformer) - "))
    if embed_choice not in [1,2,3]:
        print("You Entered Wrong Choice; Switching Over to Default")
        embed_choice = 1

    # Load the Model based on user choice
    load_model(embed_choice, model)

    # Initial Computer Guess is 50
    comp_guess, total_tries = 50, 0
    guess = False

    # Loop over the computer guesses until the computer guesses correctly.
    while guess is False:
        ai_message(comp_guess)

        # Maximum of 100 tries
        if total_tries > 15:
            ai_message(f"Computer Guess not within 100 tries; {total_tries} taken")
            ai_message(f"Game Over...")
            break

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
            # Get the SImilarity or the Model results
            indicator = get_indicator(embed_choice, model, prompt)
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

  
# Execute the main function
if __name__ == "__main__":
    main()

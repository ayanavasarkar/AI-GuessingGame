import tensorflow as tf
import streamlit as st
import random, spacy
from tensorflow.keras.preprocessing import text, sequence
import numpy as np


def random_number_generator(low, high):
    return random.randint(low, high)

def ai_message(msg: str):
    """
    Displays a message from the Computer in the chat.

    Args:
        msg (str): The message to be displayed.
    """
    print("Computer Guess - ", msg)


def main():
    # Human Number
    true_number = int(input("Enter value:"))
    prompt = None

    # Define the Variables and the models
    low, high = 1, 100
    model = Model()
    model.load_spacy()

    # Initial Computer Guess is 50
    comp_guess, total_tries = 50, 0
    guess = False

    # Loop over the computer guesses until the computer guesses correctly.
    while guess is False:
        ai_message(comp_guess)
        # Check if the guess is the correct number
        if comp_guess == true_number:
            total_tries += 1
            guess = True
            ai_message("Correct Guess")
        # Else give hints and update the variables
        else:
            prompt = input("Hint = ")
            # Get the SImilarity or the Model results
            indicator = model.get_similarity_indicator(prompt)
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

    # SUccess Messages
    ai_message(f"Computer Guess correctly in {total_tries}")
    ai_message(f"Game Over...")
        
# Execute the main function
if __name__ == "__main__":
    main()

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import random
import statistics


class AI_Agent():
    def __init__(self, api_key) -> None:
        # os.environ.get("GROQ_API_KEY")
        self.model = ChatGroq(
                api_key= api_key,
                model="llama3-70b-8192"
            )

    def ai_agent(self):
        self.agent = Agent(
                role='Guessing Game Assistant',
                goal='take in an user input text and understand the text. Your job is to thoroughly analyze the text \
                and classify whether the indication is "Hot" or "Cold". You must classify one of the two. You cannot classify any other category.',
                backstory='You are a master at understanding whether the text indicates "hot" or "cold" based on its content.',
                llm = self.model,
                verbose = True,
                allow_delegation=False,
                max_iter=5,
                # memory=True,
            )
    
    def ai_task(self, data):
        self.task = Task(
            description=f"""Conduct a comprehensive analysis of the text entered by human and categorize it into one of the following categories: \
                hot - if the text can be summarized as being related to hot \
                cold -  if the text can be summarized as being related to cold \
                
                USER INPUT:\n\n {data} \n\n
                Output a single cetgory only""",
            expected_output="""A single categtory for the type of prompt from the types \
                [hot, cold] \
                eg:
                hot """,
            output_file=f"categorized_user_input.txt",
            agent=self.agent,
            )
    
    def ai_crew(self):
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            verbose=2,
            process=Process.sequential,
            full_output=True,
            share_crew=False,
        )
    
        prompt_result = self.crew.kickoff()
        return prompt_result['final_output']

def ai_message(msg: str):
    """
    Displays a message from the Computer in the chat.

    Args:
        msg (str): The message to be displayed.
    """
    print("Computer Guess - ", msg)

def random_number_generator(low: int, high: int) -> int:
    """"
    Generate a random Int within a range.
    """
    return random.randint(low, high)


def main(api_key):
    
    print("-------------------------------------------------------------------------------------------------------------------")
    print("---------------------------- We are using AI agents to run this Human vs Computer Game ----------------------------")
    print("-------------------------------------------------------------------------------------------------------------------")

    
    if not api_key:
        print("You cannot proceed without an api key")
        exit(0)
    human_number = int(input("Enter Human Number:"))

    # Define the Variables and the models
    low, high = 1, 100
    ai = AI_Agent(api_key)
    ai.ai_agent()

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
        if comp_guess == human_number:
            total_tries += 1
            guess = True
            ai_message("Correct Guess")
            # Success Messages
            ai_message(f"Computer Guess correctly in {total_tries}")
            ai_message(f"Game Over...")
        # Else give hints and update the variables
        else:
            prompt = input("Hint - ")
            ai.ai_task(prompt)
            answer = ai.ai_crew()
            print(answer)
            total_tries+=1
            if answer == "hot":
                # Update the lower value of the range for the computer to guess
                low = comp_guess
            # If the model predicted indicator is "COLD"
            else:
                # Update the higher boundary for the computer guess
                high = comp_guess
        # Generate random number between [low, high] as the computer guess
        comp_guess = random_number_generator(low, high)
            
    print(total_tries)
    return total_tries

# Execute the main function
if __name__ == "__main__":
    results = []
    api_key = input("Enter your model API Key: ")
    for i in range(5):
        results.append(main(api_key))
    
    print("Mean == ", statistics.mean(results))
    print("STD == ", statistics.stdev(results))

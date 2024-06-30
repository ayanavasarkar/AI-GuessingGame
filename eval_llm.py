from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import random
import statistics


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

api_key = input("Api Key - ")
ai = AI_Agent(api_key)
ai.ai_agent()
import pandas as pd
df= pd.read_csv("/home/aya/AI-GuessingGame/data/train_data.csv", sep=";", encoding='cp1252')

# Create the Test data and format it with Labels [0 or 1]
data = []
label = ""
for i,v in df.iterrows():
    e = (v.tolist())[0].split(',')
    label = 1 if e[-1] == 'hot' else 0
    data.append([e[0], e[-1], label])

df = pd.DataFrame(data)
df.columns = ['phrase', 'category', 'label']
from sklearn.model_selection import train_test_split

# Dataset Split
train, test = train_test_split(df, test_size=0.2)

x_test = test.phrase.values
y = test.label.values
pred = []

# Iterate through the test data to predict
for each_text in x_test:
    ai.ai_task(each_text)
    answer = ai.ai_crew()
    if answer == "hot":
        pred.append(1)
    else:
        pred.append(0)


# Plot COnfusion Matrix
cm = confusion_matrix(y, pred)
fig = sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show(fig)
# breakpoint()
# fig.savefig('temp.png', dpi=fig.dpi)

# Get Test Metrics for Evaluation
print(accuracy_score(y, pred),f1_score(y, pred))
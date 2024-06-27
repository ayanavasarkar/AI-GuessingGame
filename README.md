# AI-GuessingGame
Human vs Computer Number Guessing Game with and without LLMs
The goal is to design a simple human-vs-machine guessing
game.
The rules are as follows:\
- The human thinks of a number between 1-100
- The computer provides a guess
- The human responds, in natural language text, whether the computer is closer or
further away from their number by describing temperature (“hotter” = closer).\
Note 1: Assume the seed guess is always 50 \
Note 2: Assume the human is always honest 

We have a training set of phrases (about 650) with 337 phrases for "HOT" and 315 for "COLD". In detail, it contains descriptive phrases and their hot/cold label. [data](https://github.com/ayanavasarkar/AI-GuessingGame/blob/main/data/train_data.csv)

1. Infer, from the human response, a binary directional indicator, and use it to guide
the computer’s guesses.
2. Evaluate the performance of (1) in terms of mean and standard deviation of
number of guesses to solution, and correctness of interpretation of the human’s
response.

## Approach - 1 (Without LLMs)

### 1-Dimensional CNN with Glove Embeddings
First, we can use different NLP embeddings to embed the text from the [data](https://github.com/ayanavasarkar/AI-GuessingGame/blob/main/data/train_data.csv) and compare each of their performances based on the hot/cold label. We trained a 1-Dimensional CNN with the training data, though, it is very very small. We embed the text with Glove Vectors for this 1-D CNNs.\
The aim of the CNN is to predict whether the enter user prompt is "HOT" or "COLD".


### Spacy's Pre-trained Embeddings
We can either use the medium or large pre-trained models for the embeddings. `en_core_web_lg`
Then we are going to:
- Take the user entered hints and embed it using the spacy model.
- Then we compare the similarity score between the user entered text hint and the words "HOT" and "COLD" respectively.
- Then we take the maximum similarity score based on whether it is "HOT" or "COLD".

### Sentence Transformer Embeddings
We can either pre-trained models for the embeddings. `Sentence Transformer`
Then we are going to:
- Take the user entered hints and embed it using the spacy model.
- Then we compare the similarity score between the user entered text hint and the words "HOT" and "COLD" respectively.
- Then we take the maximum similarity score based on whether it is "HOT" or "COLD".

## Approach - 2 (With LLMs)

import datetime
import numpy as np
import os
import json
import spacy
import time
from brain import train
nlp = spacy.load('en')

# Loading training data
training_data = []
# training_data_file = 'training_data.json' 
training_data_file = 'training_data_specific_domain.json' 
with open(training_data_file) as data_file: 
    training_data = json.load(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '-PRON-']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nlp(pattern['sentence'])
    # Stemming and removing words
    # add to our words list
    lemmas = [w1.lemma_ for w1 in w if w1.lemma_ not in ignore_words]
    words.extend(lemmas)
    # add to documents in our corpus
    documents.append(([w1.orth_ for w1 in w],lemmas, pattern['class']))    
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# remove duplicates
words = list(set(words))

# remove duplicates
classes = list(set(classes))

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of lemmatized words for the pattern
    pattern_words = doc[1]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[2])] = 1
    output.append(output_row)

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, classes, words, hidden_neurons=40, alpha=0.05, epochs=200000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
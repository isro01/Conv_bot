#general multipurpose libraries
import csv
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import spacy
from bs4 import BeautifulSoup
import time

#libraries for keras model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding , Dense
from keras.losses import SparseCategoricalCrossentropy

nltk.download('stopwords')

file_1 = "../dataset/data.csv"

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
print(stopWords)

also_remove = ['  ' , '.' , '..' , '...' , '[' , ']', '}', '{', '(' , ')' ,',']

reviews = []
labels = []

start_time = time.time()

with open(file_1 , 'r') as csvfile:
  reader = csv.reader(csvfile , delimiter = ',')
  next(reader)
  for row in reader:
    if row[1] == "positive": labels.append(1)
    if row[1] == "negative" : labels.append(0)
    # labels.append(row[1])
    review = row[0].lower()
    #BeautifulSoup(review, "lxml")                                               #remove html tags
    review = review.replace('[^\w\s]','')                                       #remove puntuations
    review = review.replace("<br /><br />"," ")                                 #remove particular tags
    review = "".join([i for i in review if not i.isdigit()])                    #remove_digits
    for word in stopWords:                                                      #remove stopwords
      stopword = " " + word + " "
      review = review.replace(stopword , " ")
    for i in also_remove:
      review = review.replace(i,"")
    reviews.append(review)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')


print("review : ",reviews[0])



max_len = 0

for review in reviews:
  length = len(review)
  if length>max_len : max_len = length

print("max_len :",max_len)
print("total reviews: ", len(reviews))


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
oov_tok = "<OOV>"
vocab_size = 10000
max_len = 200

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(reviews[:4000])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(reviews[:3000])
padded = pad_sequences(sequences,maxlen=200, truncating="post")

padded = padded[:4000]
print(padded[0])

training_set = []

for row in range(padded.shape[0]):
  train = []
  for col in range(padded.shape[1]):
    train.append(padded[row][col])
  training_set.append(train)


padded_train = training_set[:4000]
padded_test = training_set[2000:3000]
print(type(training_set))
print(training_set[0])


#CBOW_MODEL
context_word = []
target_word = []



window_size = 1

for sentence in padded_train:
  for i,word in enumerate(sentence):
    

    if i- window_size >= 0 and i+ window_size < len(sentence):
      # w = np.zeros((vocab_size+1,1))
      # w[word] = 1
      target_word.append(word)
      context = []

      for j in range(i-window_size ,i+window_size +1):
        if j!= i: 
          # w = np.zeros((vocab_size+1,1))
          # w[sentence[j]] = 1
          context.append(sentence[j])

      context_word.append(context)



test_context_word = []
test_target_word = []

window_size = 2

for sentence in padded_test:
  for i,word in enumerate(sentence):
    # w = np.zeros((vocab_size+1,1))
    # w[word] = 1
    test_target_word.append(word)
    test_context = []

    if i- window_size >= 0 and i+ window_size < len(sentence):

      for j in range(i-window_size ,i+window_size +1):
        if j!= i: 

          # w = np.zeros((vocab_size+1,1))
          # w[sentence[j]] = 1
          test_context.append(sentence[j])

      test_context_word.append(context)

# len(test_context_word)

# type(test_target_word[0])




embedding_dim = 32
max_length = 2
model = tf.keras.Sequential([
         tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
         tf.keras.layers.Flatten(),     
         tf.keras.layers.Dense(vocab_size, activation='softmax')
         ])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='adam',metrics=['accuracy'])
model.summary()


context_word = np.array(context_word)
target_word = np.array(target_word)
test_context_word = np.array(test_context_word)
test_target_word = np.array(test_target_word)

num_epochs = 5

if __name__ == '__main__':

    MODEL = model.fit(context_word , target_word , epochs=num_epochs)
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras import layers

# opening wordlist
wordlist = 'wordlist.10000'
text = open(wordlist, "r").read()

# variables to set
maxlen = 999
step = 3

# Pre-Processing data
sentences = []
next_chars = [] #holds the targets
for i in range(0, len(text)-maxlen, step):
	sentences.append(text[i:i+maxlen])
	next_chars.append(text[i+maxlen])
#VECTORIZATION
chars = sorted(set(text))
char_indices = dict((char, chars.index(char)) for char in chars)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1

#LSTM Builder
model = tf.keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation="softmax"))

#Compiling model
model.compile(loss="categorical_crossentropy", optimizer="adam")



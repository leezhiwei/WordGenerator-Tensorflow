#TextGen with numpy
import numpy	#Importing necessary modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTMV2
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tkinter as tk
from tkinter import *

#Vars
filename = "wordlist.10000"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))
n_vocab = len(chars)

#Vars Part2
dataX = []
seq_length = 100
dataY = []
for i in range(0, len(raw_text) - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([chars_to_int[char]for char in seq_in])
    dataY.append(chars_to_int[seq_out])
X = numpy.reshape(dataX,(len(dataX), seq_length, 1))
y = np_utils.to_categorical(dataY)


#Loading network weights
filename = "model.hdf5" # Change your .hdf5 here
model = Sequential()
model.add(LSTMV2(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Convert Integer to Characters
int_to_char = dict((i, c) for i, c in enumerate(chars))

#Random seed to start off
start = numpy.random.randint(0, len(dataX)-1)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in dataX[start]]), "\"")

#Character Generation
def textgen():
	num=0
	while num < 20:		# Adjust amount of loops
		x = numpy.reshape(dataX[start],(1, len(dataX[start]), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in dataX[start]]
		dataX[start].append(index)
		dataX[start] = dataX[start][1:len(dataX[start])]
		num = num + 1
		word.insert('2.0', result)

#Initialize Tkinter window
window = tk.Tk()
word = tk.Text(height=28, width=45)
button = tk.Button(text="Generate a list of words!",
		width=22,
		height=3,
		bg="black",
		fg="white",
		command = textgen)
word.pack()
word.insert('2.0','Press the "Generate a list of words!" button to continue.')
button.pack()
window.mainloop()
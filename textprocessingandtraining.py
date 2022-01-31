#From https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
#LSTM Development
import numpy    #importing main lib
from keras.models import Sequential
from keras.layers import LSTMV2, Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import CallbackList, ModelCheckpoint
from keras.utils import np_utils
import os 
import glob

#Number of epochs (set here)
epoch = 20

#Loading and lowercasing text
filename = "wordlist.10000"     #Configure wordlist here
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

#Preperation to convert text to integers
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))

#Summarise Dataset
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocabs: ", n_vocab)

#Prepare dataset of I/O pairs encoded as integers
seq_length = 100    #Set Sequence length
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([chars_to_int[char]for char in seq_in])
    dataY.append(chars_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

#Input Sequences to form
X = numpy.reshape(dataX, (n_patterns, seq_length, 1)) #X reshape to be [samples, time steps, features]
X = X / float(n_vocab) # normalise
y = np_utils.to_categorical(dataY) # one hot encode the output variable

#LSTM Define
model = Sequential()
model.add(LSTMV2(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#Define Checkpoint
filepath="weights-improvement-{epoch}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Fitting of model to data
model.fit(X, y, epochs=epoch, batch_size=128, callbacks=callbacks_list) # Change batch_size according to ram (experimental)

#Find last epoch HDF5
for file in glob.glob("./weights-improvement-{}-*.hdf5".format(epoch)):
    os.rename(file,'model.hdf5') #Rename last hdf5 to model.hdf5

#Delete other model files
for weights in glob.glob("weights*.hdf5"):
    os.remove(weights) #Removes other weights-improvement files to save space
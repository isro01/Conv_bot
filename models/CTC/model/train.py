import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import soundfile as sf
from scipy.io import wavfile #for audio processing
import random

import tensorflow as tf
print(tf.__version__)
from keras.models import Model , Sequential
from keras.utils import Sequence
import keras

all_classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'house', 'nine', 'four', 'happy', 'wow', 'five', 'zero', 'tree', 'six', 'one', 'three', 'eight', 'cat', 'two', 'seven', 'marvin', 'dog', 'sheila', 'bed', 'bird']

print("number of classes in your dataset",len(all_classes))

max_label = 40

char_map_str = """
<SPACE> 26
a 0
b 1
c 2
d 3
e 4
f 5
g 6
h 7
i 8
j 9
k 10
l 11
m 12
n 13
o 15
p 15
q 16
r 17
s 18
t 19
u 20
v 21
w 22
x 23
y 24
z 25
_ 28
> 27
"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    if ch == "<SPACE>":
        ch = " "
    char_map[ch] = int(index)
    index_map[int(index)] = ch

index_map[0] = ' '
def get_label(Y , max_label):
    new = [] 
    for c in Y:
        if c not in char_map:
            continue
        elif c == "_":
            continue
        else:
            ch = char_map[c]
            new.append(ch)
        
    new.append(27)
    while(len(new) < max_label):
        new.append(28)
    label = np.array(new)
    
    return label


print("character map",char_map)

dir_path = '../dataset/data'

os.chdir(dir_path)

list_IDs = []

for direc in os.listdir():
        file = [ f for f in os.listdir(os.getcwd() + '/' + direc ) if f.endswith('.wav')]
        for f in file:
            list_IDs.append(direc + '/' + f)

print("string :",len(list_IDs))
print((list_IDs[0]),"\n",(list_IDs[500]))


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, all_classes, list_IDs, max_label, char_map, total_timedistributed_output = 101, 
                 batch_size = 100, noise_factor = 0.1 , add_noise = False , normalise = False ,
                 dim = (101,594) ,shuffle = True ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.all_classes = all_classes
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.noise_factor = noise_factor
        self.add_noise = add_noise
        self.normalise = normalise
        self.max_label = max_label
        self.char_map  = char_map
        self.total_timedistributed_output = total_timedistributed_output
        self.on_epoch_end()

    def __len__(self):
        
        'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
        return 100

    def __getitem__(self, index):
        'Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        input_length = np.array([self.total_timedistributed_output for _ in range(self.batch_size)])
        label_length = np.array([self.max_label for _ in range(self.batch_size)])
        
        inputs = {"the_inputs": X, "the_labels": y, "input_length":input_length, "label_length": label_length}
        outputs = {"ctc": y}
        return (inputs, outputs)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size , max_label ), dtype=int)
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            index_2 = random.randint(0,len(list_IDs_temp)-1)
            index_3 = random.randint(0,len(list_IDs_temp)-1)
            
            sample_1 = self.__graph_spectrogram(os.getcwd() + '/' + ID)
            sample_2 = self.__graph_spectrogram(os.getcwd() + '/' + list_IDs_temp[index_2])
            sample_3 = self.__graph_spectrogram(os.getcwd() + '/' + list_IDs_temp[index_3])
            
            sample = np.concatenate([sample_1 , sample_2 , sample_3] , axis =1 )
            
            if(sample.shape != self.dim):
                a = np.zeros(self.dim)
                if(sample.shape[1] < 594):
                    a[: , :sample.shape[1]] = sample
                sample = a    
            if(self.add_noise):
                sample = self.__add_noise(sample , self.noise_factor)
            if(self.normalise):
                sample = self.__normalise_spectrogram(sample)
                
            X[i,] = sample

            # Store label
            y[i,] = self.__get_label((ID.split('/')[0] + ' '+ list_IDs_temp[index_2].split('/')[0] 
                                      + ' '+ list_IDs_temp[index_3].split('/')[0]) , self.max_label)

        return X, y
    
    def __graph_spectrogram(self, wav_file):
        rate, data = self.__get_wav_info(wav_file)
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        return pxx

    # Load a wav file
    def __get_wav_info(self, wav_file):
        rate , data = wavfile.read(wav_file)
        return rate, data

    def __modify_spectrogram_shape(self, sample ,shape = (101,198) ):
        a = np.zeros(shape)
        a[: , :sample.shape[1]] = sample
        return sample
    
    def __add_noise(self, sample , noise_factor):
        noise = np.random.randn(sample.shape)
        augmented_data = sample + noise_factor * noise
        augmented_data = augmented_data.astype(type(sample[0]))
        return augmented_data

    def __normalise_spectrogram(self, sample):
        mean = np.mean(sample, axis=0)
        std = np.std(sample, axis=0)
        sample = (sample - mean) / std

        return sample
    
    def __get_label(self, Y , max_label):
        new = [] 
        for c in Y:
            if c not in self.char_map:
                continue
            elif c == "_":
                continue
            else:
                ch = self.char_map[c]
                new.append(ch)

        while(len(new) < max_label):
            new.append(27)
        label = np.array(new)

        return label

training_generator = DataGenerator(all_classes ,list_IDs, 40, char_map)

from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length )    
    
class CTC():  
    def __init__(self,
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None
        self.tm = None
                   
    def build(self, 
              conv_filters = 200,
              conv2d_filters = 13,
              conv_size = 5,
              conv2d_strides = 1,
              conv_strides = 1,
              act = 'relu',
              rnn_layers = 2,
              LSTM_units = 128,
              drop_out = 0.8):
           
        input_data = Input(shape = self.input_size, name = 'the_inputs')
        x = Conv1D(conv_filters, 
                   conv_size, 
                   strides = conv_strides,
                   padding = "same", 
                   name = 'conv1d1')(input_data)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = Conv1D(conv_filters, 
                   conv_size, 
                   strides = conv_strides,
                   padding = "same", 
                   name = 'conv1d2')(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        for _ in range(rnn_layers):          
            x = Bidirectional(LSTM(LSTM_units, 
                                   return_sequences = True))(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization()(x)
        y_pred = TimeDistributed(Dense(self.output_size, 
                                       activation = 'softmax'))(x)        
        # ctc inputs
        labels = Input(name='the_labels', shape=[None,], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')    
        loss_out = Lambda(ctc_lambda_func, 
                          output_shape=(1,), 
                          name='ctc')([y_pred,
                                        labels,
                                        input_length,
                                        label_length])        
        self.tm = Model(inputs = input_data,
                        outputs = y_pred)
        self.m = Model(inputs = [input_data, 
                                 labels, 
                                 input_length, 
                                 label_length], 
                        outputs = loss_out)
        return self.m, self.tm

def ctc(y_true, y_pred):
    return y_pred

model_ctc = CTC((101,594), 28)
model_ctc.build()


model_ctc.m.compile(loss = ctc, optimizer = 'adam', metrics = ['accuracy'])

print("Now training your model!!")

history = model_ctc.m.fit_generator(training_generator, epochs=4)

os.chdir('../../checkpoints')

model_json = model_ctc.to_json()
with open("model_ctc.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# -*- coding: utf-8 -*-
# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import tensorflow as tf
from tensorflow import keras
from six.moves import zip as izip
import numpy as np

# from Util import read_base
# from Util import trata_tf_palavra
from Util import dividir_base
import _pickle as cPickle
from keras.initializers import Constant
# gpu =  tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(tf.ConfigProto(gpu_options=gpu))

# imdb = keras.datasets.imdb

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
rooting = 'data_set/data_with_count_w2v.txt'
x = cPickle.load(open(rooting,"rb"))
data_number, data_labels, w2v = x[0], x[1], x[2]

data_number = [list(w) for w in data_number]

# (train_data, train_labels), (test_data, test_labels) = trata_tf_palavra(read_base(rooting), 0.75,'word2vecs/skip_s50.txt')
(train_data, train_labels), (test_data, test_labels) = dividir_base(data_number,data_labels,0.75)
maxlen = train_data.shape[1]
print(type(maxlen))
print(type(maxlen))
print(type(train_data))
print(len(train_data))
print(type(train_data[0]))
print(len(train_data[1]))
print('X_train shape:', train_data.shape[1])
print('X_test shape:', test_data.shape)
# exit()


# NAME = "test1-{}".format(int(time.time()))
# tensor_board = tf.keras.callbacks.TensorBoard(log_dir='log/{}'.format(NAME))


# A dictionary mapping words to an integer index
vocab_size = 447

# train_data = keras.preprocessing.sequence.pad_sequences(train_data,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=vocab_size)
#
# test_data = keras.preprocessing.sequence.pad_sequences(test_data,
#                                                        value=0,
#                                                        padding='post',
#                                                        maxlen=vocab_size)


# tf.cast(train_data, tf.int32)
# tf.cast(test_data, tf.int32)

# input shape is the vocabulary count used for the movie reviews (10,000 words)np
# vocab_size = 2282

# with tf.Session(tf.ConfigProto(gpu_options=gpu)) as sess:
max_features = len(w2v)
embedding_dims = len(w2v[0])
with tf.Session() as sess:
    # n =np.int32(32)
    n = embedding_dims
    model = keras.Sequential()
    # model.add(keras.layers.Input(shape=(maxlen,), dtype='float32', name='main_input'))
    model.add(keras.layers.Embedding(max_features, n,
                                     weights=[np.matrix(w2v)],
                                     input_length=maxlen
                                     ))
    # Embedding(max_features, embedding_dims,
    #           weights=[np.matrix(W)], input_length=maxlen,
    #           name='embedding')(main_input)
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Flatten())
    # n /=2
    # model.add(keras.layers.LSTM(n,activation='softmax'))
    n /=2
    model.add(keras.layers.Dense(n, activation=tf.nn.softmax))
    n /= 2
    # model.add(keras.layers.SimpleRNN(n))
    # n /=2
    model.add(keras.layers.Dense(n, activation=tf.nn.relu6))
    #
    # # n /=2
    model.add(keras.layers.Dense(1, activation=tf.nn.tanh))
    # model.add(keras.layers.Dense(1, activation=tf.nn.softmax))

    model.summary()
    # keras.layers.CuDNNLSTM

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # limit = vocab_size
    # limit = 10
    # x_val = train_data[:limit]
    # partial_x_train = train_data[limit:]
    #
    #
    # y_val = train_labels[:limit]
    # partial_y_train = train_labels[limit:]

    # sess = tf.Session()
    history = model.fit(train_data,
                        train_labels,
                        epochs=30,
                        batch_size=1,
                        validation_split=0.2,
                        verbose=1)

    # sess
    # results = model.evaluate(test_data, test_labels)
    # model.save('epic_num_reader.model')
    # new_model = keras.models.load_model('epic_num_reader.model')
    predictions = model.predict(test_data)

    path = 'plots/result-1.txt'
    len_labels = len(test_labels)
    with open(path, mode='w', encoding='utf-8') as csv_file:
        # writer = csv.writer(csv_file)
        i=0
        while(len_labels>i):
            csv_file.writelines(str([predictions[i],test_labels[i]]) + '\n')
            i+=1
# sess.run()
# for w in predictions:
#     print(w)
# exit()
# with tf.Session() as sess:
#
#     print(results)
#     exit()
# print('---')
# print(s)

'''

history_dict = history.history
history_dict.keys()


## Inicio da classificação
import matplotlib.pyplot as plt


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
'''

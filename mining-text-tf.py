# -*- coding: utf-8 -*-
#@title MIT License
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
# import random
# import math
from tensorflow import keras
# from tensorflow.python.layers.convolutional import Conv2D

from executaveis_teste.Util import readBase,trata_tf_palavra,trata_tf_tf_idf,trata_tf_3
# import TensorBoard
# import time

# print(tf.__version__)

# gpu =  tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(tf.ConfigProto(gpu_options=gpu))

# imdb = keras.datasets.imdb

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
dir = 'data_set\colecao_dourada_2_class_unbalanced.csv'
(train_data, train_labels), (test_data, test_labels) = trata_tf_palavra(readBase(dir), 0.75)


# NAME = "test1-{}".format(int(time.time()))
# tensor_board = tf.keras.callbacks.TensorBoard(log_dir='log/{}'.format(NAME))


# A dictionary mapping words to an integer index
# print(len(train_data[0]))
# exit()
vocab_size = len(train_data[0])+1
# vocab_size = 10000
# # vocab_size = 2267
# # vocab_size = 59200
# # vocab_size = 822577
#
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=vocab_size)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=vocab_size)
# tf.cast(train_data, tf.int32)
# tf.cast(test_data, tf.int32)
# eliaquim-m
# print(train_data[0])
# print(train_data[1])
# len(train_data[0]), len(train_data[1])



# input shape is the vocabulary count used for the movie reviews (10,000 words)np
# vocab_size = 2282

# with tf.Session(tf.ConfigProto(gpu_options=gpu)) as sess:
# with tf.Session() as sess:

n =64
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, n))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Flatten())
n /=2
model.add(keras.layers.LSTM(n,recurrent_activation='softmax'))
n /=2
# model.add(keras.layers.Dense(n, activation=tf.nn.relu))

n /=2
model.add(keras.layers.Dense(n, activation=tf.nn.sigmoid))
# n /=2
model.add(keras.layers.Dense(1, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.softmax))

model.summary()



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
                    validation_split =0.2,
                    verbose=1)

# sess
# results = model.evaluate(test_data, test_labels)


# model.save('epic_num_reader.model')
# new_model = keras.models.load_model('epic_num_reader.model')
predictions = model.predict(test_data)

path = 'C:/Users/User/PycharmProjects/pcc/plots/result-1.txt'
with open(path,mode='w', encoding='utf-8') as csv_file:
    #writer = csv.writer(csv_file)
    for w in predictions:
        csv_file.writelines(str(w)+'\n')
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
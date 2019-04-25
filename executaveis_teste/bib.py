'''
import tensorflow as tf
from Util import readBase,trata_tf_palavra,trata_tf_tf_idf,trata_tf_3
# import time


# gpu =  tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(tf.ConfigProto(gpu_options=gpu))

# imdb = keras.datasets.imdb

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
dir = 'data_set\colecao_dourada_2_class_unbalanced.csv'
(train_data, train_labels), (test_data, test_labels) = trata_tf_palavra(readBase(dir), 0.75)


n_hidden_1 = 10        # 1st layer number of features
n_hidden_2 = 5         # 2nd layer number of features
n_input = len(train_data)  # Words in vocab
n_classes = 2          # Categories: graphics, space and baseball
def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)
    # Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    return out_layer_addition

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(train_data, weights, biases)
# Define loss
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=test_data, labels=test_labels)
loss = tf.reduce_mean(entropy_loss)'''


import numpy as np    #numpy is a package for scientific computing
from collections import Counter
vocab = Counter()
text = "Hi from Brazil"
#Get all words
for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase]+=1

#Convert words to indexes
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        print(str(i) + ' = ' + word)
        word2index[word] = i

    return word2index
#Now we have an index
word2index = get_word_2_index(vocab)
total_words = len(vocab)
#This is how we create a numpy array (our matrix)
matrix = np.zeros((total_words),dtype=float)
#Now we fill the values
for word in text.split():
    matrix[word2index[word.lower()]] += 1
print(matrix)

matrix = np.zeros((total_words),dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1
print(matrix)



categories = ["comp.graphics","sci.space"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
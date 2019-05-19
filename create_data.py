# -*- coding: utf-8 -*-
import numpy as np
import _pickle as cPickle
from Util import get_data_freq_Count
from Util import remocaopontos
from Util import get_all_words

from six.moves import zip as izip
import sys
# from Util import tf_idf
# from Util import contar_palavra_doc
# from Util import get_numbers
# from Util import get_data_idf

def read_base(csvFile):
    import csv
    base = []

    with open(csvFile, newline='\n', encoding='utf-8') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                # com caracteres especiais
                temp1 = row[2].lower()
                temp2 = row[1].lower()
                base.append(tuple([temp1, temp2]))
            except IndexError:
                pass
        return base


def get_W2V_numbers(word_vecs,all_words, limiar=50):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(all_words)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, limiar), dtype='float32')
    W[0] = np.zeros(limiar, dtype='float32')
    i = 1
    for word in all_words:
        if(word_vecs.get(word) is None):
            W[i] = np.random.uniform(-0.25, 0.25, limiar)
        else:
            W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def embeding(data, label,dir_w2v):
    import gensim
    # wang2vector
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v,binary=True)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v)
    data_w2v = {}
    # distancia de palavra com no maximo 5
    data_documents = []
    label_documents = []
    len_data = len(data)

    def word_embeding(op,data_w2v, limiar = 50):
        len_opinion = len(op)
        i = 0
        while (i < len_opinion):
            if(data_w2v.get(op[i]) is None):
                data_w2v[op[i]] = w2v.distance(op[i])[:limiar]
            i += 1
        return data_w2v

    def op_embeding(op, number, rate):
        len_opinion = len(op)
        data_news = []
        i = 0
        while (i < len_opinion):
            similiar_words = [(op[i], float(1) )]
            try:
                similiar_words += w2v.most_similar(op[i], topn=number)
            except Exception:
                pass
            for similiar_word, freq in similiar_words:
                if (freq > rate):
                    data_news.append(op[:i] + [similiar_word] + op[i + 1:])
                else:
                    break
            i += 1
        return data_news

    def op_embeding_news_skip(op, rate,data_w2v):
        len_opinion = len(op)
        data_news = []
        i = 0
        while (i < len_opinion):
            similiar_words = []

            try:
                similiar_words += w2v.most_similar(op[i])
            except Exception:
                pass
            words = []
            j = 0
            for similiar_word, freq in similiar_words:
                if(j==i):
                    words.append(op[i])
                else:
                    if (freq > rate):
                        words.append(similiar_word)
                j+=1
            if(len(words)>0):
                data_news.append(words)
            i += 1
        return data_news,data_w2v

    i = 0
    while (i < len_data):  # percorre as palavras da linha
        op_new,aux_w2v = op_embeding_news_skip(data[i], 0.9,data_w2v)


        data_w2v.update(aux_w2v)
        data_documents.append(data[i])
        data_documents += op_new
        label_documents.append(label[i])
        label_documents += [label[i]] * (len(op_new))

        print(' - ' + str(i))
        i += 1

    return data_documents, label_documents,data_w2v

def trata_tf_palavra(base,dir_w2v,b_w2v = True,len_data_init=100):
    import nltk
    # import random

    # montar base baseado se tem a palavra/character , com a sequencia
    data = []
    data_labels = []
    # random.shuffle(base)
    tknzr = nltk.tokenize.TweetTokenizer()

    i = len(base) - 1
    if(len_data_init<i):
        i = len_data_init
    # tokenização e remoção de pontuação
    while(i>=0):
        data.append(remocaopontos(tknzr.tokenize(base[i][0])))
        data_labels.append(int(base[i][1]) - 1)
        i -= 1

    # pegar todas as palavras
    all_words = get_all_words(data)
    data_w2v = {}
    print('all_words',len(all_words))
    if(b_w2v):
        data, data_labels,data_w2v = embeding(data, data_labels, dir_w2v)

    # pega todas as palavras

    all_words = [w for w in get_all_words(data)]
    rand_idx = np.random.permutation(range(len(all_words)))
    all_words = [all_words[i] for i in rand_idx]
    print('all_words',len(all_words))
    # all_words = random.shuffle(all_words)

    # data_number = get_numbers(data,all_words)
    data_number = get_data_freq_Count(data,all_words)
    W,_ = get_W2V_numbers(data_w2v,all_words)


    # data_number = get_data_freq_Count_space(data, all_words)
    # data_number = get_data_idf(data)
    data_number = [tuple(u) for u in data_number]

    return data_number.copy(), data_labels.copy(),W

# if __name__ == '__main__':
# w2v,data_name,pickle_name = sys.argv[1],sys.argv[2],sys.argv[3]
w2v = 'word2vecs/skip_s50-1.txt'
data_name = 'data_set/colecao_dourada_2_class_unbalanced.csv'
pickle_name = 'data_set/data_with_count_w2v.txt'
# # data_name = 'data_set/data_with_tfidf_w2v.txt'
data_number, data_labels, data_w2v = trata_tf_palavra(read_base(data_name), w2v)
cPickle.dump([ data_number, data_labels,data_w2v], open(pickle_name, "wb"))
x = cPickle.load(open(pickle_name, "rb"))
d_n, d_l, wv = x[0], x[1],x[2]
# print(wv)
# print(d_l)

print ("dataset created!")
#python create_data.py word2vecs/skip_s50-1.txt data_set/colecao_dourada_2_class_unbalanced.csv data_set/data_with_count_w2v.txt

# -*- coding: utf-8 -*-
import numpy as np
import _pickle as cPickle
from Util import get_data_freq_Count
from Util import remocaopontos
from Util import get_all_words
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

def get_W2V_numbers(word_vecs,all_words, limiar):
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
            W[i] = word_vecs[word][:limiar]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    code = {
        '':bytes('', 'utf-8'),
        ' ':bytes(' ', 'utf-8'),
        '\n':bytes('\n', 'utf-8')
    }

    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)

                if ch == code[' ']:
                    word = code[''].join(word)
                    break
                if ch != code['\n']:
                    word.append(ch)
            word = word.decode('utf-8')
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def embeding(data, dir_w2v):
    import gensim
    # wang2vector
    if(dir_w2v[len(dir_w2v)-3:]=='bin'):
        all_words = get_all_words(data)
        w2v = load_bin_vec(dir_w2v, all_words.keys())
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v)
    data_w2v = {}
    len_data = len(data)

    def word_embeding(op,data_w2v):
        len_opinion = len(op)
        i = 0
        while (i < len_opinion):
            if(data_w2v.get(op[i]) is None):
                try:
                    data_w2v[op[i]] = w2v.distance(op[i])
                except Exception:
                    pass
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
        # op_new,aux_w2v = op_embeding_news_skip(data[i], 0.9,data_w2v)
        aux_w2v = word_embeding(data[i],data_w2v)
        data_w2v.update(aux_w2v)
        i += 1
    return data_w2v

def trata_tf_palavra(base,dir_w2v,b_w2v = True,len_data_init=1000,limiar = 50):
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
    print('w2v embeding = >>')
    if(b_w2v):
        data_w2v = embeding(data, dir_w2v)

    # pega todas as palavras

    all_words = [w for w in get_all_words(data)]
    rand_idx = np.random.permutation(range(len(all_words)))
    all_words = [all_words[i] for i in rand_idx]
    # all_words = random.shuffle(all_words)

    # data_number = get_numbers(data,all_words)
    data_number = get_data_freq_Count(data,all_words)
    data_w2v,_ = get_W2V_numbers(data_w2v,all_words,limiar=limiar)

    # data_number = get_data_freq_Count_space(data, all_words)
    # data_number = get_data_idf(data)
    data_number = [tuple(u) for u in data_number]

    return data_number.copy(), data_labels.copy(),data_w2v

if __name__ == '__main__':
    w2v,data_name,pickle_name = sys.argv[1],sys.argv[2],sys.argv[3]
    # w2v = 'word2vecs/skip_s50-1.txt'
    # data_name = 'data_set/colecao_dourada_2_class_unbalanced.csv'
    # pickle_name = 'data_set/data_with_count_w2v.txt'
    # # data_name = 'data_set/data_with_tfidf_w2v.txt'
    data_number, data_labels, data_w2v = trata_tf_palavra(read_base(data_name), w2v)
    cPickle.dump([ data_number, data_labels,data_w2v], open(pickle_name, "wb"))
    x = cPickle.load(open(pickle_name, "rb"))
    d_n, d_l, wv = x[0], x[1],x[2]
# print(wv)
# print(d_l)

    print ("dataset created!")

#python create_data.py word2vecs/skip_s50-1.txt data_set/colecao_dourada_2_class_unbalanced.csv data_set/data_with_count_w2v_wang.txt
#python create_data.py word2vecs/skip_s300.txt data_set/colecao_dourada_2_class_unbalanced.csv data_set/data_with_count_w2v_wang.txt

#python create_data.py executaveis_teste/crnn_master/word2vecs/GoogleNews-vectors-negative300.bin data_set/colecao_dourada_2_class_unbalanced.csv data_set/data_with_count_w2v.txt

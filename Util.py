# -*- coding: utf-8 -*-



def read_base(csvFile):
    import csv
    base = []

    with open(csvFile, newline='\n', encoding='utf-8') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                # sem caracteres especiais
                # vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
                # print(row[2])
                # temp1 = ''
                # for w in row[2]:
                #     if (w not in vida):
                #         temp1 +=str(w).lower()
                # temp2 = row[1]

                # com caracteres especiais
                temp1 = row[2].lower()
                temp2 = row[1].lower()
                base.append(tuple([temp1, temp2]))
            except IndexError:
                pass
        return base

def trata_tf_palavra(base, porc_traing,dir_w2v):
    import nltk
    import numpy as np
    b_w2v = True

    # montar base baseado se tem a palavra/character , com a sequencia
    data = []
    data_labels = []
    # random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    # i = len(base) - 1
    i = 50
    # tokenização e remoção de pontuação
    while(i>=0):
        data.append(remocaopontos(tknzr.tokenize(base[i][0])))
        data_labels.append(base[i][1])
        i -= 1

    # pegar todas as palavras
    # all_words = get_all_words(data)

    # distancia de palavra com no maximo 5
    len_data = len(data)
    # print(len(data))
    if(b_w2v):
        data, data_labels = embeding(data, data_labels,dir_w2v)
    # print(len(data))
    # pega todas as palavras
    all_words = get_all_words(data)

    # data_number = get_numbers(data,all_words)
    # data_number = get_data_freq_Count(data,all_words)
    # data_number = get_data_freq_Count_space(data, all_words)
    data_number = get_data_idf(data)

    dttype = np.int64
    ''''''
    f = lambda x, g: g(x, dtype=np.float32)

    data_fraq_1 = data_number[0:int(len_data * porc_traing)]
    data_fraq_2 = data_number[int(len_data * porc_traing):]

    train_data = np.array([f(w, np.array) for w in data_fraq_1])
    test_data = np.array([f(w, np.array) for w in data_fraq_2])

    # train_data = np.array(data_number[0:int(len_data*porc_traing)])
    # test_data = np.array(data_number[0:int(len_data*porc_traing)])

    train_labels = np.array(data_labels[0:int(len_data * porc_traing)], dtype=dttype)
    test_labels = np.array(data_labels[int(len_data * porc_traing):], dtype=dttype)
    return (train_data, train_labels), (test_data, test_labels)


def palavra(numero_caracter, palavras_caracter, palavra):
    i = len(palavras_caracter) - 1
    l = len(palavra) - 1

    while (i >= 0):
        while (l >= 0):
            if (palavras_caracter[i] == palavra[l]):
                numero_caracter[i] = 1
            l -= 1
        i -= 1

    return numero_caracter


def contar_palvra(data, palavra):
    i = 0
    for w in data:
        for c in w[0]:
            if (palavra == c):
                i += 1
    return i


def contar_palavra_doc(document, palavra):
    i = 0
    for word in document:
        if (palavra == word):
            i += 1
    return i


def palavra_contexto(word, context):
    j = 0
    while (j < len(context)):  # olha todas as palavras no dicionario
        if (word == context[j][0]):
            return j
        j += 1
    return None


def tf_idf(term, len_doc, term_docs, len_docs):
    import math
    return (term / len_doc) / (math.log(len_docs / term_docs, 10))


def remocaoacento(base):
    vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
    base2 = []
    for row in base:
        temp1 = ''
        for w in row:
            if (w not in vida):
                temp1 += w.lower()
        base2.append(temp1)
    return base2


def remocaopontos(opinion):
    point = ['.', ',', '!', '?', '<', '>', ']', '[', '*', '(', ')', '+', '-', ';', ':', '...', '$', '%', '\'', '..',
             '"', '/']
    opinion_news = []
    for row in opinion:
        if (row not in point):
            opinion_news.append(row.lower())
    return opinion_news


def remocao_ponto(c):
    vida = ['.', ',', '!', '?', '<', '>', ']', '[', '*', '(', ')', '+', '-', ';', ':', '...', '$', '%', '\'', '..', '"',
            '/']
    return (c not in vida)


def matriz_confusao(label, predicao):
    w = [[0] * 2] * 2
    i = 0
    while (i < len(label)):
        w[label[i]][predicao[i]] += 1
        i += 1
    return w


def embeding(data, label,dir_w2v):
    import gensim
    # wang2vector
    # w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v,binary=True)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v)


    # distancia de palavra com no maximo 5
    data_documents = []
    label_documents = []
    len_data = len(data)

    def word_embeding(word, number, rate):
        words = []
        similiar_words = [(word, 1)]
        try:
            similiar_words += w2v.most_similar(word, topn=number)
        except Exception:
            pass
        for similiar_word, freq in similiar_words:
            if (freq > rate):
                words.append(similiar_word)
            else:
                break
        return words


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
    def op_embeding_news_skip(op, number, rate):
        len_opinion = len(op)
        data_news = []
        i = 0
        while (i < len_opinion):
            similiar_words = []
            try:
                similiar_words += w2v.most_similar(op[i], topn=len_opinion - 1)
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
        return data_news

    i = 0
    while (i < len_data):  # percorre as palavras da linha
        # j=0
        # len_opinion = len(data[i])
        # word_embedings = []
        # while (j < len_opinion):
        op_new = op_embeding_news_skip(data[i], 5, 0.9)

        data_documents.append(data[i])
        data_documents += op_new
        label_documents.append(label[i])
        label_documents += [label[i]] * (len(op_new))


        print(' - ' + str(i))
        i += 1
    return data_documents, label_documents


def get_all_words(data):
    # pegar todas as palavras
    i = 0
    len_data = len(data)
    all_words = {}
    while (i < len_data):  # olha cada op
        m = len(data[i])
        n = 0
        while (n < m):  # olha cada palavra da op
            if (data[i][n] not in all_words):  # Tem a palavra?
                all_words[data[i][n]] = 1
            else:
                all_words[data[i][n]] += 1
            n += 1
        i += 1
    return all_words


def get_numbers(data, all_words):
    i = 0
    data_news_numbers = []
    len_data = len(data)
    while (i < len_data):  # percorre as palavras da linha
        words = []
        for word in all_words:
            words.append(contar_palavra_doc(data[i], word))
        data_news_numbers.append(words.copy())
        i += 1
    return data_news_numbers


def get_data_idf(data):
    i = 0
    sum_words = 0
    len_data = len(data)
    all_words = {}
    while (i < len_data):  # olha cada op
        m = len(data[i])
        sum_words += m
        n = 0
        while (n < m):  # olha cada palavra da op
            if (data[i][n] not in all_words):  # Tem a palavra?
                all_words[data[i][n]] = 1
            else:
                all_words[data[i][n]] += 1
            n += 1
        i += 1

    # len_all_word = len(all_words)
    data_old_numbers = get_numbers(data, all_words)

    words_all_number = []
    for p, q in all_words.items():
        words_all_number.append(q)
    i = 0
    data_number = []
    len_data = len(data_old_numbers)
    len_all_word = len(words_all_number)
    while (i < len_data):  # percore as linhas
        w = []
        j = 0
        while (j < len_all_word):  # percorre as character que está no contexto
            if (data_old_numbers[i][j] != 0):
                tfi = tf_idf(data_old_numbers[i][j], len(data_old_numbers[i]), words_all_number[j], sum_words)
                # tfi = int(tfi * 10000)
                w.append(tfi)
            else:
                w.append(0)
            j += 1
        data_number.append(w.copy())
        i += 1
    return data_number


def get_data_freq_boolean(data, all_words):
    i = 0
    data_number = []
    len_data = len(data)
    while (i < len_data):  # percore as linhas
        w = []
        for word in all_words:
            w.append(1 if (word in data[i]) else 0)
        data_number.append(w)
        i += 1
    return data_number


def get_data_freq_Count(data, all_words):
    # import random
    i = 0
    data_number = []
    len_data = len(data)
    all_word = [word for word in all_words]
    len_all_word = len(all_word)
    # random.shuffle(all_word)
    while (i < len_data):  # percore as linhas
        w = []
        j = 1
        for word in all_word:
            if (word in data[i] and len(word)<len_all_word):
                w.append(j)
            j += 1
        len_w = len(w)
        l = len_all_word - len_w
        if (l > 0):
            w += [0] * (l)
        data_number.append(w.copy())
        i += 1
    return data_number


def get_data_freq_Count_space(data, all_words):
    import random
    i = 0
    data_number = []
    len_data = len(data)
    all_word = [word for word in all_words]
    len_all_word = len(all_word)
    random.shuffle(all_word)
    while (i < len_data):  # percore as linhas
        w = []
        j = 1
        for word in all_word:
            w.append(j if (word in data[i]) else 0)
            j += 1
        data_number.append(w)
        i += 1
    return data_number

def dividir_base(data_number,data_labels,porc_traing):
    import numpy as np
    from tensorflow import keras
    dttype = np.int32
    len_data = len(data_number)
    limiar = int(len_data * porc_traing)
    f = lambda x: np.array(x, dtype=np.float32)
    g = lambda y: [g(x) for x in y] if (type(y[0])==list) else np.max(y)
    vocab_size = 64
    # vocab_size = g(g(data_number))


    train_data = data_number[0:limiar].copy()
    # train_data = [f(w) for w in train_data]
    # train_data = f(train_data)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=vocab_size)
    test_data = data_number[limiar:].copy()
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=vocab_size)



    # test_data = [f(w) for w in test_data]

    # train_data = np.array(data_number[0:int(len_data*porc_traing)])
    # test_data = np.array(data_number[0:int(len_data*porc_traing)])

    train_labels = np.array(data_labels[0:limiar], dtype=dttype)
    test_labels = np.array(data_labels[limiar:], dtype=dttype)
    return (train_data.copy(), train_labels.copy()), (test_data.copy(), test_labels.copy())

def build_model(maxlen,max_features, embedding_dims,W):
    import numpy as np
    import keras
    from keras.layers import Dropout, Embedding, Conv1D, MaxPooling1D, Input, Dense
    from keras.layers.recurrent import  GRU
    from keras.constraints import maxnorm
    from keras.models import Model
    from keras.optimizers import Adagrad
    import time


    batch_size = 50
    nb_filter = 200
    # filter_length = 4
    hidden_dims = nb_filter * 2
    # nb_epoch = 60
    RNN = GRU
    rnn_output_size = 100
    # folds = 10
    main_input = Input(shape=(maxlen, ), dtype='int32', name='main_input')
    embedding  = Embedding(max_features, embedding_dims,
                           weights=[np.matrix(W)], input_length=maxlen,
                           name='embedding')(main_input)

    embedding = Dropout(0.50)(embedding)

    conv4 = Conv1D(filters=nb_filter,
                   kernel_size=4,
                   padding='valid',
                   activation='relu',
                   strides=1,
                   name='conv4')(embedding)
    maxConv4 = MaxPooling1D(pool_size=2,
                            name='maxConv4')(conv4)

    conv5 = Conv1D(filters=nb_filter,
                   kernel_size=5,
                   padding='valid',
                   activation='relu',
                   strides=1,
                   name='conv5')(embedding)
    maxConv5 = MaxPooling1D(pool_size=2,
                            name='maxConv5')(conv5)

    # x = merge([maxConv4, maxConv5], mode='concat')
    x = keras.layers.concatenate([maxConv4, maxConv5])

    x = Dropout(0.15)(x)

    x = RNN(rnn_output_size)(x)

    x = Dense(hidden_dims, activation='relu', kernel_initializer='he_normal',
              kernel_constraint = maxnorm(3), bias_constraint=maxnorm(3),
              name='mlp')(x)

    x = Dropout(0.10, name='drop')(x)

    output = Dense(1, kernel_initializer='he_normal',
                   activation='sigmoid', name='output')(x)

    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy',
                  # optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                  # optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                  optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4),
                  metrics=["accuracy"])
    return model
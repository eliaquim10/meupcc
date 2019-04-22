# -*- coding: utf-8 -*-
import csv
import numpy as np
import random
import nltk
import numpy
import math


def readBase(csvFile = str):
    base = []

    with open(csvFile, newline='\n',encoding='utf-8') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                #sem caracteres especiais
                # vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
                # print(row[2])
                # temp1 = ''
                # for w in row[2]:
                #     if (w not in vida):
                #         temp1 +=str(w).lower()
                # temp2 = row[1]

                #com caracteres especiais
                temp1 = row[2].lower()
                temp2 = row[1].lower()
                base.append(tuple([temp1, temp2]))
            except IndexError:
                pass
        return base

def arrays(array, maxlen):
    i = len(array)-1
    # data
    print(array[0])
    tam = 0
    while (i>=0):
        l = len(array[i])
        tam = maxlen - l
        w = [numpy.int64(0)]*tam
        array[i] += w
        i-=1
    return array

def trata(base,porc_traing):
    data = []

    # random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    i = len(base) - 1


    while (i>=0):
        data.append([tknzr.tokenize(base[i][0]),numpy.int64(base[i][1])-1])
        i -= 1

    k = 0
    l = len(data)
    all_words = []
    while (k < l):
        m = len(data[k][0])
        n = 0
        while (n < m):
            if data[k][0][n] not in all_words:
                all_words.append(data[k][0][n])
            n += 1
        k += 1

    print(len(all_words))
    random.shuffle(all_words)

    k = 0
    l = len(data)
    data_number =[]
    data_labels =[]
    maxlen=0
    t = len(all_words)
    while (k < l):#percore as linhas
        m = len(data[k][0])
        n = 0
        w =[]
        if m>maxlen:
            maxlen = m
        while (n < m):#percorre as palavras da linha
            j =0
            while(j<t):#percorre as palavras que está no contexto
                if (data[k][0][n] == all_words[j]):
                    w.append(j+1)
                j+=1
            n += 1
        data_number.append(w)
        data_labels.append(data[k][1])
        k += 1

    # print(str(len(all_words)))
    # keras.layers.CuDNNLSTM()

    train_data = data_number[0:int(l*porc_traing)]
    test_data = data_number[int(l*porc_traing):]

    # train_labels = numpy.ndarray(data_labels[0:int(l*porc_traing)])
    train_labels = np.array(data_labels[0:int(l*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(l*porc_traing):], dtype=np.int64)

    '''
    train_data = data_number[0:int(l*porc_traing)]
    train_labels = [w[1] for w in data[0:int(l*porc_traing)]]
    test_data = data_number[int(l*porc_traing):]
    test_labels = [w[1] for w in data[int(l*porc_traing):]]
    '''

    return (train_data,train_labels),(test_data, test_labels)

def trata_tf_palavra(base, porc_traing):
    import nltk
    b_w2v = False

    # montar base baseado se tem a palavra/character , com a sequencia
    data = []
    label = []
    # random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    # i = len(base) - 1
    i = 10
    #tokenização e remoção de pontuação
    while (i>=0):
        data.append(remocaopontos(tknzr.tokenize(base[i][0])))
        label.append(base[i][1])
        i -= 1

    # pegar todas as palavras
    # all_words = get_all_words(data)

    # distancia de palavra com no maximo 5
    len_data = len(data)
    if(b_w2v):
        data = embeding(data)

    # pega todas as palavras
    all_words = get_all_words(data)

    # data_numbers = get_numbers(data,all_words)
    data_number = get_data_freq_Count(data, all_words)

    # data_number = get_data_idf(data)
    data_labels =[w for w in label]

    train_data = np.array(data_number[0:int(len_data*porc_traing)], dtype=np.int64)
    test_data = np.array(data_number[int(len_data*porc_traing):], dtype=np.int64)

    train_labels = np.array(data_labels[0:int(len_data*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(len_data*porc_traing):], dtype=np.int64)
    return (train_data,train_labels),(test_data, test_labels)

def trata_tf_tf_idf(base, porc_traing):
    b_w2v = True
    if(b_w2v):
        import gensim
        w2v = gensim.models.KeyedVectors.load_word2vec_format("word2vec-pt-br-master/exemplo/skip_s50-1.txt")
    # montar base baseado se tem a palavra/character , com a sequencia
    data = []
    # random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    i = len(base) - 1
    #tokenização e remoção de pontuação
    while (i>=0):
        data.append([remocaopontos(tknzr.tokenize(base[i][0])), numpy.int64(base[i][1])-1])
        i -= 1

    if(b_w2v):
        i = 0
        dataset = []
        # word to vector
        for opinion_class in data:
            dataset.append([[],opinion_class[1]])
            for word in opinion_class[0]:
                try:
                    similiar_words = w2v.most_similar(word, topn=5)

                    dataset[i][0].append(word)
                    dataset[i][0] += list(similiar_words.keys())

                    # for similiar_word,freq in similiar_words:
                    #     dataset[i][0].append(similiar_word)
                except Exception:
                    dataset[i][0].append(word)
            i+=1
        data = dataset.copy()

    i = 0
    sum_words = 0
    len_data = len(data)
    all_words = {}
    while (i < len_data):# olha cada op
        sum_words += len(base[i][0])
        m = len(data[i][0])
        n = 0
        while (n < m):# olha cada palavra da op
            if(data[i][0][n] not in all_words):# Tem a palavra?
                all_words[data[i][0][n]] = 1
            else:
                all_words[data[i][0][n]] += 1
            n += 1
        i += 1

    # for p,q in all_words.items():
    #     try:
    #         print(p,' - ', q)
    #     except Exception:
    #         pass
    # exit()

    # random.shuffle(all_words)
    i = 0

    len_all_word = len(all_words)
    data_documents =[]
    while (i < len_data):#percorre as palavras da linha

        w=[0]*len_all_word
        j=0
        for word in all_words:
            if(w[j]==0):
                w[j] = contar_palavra_doc(data[i][0], word)
            j+=1
        data_documents.append(w.copy())
        i+=1


    words_all_number = []
    for p,q in all_words.items():
        words_all_number.append(q)

    i=0
    data_number =[]
    data_labels =[]
    while (i < len_data):#percore as linhas
        m = len(data[i][0])
        w =[0]*len_all_word
        j = 0
        while(j<len_all_word):#percorre as character que está no contexto
            if(data_documents[i][j]!=0):
                w[j] = tf_idf(data_documents[i][j],len(data[i][0]),words_all_number[j],sum_words)
            j+=1
        # print(len(w))
        data_number.append(w)
        data_labels.append(data[i][1])
        i += 1

    train_data = data_number[0:int(len_data*porc_traing)]
    test_data = data_number[int(len_data*porc_traing):]

    # print(len(train_data[0]))
    # print(len([ x for x in train_data[0] if x > 0 ]))
    # print(len(train_data[1]))
    # print(len([ x for x in train_data[1] if x > 0 ]))
    # exit()

    # train_labels = numpy.ndarray(data_labels[0:int(l*porc_traing)])
    train_labels = np.array(data_labels[0:int(len_data*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(len_data*porc_traing):], dtype=np.int64)

    return (train_data,train_labels),(test_data, test_labels)

def trata_tf_3(base,porc_traing):
    import gensim
    # montar base baseado se tem a palavra/character , não a sequencia
    data = []

    # random.shuffle(base)
    # C:\Users\User\PycharmProjects\pcc\word2vec-pt-br-master\exemplo\skip_s50.txt
    w2v = gensim.models.KeyedVectors.load_word2vec_format("word2vec-pt-br-master/exemplo/skip_s50.txt")
    tknzr = nltk.tokenize.TweetTokenizer()



    i = len(base) - 1

    while (i>=0):
        data.append([
            remocaopontos(tknzr.tokenize(base[i][0])),
            numpy.int64(base[i][1])-1])
        i -= 1


    i = 0

    # word to vector
    dataset =[]
    
    for opinion_class in data:
        dataset.append([[],0])
        for word in opinion_class[0]:
            try:
                similiar_word = w2v.most_similar(word, topn=10)
                j = 0
                for words,freq in similiar_word:
                    dataset[i][0].append(words)
            except Exception:
                pass
        i+=1
    print(dataset)
    exit()
    print(dataset[50][0])

    k = 0
    l = len(data)
    all_words = []
    while (k < l):# olha cada op
        m = len(data[k][0])
        n = 0
        while (n < m):# olha cada palavra da op

            resul = palavra_contexto(data[k][0][n],all_words)
            if(not (data[k][0][n] in all_words)):# Tem a palavra?
                all_words.append(data[k][0][n])
            n += 1
        k += 1
    #monta o dicionario de palavras


    i = len(all_words) - 1
    character_words_all = []
    while(i>=0):
        j = len(all_words[i]) - 1
        while(j>=0):
            character_words_all.append(all_words[i][j])
            j-=1
        i-=1
    # monta um dicionario com todas as palavras ordenado os caracteres

    character_words_all = character_words_all[::-1]

    k = 0
    l = len(data)
    data_number =[]
    data_labels =[]
    t = len(character_words_all)
    sum_letra = 0
    while(k < l):
        sum_letra += len(base[k][0])
        k+=1
    k=0

    while (k < l):#percore as linhas
        m = len(data[k][0])
        numbers =[0]*t
        n = 0

        while (n < m):#percorre as palavras da linha
            numbers = palavra_1(numbers,character_words_all,data[k][0][n],all_words)
            n+=1

        data_number.append(numbers)
        data_labels.append(data[k][1])
        k += 1

    # random.shuffle(data_number)

    train_data = data_number[0:int(l*porc_traing)]
    test_data = data_number[int(l*porc_traing):]

    # train_labels = numpy.ndarray(data_labels[0:int(l*porc_traing)])
    train_labels = np.array(data_labels[0:int(l*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(l*porc_traing):], dtype=np.int64)

    return (train_data,train_labels),(test_data, test_labels)

def palavra(numero_caracter,palavras_caracter,palavra):
    i = len(palavras_caracter) - 1
    l = len(palavra) - 1

    while(i>=0):
        while(l>=0):
            if(palavras_caracter[i]==palavra[l]):
                numero_caracter[i] = 1
            l-=1
        i-=1

    return numero_caracter

def palavra_1(numero_caracter,palavras_caracter,palavra,palavras):
    i = 0
    boolean = False
    k = i

    while(i<len(palavras)):
        if(palavras[i]==palavra):
            boolean = True
            break
        else:
            if(not boolean):
                k +=len(palavras[i])
        i+=1
    li = k
    ls = (k + len(palavra))
    while(li<ls):
        numero_caracter[li] = 1
        li+=1
    return numero_caracter

def contar_palvra(data, palavra):
    i = 0
    for w in data:
        for c in w[0]:
            if(palavra == c):
                i+=1
    return i

def contar_palavra_doc(document, palavra):
    i = 0
    for word in document:
        if(palavra == word):
            i+=1
    return i

def palavra_contexto(word,context):
    j=0
    while(j < len(context)): # olha todas as palavras no dicionario
        if (word == context[j][0]):
            return j
        j+=1
    return None

def tf_idf(term,len_doc,term_docs,len_docs):
    return (term/len_doc)/(math.log(len_docs/term_docs,10))

def remocaoacento(base):
    vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
    base2 = []
    for row in base:
        temp1 = ''
        for w in row:
            if (w not in vida):
                temp1 +=w.lower()
        base2.append(temp1)
    return base2

def remocaopontos(opinion):
    point = ['.',',','!','?','<','>',']','[','*','(',')','+','-',';',':','...','$', '%', '\'', '..', '"', '/']
    opinion_news = []
    for row in opinion:
        if (row not in point):
            opinion_news.append(row.lower())
    return opinion_news

def remocao_ponto(c):
    vida = ['.',',','!','?','<','>',']','[','*','(',')','+','-',';',':','...','$', '%', '\'', '..', '"', '/']
    return (c not in vida)

def matriz_confusao(label,predicao):
    w = [[0]*2]*2
    i = 0
    while(i<len(label)):
        w[label[i]][predicao[i]]+=1
        i+=1
    return w

def embeding(data):
    import gensim
    # wang2vector
    w2v = gensim.models.KeyedVectors.load_word2vec_format("word2vec-pt-br-master/exemplo/skip_s50-1.txt")
    i = 0
    # distancia de palavra com no maximo 5
    data_documents =[]
    len_data = len(data)
    def word_embeding(word,number):
        words = [word]
        similiar_words = w2v.most_similar(word, topn=number)
        for similiar_word,freq in similiar_words:
            if(freq>0.9):
                words.append(similiar_word)
            else:
                break
        return words

    while (i < len_data):#percorre as palavras da linha
        words=[]
        for word in data[i]:
            try:
                words+=word_embeding(word,5)
            except Exception:
                words.append(word)
            # else:
            #     words.append(word)
        # print(words)
        data_documents.append(words.copy())
        print(' - ' + str(i))
        i+=1
    return data_documents
def get_all_words(data):
    # pegar todas as palavras
    i = 0
    len_data = len(data)
    all_words = {}
    while (i < len_data):# olha cada op
        m = len(data[i])
        n = 0
        while (n < m):# olha cada palavra da op
            if(data[i][n] not in all_words):# Tem a palavra?
                all_words[data[i][n]] = 1
            else:
                all_words[data[i][n]] += 1
            n += 1
        i += 1
    return all_words

def get_numbers(data,all_words):
    i=0
    data_news_numbers =[]
    len_data = len(data)
    while (i < len_data):#percorre as palavras da linha
        words=[]
        for word in all_words:
            words.append(contar_palavra_doc(data[i], word))
        data_news_numbers.append(words.copy())
        i+=1
    return data_news_numbers

def get_data_idf(data):
    i = 0
    sum_words = 0
    len_data = len(data)
    all_words = {}
    while (i < len_data):# olha cada op
        m = len(data[i])
        sum_words += m
        n = 0
        while (n < m):# olha cada palavra da op
            if(data[i][n] not in all_words):# Tem a palavra?
                all_words[data[i][n]] = 1
            else:
                all_words[data[i][n]] += 1
            n += 1
        i += 1

    len_all_word = len(all_words)
    data_old_numbers = get_numbers(data,all_words)


    words_all_number = []
    for p,q in all_words.items():
        words_all_number.append(q)
    i=0
    data_number =[]
    len_data = len(data_old_numbers)
    len_all_word = len(words_all_number)
    while (i < len_data):#percore as linhas
        w =[]
        j = 0
        while(j<len_all_word):#percorre as character que está no contexto
            if(data_old_numbers[i][j]!=0):
                tfi = tf_idf(data_old_numbers[i][j],len(data_old_numbers[i]),words_all_number[j],sum_words)
                tfi = int(tfi*10000)
                w.append(tfi)
            else:
                w.append(0)
            j+=1
        data_number.append(w.copy())
        i += 1
    return data_number

def get_data_freq_boolean(data, all_words):
    i=0
    data_number =[]
    len_data = len(data)
    while (i < len_data):#percore as linhas
        w =[]
        for word in all_words:
            w.append(1 if(word in data[i]) else 0)
        data_number.append(w)
        i += 1
    return data_number

def get_data_freq_Count(data, all_words):
    import random
    i=0
    data_number =[]
    len_data = len(data)
    all_word = [word for word in all_words]
    random.shuffle(all_word)
    while (i < len_data):#percore as linhas
        w =[]
        j = 10
        for word in all_word:
            w.append(j if(word in data[i]) else 0)
            j+=1
        data_number.append(w)
        i += 1
    return data_number


# -*- coding: utf-8 -*-
import csv
import numpy as np
import random
import nltk
import numpy
import math
import gensim

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
    data = []

    random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    i = len(base) - 1

    while (i>=0):
        data.append([
            remocaopontos(tknzr.tokenize(base[i][0])),
            numpy.int64(base[i][1])-1])
        i -= 1

    k = 0
    l = len(data)
    all_words = []
    frenq_word_doc = []
    while (k < l):# olha cada op
        m = len(data[k][0])
        n = 0
        while (n < m):# olha cada palavra da op
            entrou = 1
            resul = palavra_contexto(data[k][0][n],all_words)
            if(resul!=None):# Tem a palavra? none-que nao tem no dic
                all_words[resul][1] +=1
            else:
                all_words.append([data[k][0][n],1])

            resul = palavra_contexto(data[k][0][n],frenq_word_doc)
            if(resul!=None):# Tem a palavra? none-que nao tem na op
                frenq_word_doc[resul][1] +=1
            else:
                frenq_word_doc.append([data[k][0][n],1])
            n += 1
        k += 1

    # print([(all_words[i],all_words_f[i])for i in range(len(all_words)) if (all_words_f[i]>50)])
    # print(all_words_f)

    # random.shuffle(all_words)

    k = 0
    l = len(data)
    data_number =[]
    data_labels =[]

    tam = 0
    for w in data:
        tam+=len(w[0])
    maxlen=0
    t = len(all_words)
    while (k < l):#percore as linhas
        m = len(data[k][0])
        w =[]
        j = 0
        entrou = 1
        while(j<t):#percorre as palavras que está no contexto
            n = 0
            while (n < m):#percorre as palavras da linha
                if (all_words[j][0] == data[k][0][n]):
                    tfidf = tf_idf(frenq_word_doc[n][1],m,all_words[j][1],tam)
                    w.append(tfidf)
                    # w.append(frenq_word_doc[n][1]/m)
                    entrou = 0
                n += 1
            if(entrou):
                w.append(0)
            j+=1
        data_number.append(w)
        data_labels.append(data[k][1])
        k += 1


    train_data = data_number[0:int(l*porc_traing)]
    test_data = data_number[int(l*porc_traing):]

    # train_labels = numpy.ndarray(data_labels[0:int(l*porc_traing)])
    train_labels = np.array(data_labels[0:int(l*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(l*porc_traing):], dtype=np.int64)

    return (train_data,train_labels),(test_data, test_labels)

def trata_tf_tf_idf(base, porc_traing):
    # montar base baseado se tem a palavra/character , com a sequencia
    data = []
    w2v = gensim.models.KeyedVectors.load_word2vec_format("skip_s50.txt")
    # random.shuffle(base)

    tknzr = nltk.tokenize.TweetTokenizer()

    i = len(base) - 1
    #tokenização e remoção de pontuação
    while (i>=0):
        data.append([remocaopontos(tknzr.tokenize(base[i][0])), numpy.int64(base[i][1])-1])
        i -= 1


    i = 0

    # word to vector
    dataset =[[]*(len(base)),0]
    for opinion_class in data:
        for word in opinion_class[0]:
            try:
                similiar_word = w2v.most_similar(word, topn=10)
                j = 0
                while(j<len(similiar_word)):
                    dataset[i][0].append(similiar_word[j])
            except Exception:
                pass
        i+=1
    print(dataset[0])
    print(dataset[50])
    exit()

    i = 0
    sum_words = 0
    l = len(data)
    all_words = {}
    while (i < l):# olha cada op
        sum_words += len(base[i][0])
        m = len(data[i][0])
        n = 0
        while (n < m):# olha cada palavra da op
            resul = palavra_contexto(data[i][0][n],all_words)
            if(data[i][0][n] not in all_words):# Tem a palavra?
                all_words.append(data[i][0][n])
                all_words.append(data[i][0][n])
            else:
                all_words.append(data[i][0][n])
            n += 1
        i += 1


    # print([(all_words[i],all_words_f[i])for i in range(len(all_words)) if (all_words_f[i]>50)])
    # print(all_words_f)

    # random.shuffle(all_words)
    i = 0

    t_all_word = len(all_words)
    data_documents =[]
    while (i < l):#percorre as palavras da linha

        w=[0]*t_all_word
        j=0
        while(j<t_all_word): #percore todas as palavras do dicionario
            if(w[j]==0):
                w[j] = contar_palavra_doc(data[i][0], all_words[j])
            j+=1
        data_documents.append(w)
        i+=1

    i = 0
    words_all_number = [0]*t_all_word
    while(i<t_all_word):#quantidade por palavra
        if(words_all_number[i]==0):
            words_all_number[i] = contar_palvra(data, all_words[i])
        i+=1

    i=0
    data_number =[]
    data_labels =[]
    while (i < l):#percore as linhas
        m = len(data[i][0])
        w =[0]*t_all_word
        j = 0
        while(j<t_all_word):#percorre as character que está no contexto
            if(data_documents[i][j]!=0):
                w[j] = tf_idf(data_documents[i][j],len(data[i][0]),words_all_number[j],sum_words)
            j+=1
        # print(len(w))
        data_number.append(w)
        data_labels.append(data[i][1])
        i += 1

    train_data = data_number[0:int(l*porc_traing)]
    test_data = data_number[int(l*porc_traing):]

    # train_labels = numpy.ndarray(data_labels[0:int(l*porc_traing)])
    train_labels = np.array(data_labels[0:int(l*porc_traing)], dtype=np.int64)
    test_labels = np.array(data_labels[int(l*porc_traing):], dtype=np.int64)

    return (train_data,train_labels),(test_data, test_labels)

def trata_tf_3(base,porc_traing):
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

def remocaopontos(base):
    vida = ['.',',','!','?','<','>',']','[','*','(',')','+','-',';',':','...','$', '%', '\'', '..', '"', '/']
    base2 = []
    for row in base:
        if (row not in vida):
            base2.append(row)
    return base2

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
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

    random.shuffle(base)

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

def trata_tf(base,porc_traing):
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
def trata_tf_2(base,porc_traing):
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
    while (k < l):# olha cada op
        m = len(data[k][0])
        n = 0
        while (n < m):# olha cada palavra da op

            resul = palavra_contexto(data[k][0][n],all_words)
            if(not (data[k][0][n] in all_words)):# Tem a palavra?
                all_words.append(data[k][0][n])
            n += 1
        k += 1

    # print([(all_words[i],all_words_f[i])for i in range(len(all_words)) if (all_words_f[i]>50)])
    # print(all_words_f)

    # random.shuffle(all_words)
    i = len(all_words) - 1
    character_words_all = []
    while(i>=0):
        j = len(all_words[i]) - 1
        while(j>=0):
            character_words_all.append(all_words[i][j])
            j-=1
        i-=1



    k = 0
    l = len(data)
    data_number =[]
    data_labels =[]
    t = len(character_words_all)
    character_words_all_number = [0]*t
    sum_letra = 0
    while(k < l):
        sum_letra += len(base[k][0])
        k+=1
    k=0
    maxlen=0
    while (k < l):#percore as linhas
        m = len(data[k][0])
        w =[]
        j = 0
        entrou = 1
        while(j<t):#percorre as character que está no contexto
            n = 0
            while (n < m):#percorre as palavras da linha
                h = len(data[k][0][n]) - 1
                while (h >= 0):#percorre as palavras da linha
                    if (character_words_all[j] == data[k][0][n][h]):
                        w.append(1)
                        h = -1
                        entrou = 0

                    '''
                    if (character_words_all[j] == data[k][0][n][h]):
                        print(str(k)+" "+str(j)+" "+str(n)+" "+str(h))
                        if(character_words_all_number[j]==0):
                            character_words_all_number[j] = contar_letra_base(base,character_words_all[j])
                            character_number_doc = contar_letra_doc(base[k][0],character_words_all[j])
                            tfidf = tf_idf(character_number_doc,len(base[k][0]),character_words_all_number[j],sum_letra)
                            w.append(tfidf)
                            h = -1
                            entrou = 0
                        else:
                            character_number_doc = contar_letra_doc(base[k][0],character_words_all[j])
                            tfidf = tf_idf(character_number_doc,len(base[k][0]),character_words_all_number[j],sum_letra)
                            w.append(tfidf)
                            h = -1
                            entrou = 0
                    '''
                    h-=1
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

def contar_letra_base(base , charac):
    i = 0
    for w in base:
        for c in w[0]:
            if(charac == c):
                i+=1
    return i
def contar_letra_doc(document , charac):
    i = 0
    for w in document:
        if(charac == w):
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

def precisionRecallFmeasure(classifier, gold):
    #results = classifier.classify_many([fs for (fs, l) in gold])
    #correct = [l == r for ((fs, l), r) in zip(gold, results)]

    testclas = classifier.classify_many([fs for (fs, l) in gold])
    testgold = [l for (fs, l) in gold]

    h=0
    #cont = 0
    tpos=0
    fpos=0
    tneg=0
    fneg=0
    tneu=0
    fneu=0
    fneuPos=0
    fneuNeg=0
    fnegPos=0
    fnegNeu=0
    fposNeg=0
    fposNeu=0
    while(h<len(testgold)):
        if (testgold[h]==testclas[h])and (testclas[h]== u"positivo"):
            tpos=tpos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo"):
            fpos=fpos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo") and (testgold[h]==u'negativo'):
            fposNeg=fposNeg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo") and (testgold[h]==u'neutro'):
            fposNeu=fposNeu+1
        if (testgold[h]==testclas[h])and (testclas[h]== u"negativo"):
            tneg=tneg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo"):
            fneg=fneg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo") and (testgold[h]==u'positivo'):
            fnegPos=fnegPos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo") and (testgold[h]==u'neutro'):
            fnegNeu=fnegNeu+1
        if (testgold[h]==testclas[h])and (testclas[h]== u"neutro"):
            tneu=tneu+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro"):
            fneu=fneu+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro") and (testgold[h]==u'positivo'):
            fneuPos=fneuPos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro") and (testgold[h]==u'negativo'):
            fneuNeg=fneuNeg+1
        h=h+1

#-------------------------
    precisionPos = float(tpos)/(tpos+fpos)
    precisionNeg = float(tneg)/(tneg+fneg)
    if((len(set([l for (fs, l) in gold])))==3):
        precisionNeu = float(tneu)/(tneu+fneu)
    else:
        precisionNeu = 0
    precision = float(precisionPos+precisionNeg+precisionNeu)/(len(set([l for (fs, l) in gold])))
#--------------------------
    recallPos = float(tpos)/(tpos+fnegPos+fneuPos)
    recallNeg = float(tneg)/(tneg+fposNeg+fneuNeg)
    if((len(set([l for (fs, l) in gold])))==3):
        recallNeu = float(tneu)/(tneu+fnegNeu+fposNeu)
    else:
        recallNeu = 0
    recall = float(recallPos+recallNeg+recallNeu)/(len(set([l for (fs, l) in gold])))
#---------------------------
    if((len(set([l for (fs, l) in gold])))==3):
        mc = '''
             Pos   Neg   Neu\n
        Pos   '''+str(tpos)+"    "+str(fposNeg)+"    "+str(fposNeu)+'''\n
        Neg   '''+str(fnegPos)+"    "+str(tneg)+"    "+str(fnegNeu)+'''\n
        Neu   '''+str(fneuPos)+"    "+str(fneuNeg)+"    "+str(tneu)
    if((len(set([l for (fs, l) in gold])))==2):
        mc ='''
               Pos   Neg \n
        Pos    '''+str(tpos)+"   "+str(fpos)+'''\n
        Neg    '''+str(fneg)+'''   '''+str(tneg)
#---------------------------

    fmeasure = (2*precision*recall)/(precision+recall)
    fmeasurePos = (2*precisionPos*recallPos)/(precisionPos+recallPos)
    fmeasureNeg = (2*precisionNeg*recallNeg)/(precisionNeg+recallNeg)
    fmeasureNeu = (2*precisionNeu*recallNeu)/(precisionNeu+recallNeu)
#---------------------------

    string = "precision: "+str(precision)+"  precisionPos: "+str(precisionPos)+'  precisionNeg: '+str(precisionNeg)+'  precisionNeu: '+str(precisionNeu)+"" \
             "\n recall: "+str(recall)+"  recallPos: "+str(recallPos)+"  recallNeg: "+str(recallNeg)+"  recallnNeu: "+str(recallNeu)+"" \
             "\n F-measure: "+str(fmeasure)+"  f-measurePos: "+str(fmeasurePos)+"  f-measureNeg: "+str(fmeasureNeg)+"  f-measureNeu: "+str(fmeasureNeu)+"" \
                                                                                                                                                                                "\n"+str(len(set([l for (fs, l) in gold])))+"\nf-measure: "+str(fmeasure)
    return string
'''
    if correct:
        return float(sum(correct))/len(correct)
    #acuracia = float(sum(x == y for x, y in izip(reference, test))) / len(test)
    #precision = float(len(reference.intersection(test)))/len(test)
    #recal    = float(len(reference.intersection(test)))/len(reference)
    #fmeasure =
    else:
        return 0
'''

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
    i = len(all_words) - 1
    words_all = []
    while(i>=0):
        j = len(all_words[i][0]) - 1
        while(j>=0):
            words_all.append(all_words[i][0][j])
            j-=1
        i-=1


    print(words_all)
    exit()

    k = 0
    l = len(data)
    data_number =[]
    data_labels =[]
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
                    tfidf = tf_idf(frenq_word_doc[n][1],m,all_words[j][1],t)
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

def write_results(resultados =[], times =[], silhoetes =[], path = str):

    '''
    Função responsável por escrever um arquivo csv com todos os resultados obtidos
    :param resultados: resultados de cada iteração do PSO
    :param times: o tempo de execução para cada iteração do PSO
    :param silhoetes: Avaliação da semelhança dos clusters antes e depois da execução do PSO
    em cada execução
    :param path: caminho e nome do arquivo a ser salvo
    '''


    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([" ", "Classe 0", "Classe 1","Classe 2", "All Class"])
        writer.writerow(["Execution Number","Precision/Recall/F-Score",
        "Precision/Recall/F-Score","Precision/Recall/F-Score",
        "Precision/Recall/F-Score/Support",
        "Silhouette Coefficient","Time (seconds)"])

    print_vetor = [[z+1, a[0], a[1], a[2],np.mean(a, axis=1), b, c]for z, a, b, c in zip(range(0,len(resultados)),resultados,silhoetes,times)]

    for i in range(len(print_vetor)):
        writer.writerow(print_vetor[i])

    media = np.array(reduce(operator.add, np.mean(np.array(resultados), axis=2)))/len(resultados)
    writer.writerow([" ","Precison", "Recall", "F-Score"])
    writer.writerow(["Media Final:",media[0], media[1], media[2]])
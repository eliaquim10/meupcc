# -*- coding: utf-8 -*-
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import nltk
from Util import readBase,remocaopontos,remocaoacento
import random
import nltk.corpus
import sklearn
import numpy
import csv


def executar(experimento,nome_Base,acento):
    nomeBase = nome_Base
    path = experimento+nomeBase
    # print('executando:\n'+path)
    # print('Sem acento:\n'+('Sim' if(acento) else 'Não'))

    base = readBase(nomeBase)
    tamBase = len(base)
    i=0
    documents = []
    #print base[0][0].split()
    tknzr = nltk.tokenize.TweetTokenizer()

    while (i<tamBase):
        if(acento):
            w = [q.lower() for q in remocaoacento(tknzr.tokenize(base[i][0]))]
        else:
            w = [q.lower() for q in tknzr.tokenize(base[i][0])]
        w = remocaopontos(w)
        conteudoLista = (w,base[i][1])
        documents.append(conteudoLista)
        i += 1

    ################################ Pre Processamento
    stopwords = nltk.corpus.stopwords.words('portuguese')

    stemmer = nltk.stem.RSLPStemmer()

    # h=0
    # j=len(documents)
    # while (h<j):
    #    g=len(documents[h][0])
    #    f=0
    #    while(f<g):
    #        stemmer.stem(documents[h][0][f])
    #        f+=1
    #    h += 1

    ################################

    random.shuffle(documents)

    all_words = []

    k=0
    l=len(documents)
    while (k<l):
        m=len(documents[k][0])
        n=0
        while(n<m):
            all_words.append(documents[k][0][n])
            n+=1
        k += 1
    # all_words = remocaopontos(all_words)
    
    all_words = [w.lower() for w in all_words if w not in stopwords]
    # print(str(all_words))

    #all_words = nltk.FreqDist(all_words) #calcula frequencia de palavras, definir o limite de palavras
    #all_words = nltk.LaplaceProbDist(nltk.FreqDist(all_words))
    #all_words = nltk.SimpleGoodTuringProbDist(nltk.FreqDist(all_words))
    #all_words = nltk.LidstoneProbDist(nltk.FreqDist(all_words), 0.1)
    #all_words = nltk.WittenBellProbDist(nltk.FreqDist(all_words))
    #nltk.WittenBellProbDist() procurar como mudar o ngram
    #all_words = nltk.MLEProbDist(nltk.FreqDist(all_words))
    #all_words = nltk.SimpleGoodTuringProbDist(nltk.FreqDist(w.lower() for w in all_words if w not in stopwords))

    all_words = nltk.LidstoneProbDist(nltk.FreqDist(all_words), 0.1)
    #all_words = nltk.FreqDist(nltk.FreqDist(w.lower() for w in all_words if w not in stopwords))

    word_features = list(all_words.samples()) #se usando FreqDistlista com palavras que aparecem mais de 3000
    
    # word_features =nltk.LidstoneProbDist(nltk.FreqDist(word_features), 0.1)
    # word_features = word_features.samples()
    #word_features = list(all_words.keys())
    '''aqui que modifiquei
    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features
    '''
    #aquii
    
    def wordbigram(word_feature):
        bigram =[]
        i=0
        l = len(word_feature)-1
        while (i<l):
            # if ((not word_feature[i] in stopwords) or (not word_feature[i+1]in stopwords)):
            s = tuple([stemmer.stem(word_feature[i]),stemmer.stem(word_feature[i+1])])
            bigram.append(s)
            i+=1
        return bigram

    def removerpalavras(todas_palavras,document):
        #remover as palavras que não estãoem todas as palavras
        linha = []
        for w in document:
            if(w in todas_palavras):
                linha.append(w)
        return linha

    def wordFeature(documents):
        #cria um dicionario de dados
        dicionario = []
        for w in documents:
            for q in w[0]:
                if(not q in dicionario):
                    dicionario.append(q)   
        return dicionario

    documents = [[removerpalavras(all_words.samples(),w[0]),w[1]] for w in documents]
    documents = [[wordbigram(w[0]),w[1]] for w in documents]
    word_features = wordFeature(documents) #se 0usando FreqDistlista com palavras que aparecem mais de 3000
    # print(str(len(word_features)))
    # exit()
    # word_features = list(all_words.samples())#se 0usando FreqDistlista com palavras que aparecem mais de 3000
    
    def find_features(document):
        # words = set(document)
        features = {}
        i=0
        l = len(word_features)
        while(i<l):
            features[str(i)] = (word_features[i] in document)
            i+=1
        return features
    featuresets = [(find_features(rev), category) for (rev, category) in documents  if (category==1)]
    

    kfold = 4

    baseInteira = featuresets

    tamT = len(featuresets)
    divisao = tamT//kfold

    ###### ajustar divisao
    baseDividida1 = featuresets[0:divisao]
    baseDividida2 = featuresets[divisao:(divisao*2)]
    baseDividida3 = featuresets[(divisao*2):(divisao*3)]
    baseDividida4 = featuresets[(divisao*3):tamT]

    #tamT = len(featuresets)
    #umQuarto = tamBase/4

    #training_set = featuresets[umQuarto:]
    #testing_set = featuresets[:umQuarto]

    #training_set = featuresets[100:]
    #testing_set = featuresets[0:100]

    ########################## 1 rodada
    #print "## RODADA 1 ##"

    training_set = baseDividida2+baseDividida3+baseDividida4
    testing_set = baseDividida1

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    testclas = MNB_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    MNBmc1 = sklearn.metrics.confusion_matrix(testgold, testclas)
    MNBa1 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    MNBpp1 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoMNB1 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPMNB1 = 0
    while(g<len(precisaoMNB1)):
        somaPMNB1 = somaPMNB1+precisaoMNB1[g]
        g=g+1
    MNBpt1 = (somaPMNB1/len(precisaoMNB1))*100
    MNBrp1 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallMNB1 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRMNB1 = 0
    while(g<len(recallMNB1)):
        somaRMNB1 = somaRMNB1+recallMNB1[g]
        g=g+1
    MNBrt1 = (somaRMNB1/len(recallMNB1))*100
    MNBfp1 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1MNB1 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFMNB1 = 0
    while(g<len(f1MNB1)):
        somaFMNB1 = somaFMNB1+f1MNB1[g]
        g=g+1
    MNBft1 = (somaFMNB1/len(f1MNB1))*100
    '''
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    BernoulliNB_classifierRodada2 = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
    print("BernoulliNB_classifier accuracy percent:", BernoulliNB_classifierRodada2*100)
    '''
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    testclas = LogisticRegression_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Rmc1 = sklearn.metrics.confusion_matrix(testgold, testclas)
    Ra1 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Rpp1 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoR1 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPR1 = 0
    while(g<len(precisaoR1)):
        somaPR1 = somaPR1+precisaoR1[g]
        g=g+1
    Rpt1 = (somaPR1/len(precisaoR1))*100
    Rrp1 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallR1 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRR1 = 0
    while(g<len(recallR1)):
        somaRR1 = somaRR1+recallR1[g]
        g=g+1
    Rrt1 = (somaRR1/len(recallR1))*100
    Rfp1 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1R1 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFR1 = 0
    while(g<len(f1R1)):
        somaFR1 = somaFR1+f1R1[g]
        g=g+1
    Rft1 = (somaFR1/len(f1R1))*100

    '''
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    SGDClassifier_classifierRodada2 = nltk.classify.accuracy(SGDClassifier_classifier, testing_set)
    print("SGDClassifier_classifier accuracy percent:", SGDClassifier_classifierRodada2*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    SVC_classifierRodada2 = nltk.classify.accuracy(SVC_classifier, testing_set)
    print("SVC_classifier accuracy percent:", SVC_classifierRodada2*100)
    '''
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    testclas = LinearSVC_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Lmc1 = sklearn.metrics.confusion_matrix(testgold, testclas)
    La1 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Lpp1 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoL1 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPL1 = 0
    while(g<len(precisaoL1)):
        somaPL1 = somaPL1+precisaoL1[g]
        g=g+1
    Lpt1 = (somaPL1/len(precisaoL1))*100
    Lrp1 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallL1 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRL1 = 0
    while(g<len(recallL1)):
        somaRL1 = somaRL1+recallL1[g]
        g=g+1
    Lrt1 = (somaRL1/len(recallL1))*100
    Lfp1 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1L1 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFL1 = 0
    while(g<len(f1L1)):
        somaFL1 = somaFL1+f1L1[g]
        g=g+1
    Lft1 = (somaFL1/len(f1L1))*100

    ######################## Rodada 2
    #print "## RODADA 2 ##"

    training_set = baseDividida1+baseDividida3+baseDividida4
    testing_set = baseDividida2

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    testclas = MNB_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    MNBmc2 = sklearn.metrics.confusion_matrix(testgold, testclas)
    MNBa2 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    MNBpp2 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoMNB2 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPMNB2 = 0
    while(g<len(precisaoMNB2)):
        somaPMNB2 = somaPMNB2+precisaoMNB2[g]
        g=g+1
    MNBpt2 = (somaPMNB2/len(precisaoMNB2))*100
    MNBrp2 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallMNB2 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRMNB2 = 0
    while(g<len(recallMNB2)):
        somaRMNB2 = somaRMNB2+recallMNB2[g]
        g=g+1
    MNBrt2 = (somaRMNB2/len(recallMNB2))*100
    MNBfp2 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1MNB2 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFMNB2 = 0
    while(g<len(f1MNB2)):
        somaFMNB2 = somaFMNB2+f1MNB2[g]
        g=g+1
    MNBft2 = (somaFMNB2/len(f1MNB2))*100
    '''
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    BernoulliNB_classifierRodada2 = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
    print("BernoulliNB_classifier accuracy percent:", BernoulliNB_classifierRodada2*100)
    '''
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    testclas = LogisticRegression_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Rmc2 = sklearn.metrics.confusion_matrix(testgold, testclas)
    Ra2 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Rpp2 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoR2 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPR2 = 0
    while(g<len(precisaoR2)):
        somaPR2 = somaPR2+precisaoR2[g]
        g=g+1
    Rpt2 = (somaPR2/len(precisaoR2))*100
    Rrp2 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallR2 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRR2 = 0
    while(g<len(recallR2)):
        somaRR2 = somaRR2+recallR2[g]
        g=g+1
    Rrt2 = (somaRR2/len(recallR2))*100
    Rfp2 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1R2 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFR2 = 0
    while(g<len(f1R2)):
        somaFR2 = somaFR2+f1R2[g]
        g=g+1
    Rft2 = (somaFR2/len(f1R2))*100

    '''
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    SGDClassifier_classifierRodada2 = nltk.classify.accuracy(SGDClassifier_classifier, testing_set)
    print("SGDClassifier_classifier accuracy percent:", SGDClassifier_classifierRodada2*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    SVC_classifierRodada2 = nltk.classify.accuracy(SVC_classifier, testing_set)
    print("SVC_classifier accuracy percent:", SVC_classifierRodada2*100)
    '''
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    testclas = LinearSVC_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Lmc2 = sklearn.metrics.confusion_matrix(testgold, testclas)
    La2 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Lpp2 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoL2 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPL2 = 0
    while(g<len(precisaoL2)):
        somaPL2 = somaPL2+precisaoL2[g]
        g=g+1
    Lpt2 = (somaPL2/len(precisaoL2))*100
    Lrp2 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallL2 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRL2 = 0
    while(g<len(recallL2)):
        somaRL2 = somaRL2+recallL2[g]
        g=g+1
    Lrt2 = (somaRL2/len(recallL2))*100
    Lfp2 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1L2 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFL2 = 0
    while(g<len(f1L2)):
        somaFL2 = somaFL2+f1L2[g]
        g=g+1
    Lft2 = (somaFL2/len(f1L2))*100

    ##################### rodada 3
    #print "## RODADA 3 ##"

    training_set = baseDividida1+baseDividida2+baseDividida4
    testing_set = baseDividida3

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    testclas = MNB_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    MNBmc3 = sklearn.metrics.confusion_matrix(testgold, testclas)
    MNBa3 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    MNBpp3 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoMNB3 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPMNB3 = 0
    while(g<len(precisaoMNB3)):
        somaPMNB3 = somaPMNB3+precisaoMNB3[g]
        g=g+1
    MNBpt3 = (somaPMNB3/len(precisaoMNB3))*100
    MNBrp3 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallMNB3 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRMNB3 = 0
    while(g<len(recallMNB3)):
        somaRMNB3 = somaRMNB3+recallMNB3[g]
        g=g+1
    MNBrt3 = (somaRMNB3/len(recallMNB3))*100
    MNBfp3 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1MNB3 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFMNB3 = 0
    while(g<len(f1MNB3)):
        somaFMNB3 = somaFMNB3+f1MNB3[g]
        g=g+1
    MNBft3 = (somaFMNB3/len(f1MNB3))*100
    '''
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    BernoulliNB_classifierRodada2 = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
    print("BernoulliNB_classifier accuracy percent:", BernoulliNB_classifierRodada2*100)
    '''
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    testclas = LogisticRegression_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Rmc3 = sklearn.metrics.confusion_matrix(testgold, testclas)
    Ra3 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Rpp3 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoR3 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPR3 = 0
    while(g<len(precisaoR3)):
        somaPR3 = somaPR3+precisaoR3[g]
        g=g+1
    Rpt3 = (somaPR3/len(precisaoR3))*100
    Rrp3 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallR3 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRR3 = 0
    while(g<len(recallR3)):
        somaRR3 = somaRR3+recallR3[g]
        g=g+1
    Rrt3 = (somaRR3/len(recallR3))*100
    Rfp3 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1R3 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFR3 = 0
    while(g<len(f1R3)):
        somaFR3 = somaFR3+f1R3[g]
        g=g+1
    Rft3 = (somaFR3/len(f1R3))*100

    '''
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    SGDClassifier_classifierRodada2 = nltk.classify.accuracy(SGDClassifier_classifier, testing_set)
    print("SGDClassifier_classifier accuracy percent:", SGDClassifier_classifierRodada2*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    SVC_classifierRodada2 = nltk.classify.accuracy(SVC_classifier, testing_set)
    print("SVC_classifier accuracy percent:", SVC_classifierRodada2*100)
    '''
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    testclas = LinearSVC_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Lmc3 = sklearn.metrics.confusion_matrix(testgold, testclas)
    La3 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Lpp3 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoL3 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPL3 = 0
    while(g<len(precisaoL3)):
        somaPL3 = somaPL3+precisaoL3[g]
        g=g+1
    Lpt3 = (somaPL3/len(precisaoL3))*100
    Lrp3 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallL3 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRL3 = 0
    while(g<len(recallL3)):
        somaRL3 = somaRL3+recallL3[g]
        g=g+1
    Lrt3 = (somaRL2/len(recallL2))*100
    Lfp3 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1L3 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFL3 = 0
    while(g<len(f1L3)):
        somaFL3 = somaFL3+f1L3[g]
        g=g+1
    Lft3 = (somaFL3/len(f1L3))*100

    ############################ rodada 4
    #print "## RODADA 4 ##"

    training_set = baseDividida1+baseDividida2+baseDividida3
    testing_set = baseDividida4

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    testclas = MNB_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    MNBmc4 = sklearn.metrics.confusion_matrix(testgold, testclas)
    MNBa4 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    MNBpp4 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoMNB4 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPMNB4 = 0
    while(g<len(precisaoMNB4)):
        somaPMNB4 = somaPMNB4+precisaoMNB4[g]
        g=g+1
    MNBpt4 = (somaPMNB4/len(precisaoMNB4))*100
    MNBrp4 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallMNB4 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRMNB4 = 0
    while(g<len(recallMNB4)):
        somaRMNB4 = somaRMNB4+recallMNB4[g]
        g=g+1
    MNBrt4 = (somaRMNB4/len(recallMNB4))*100
    MNBfp4 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1MNB4 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFMNB4 = 0
    while(g<len(f1MNB4)):
        somaFMNB4 = somaFMNB4+f1MNB4[g]
        g=g+1
    MNBft4 = (somaFMNB4/len(f1MNB4))*100
    '''
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    BernoulliNB_classifierRodada2 = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)
    print("BernoulliNB_classifier accuracy percent:", BernoulliNB_classifierRodada2*100)
    '''
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    testclas = LogisticRegression_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Rmc4 = sklearn.metrics.confusion_matrix(testgold, testclas)
    Ra4 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Rpp4 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoR4 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPR4 = 0
    while(g<len(precisaoR4)):
        somaPR4 = somaPR4+precisaoR4[g]
        g=g+1
    Rpt4 = (somaPR4/len(precisaoR4))*100
    Rrp4 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallR4 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRR4 = 0
    while(g<len(recallR4)):
        somaRR4 = somaRR4+recallR4[g]
        g=g+1
    Rrt4 = (somaRR4/len(recallR4))*100
    Rfp4 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1R4 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFR4 = 0
    while(g<len(f1R4)):
        somaFR4 = somaFR4+f1R4[g]
        g=g+1
    Rft4 = (somaFR4/len(f1R4))*100

    '''
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    SGDClassifier_classifierRodada2 = nltk.classify.accuracy(SGDClassifier_classifier, testing_set)
    print("SGDClassifier_classifier accuracy percent:", SGDClassifier_classifierRodada2*100)

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    SVC_classifierRodada2 = nltk.classify.accuracy(SVC_classifier, testing_set)
    print("SVC_classifier accuracy percent:", SVC_classifierRodada2*100)
    '''
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    testclas = LinearSVC_classifier.classify_many([fs for (fs, l) in testing_set])
    testgold = [l for (fs, l) in testing_set]
    Lmc4 = sklearn.metrics.confusion_matrix(testgold, testclas)
    La4 = (sklearn.metrics.accuracy_score(testgold, testclas))*100
    Lpp4 = sklearn.metrics.precision_score(testgold, testclas, average=None)*100
    precisaoL4 = sklearn.metrics.precision_score(testgold, testclas, average=None)
    g=0
    somaPL4 = 0
    while(g<len(precisaoL4)):
        somaPL4 = somaPL4+precisaoL4[g]
        g=g+1
    Lpt4 = (somaPL4/len(precisaoL4))*100
    Lrp4 = (sklearn.metrics.recall_score(testgold, testclas, average=None))*100
    recallL4 = sklearn.metrics.recall_score(testgold, testclas, average=None)
    g=0
    somaRL4 = 0
    while(g<len(recallL4)):
        somaRL4 = somaRL4+recallL4[g]
        g=g+1
    Lrt4 = (somaRL4/len(recallL4))*100
    Lfp4 = (sklearn.metrics.f1_score(testgold, testclas, average=None))
    f1L4 = sklearn.metrics.f1_score(testgold, testclas, average=None)
    g=0
    somaFL4 = 0
    while(g<len(f1L4)):
        somaFL4 = somaFL4+f1L4[g]
        g=g+1
    Lft4 = (somaFL4/len(f1L4))*100


    ################# medias
    #print "## MEDIA ##"

    #MULTINOMINAL
    MNBmc = (MNBmc1+MNBmc2+MNBmc3+MNBmc4)/4
    MNBa = (MNBa1+MNBa2+MNBa3+MNBa4)/4
    MNBamax = max([MNBa1, MNBa2, MNBa3, MNBa4])
    MNBamin = min([MNBa1, MNBa2, MNBa3, MNBa4])
    MNBpp = (MNBpp4+MNBpp4+MNBpp4+MNBpp4)/4
    MNBpt = (MNBpt1+MNBpt2+MNBpt3+MNBpt4)/4
    MNBpmax = max([MNBpt1, MNBpt2, MNBpt3, MNBpt4])
    MNBpmin = min([MNBpt1, MNBpt2, MNBpt3, MNBpt4])
    MNBrp = (MNBrp1+MNBrp2+MNBrp3+MNBrp4)/4
    MNBrt = (MNBrt1+MNBrt2+MNBrt3+MNBrt4)/4
    MNBrmax = max([MNBrt1, MNBrt2, MNBrt3, MNBrt4])
    MNBrmin = min([MNBrt1, MNBrt2, MNBrt3, MNBrt4])
    MNBfp = (MNBfp1+MNBfp2+MNBfp3+MNBfp4)/4
    MNBft = (MNBft1+MNBft2+MNBft3+MNBft4)/4
    MNBfmax = max([MNBft1, MNBft2, MNBft3, MNBft4])
    MNBfmin = min([MNBft1, MNBft2, MNBft3, MNBft4])

    '''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(MNBmc, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
    '''

    #REGRESSAO LINEAR
    Rmc = (Rmc1+Rmc2+Rmc3+Rmc4)/4
    Ra = (Ra1+Ra2+Ra3+Ra4)/4
    Ramax = max([Ra1, Ra2, Ra3, Ra4])
    Ramin = min([Ra1, Ra2, Ra3, Ra4])
    Rpp = (Rpp4+Rpp4+Rpp4+Rpp4)/4
    Rpt = (Rpt1+Rpt2+Rpt3+Rpt4)/4
    Rpmax = max([Rpt1, Rpt2, Rpt3, Rpt4])
    Rpmin = min([Rpt1, Rpt2, Rpt3, Rpt4])
    Rrp = (Rrp1+Rrp2+Rrp3+Rrp4)/4
    Rrt = (Rrt1+Rrt2+Rrt3+Rrt4)/4
    Rrmax = max([Rrt1, Rrt2, Rrt3, Rrt4])
    Rrmin = min([Rrt1, Rrt2, Rrt3, Rrt4])
    Rfp = (Rfp1+Rfp2+Rfp3+Rfp4)/4
    Rft = (Rft1+Rft2+Rft3+Rft4)/4
    Rfmax = max([Rft1, Rft2, Rft3, Rft4])
    Rfmin = min([Rft1, Rft2, Rft3, Rft4])

    #SVC LINEAR
    Lmc = (Lmc1+Lmc2+Lmc3+Lmc4)/4
    La = (La1+La2+La3+La4)/4
    Lamax = max([La1, La2, La3, La4])
    Lamin = min([La1, La2, La3, La4])
    Lpp = (Lpp4+Lpp4+Lpp4+Lpp4)/4
    Lpt = (Lpt1+Lpt2+Lpt3+Lpt4)/4
    Lpmax = max([Lpt1, Lpt2, Lpt3, Lpt4])
    Lpmin = min([Lpt1, Lpt2, Lpt3, Lpt4])
    Lrp = (Lrp1+Lrp2+Lrp3+Lrp4)/4
    Lrt = (Lrt1+Lrt2+Lrt3+Lrt4)/4
    Lrmax = max([Lrt1, Lrt2, Lrt3, Lrt4])
    Lrmin = min([Lrt1, Lrt2, Lrt3, Lrt4])
    Lfp = (Lfp1+Lfp2+Lfp3+Lfp4)/4
    Lft = (Lft1+Lft2+Lft3+Lft4)/4
    Lfmax = max([Lft1, Lft2, Lft3, Lft4])
    Lfmin = min([Lft1, Lft2, Lft3, Lft4])
    '''
    print "SVC Linear"
    print "Matriz de confusão: ", Lmc
    print "Acuracia: ", La
    print "Precisão parcial: ", Lpp
    print "Precisão total: ", Lpt
    print "Recall parcial: ", Lrp
    print "Recall total: ", Lrt
    print "F-medida parcial: ", Lfp
    print "F-medida total: ", Lft
    '''


    print(experimento + ':' + str(MNBa) +'\t'+str(Ra)+'\t'+str(La))
    with open(path,mode='w') as csv_file:
        #writer = csv.writer(csv_file)
        csv_file.writelines('Algoritmo'+';'+'Multinominal Naïve-Bayes'+'\n')
        csv_file.writelines('Iteração'+';'+'Acurácia'+';'+'Precisão parcial'+';'+'Precisão total'+';'+'revocação parcial'+';'+'revocação total'+';'+'f-medida parcial'+';'+'f-medida total'+'\n')
        csv_file.writelines('1;'+ str(MNBa1)+';'+str(MNBpp1)+';'+str(MNBpt1)+';'+str(MNBrp1)+';'+str(MNBrt1)+';'+str(MNBfp1)+';'+str(MNBft1)+'\n')
        csv_file.writelines('2;'+ str(MNBa2)+';'+str(MNBpp2)+';'+str(MNBpt2)+';'+str(MNBrp2)+';'+str(MNBrt2)+';'+str(MNBfp2)+';'+str(MNBft2)+'\n')
        csv_file.writelines('3;'+ str(MNBa3)+';'+str(MNBpp3)+';'+str(MNBpt3)+';'+str(MNBrp3)+';'+str(MNBrt3)+';'+str(MNBfp3)+';'+str(MNBft3)+'\n')
        csv_file.writelines('4;'+ str(MNBa4)+';'+str(MNBpp4)+';'+str(MNBpt4)+';'+str(MNBrp4)+';'+str(MNBrt4)+';'+str(MNBfp4)+';'+str(MNBft4)+'\n')
        csv_file.writelines('=================='+'\n')
        csv_file.writelines('Total'+'\n')
        csv_file.writelines('Média;'+ str(MNBa)+';'+str(MNBpp)+';'+str(MNBpt)+';'+str(MNBrp)+';'+str(MNBrt)+';'+str(MNBfp)+';'+str(MNBft)+'\n')
        csv_file.writelines('Máximo;'+ str(MNBamax)+""+';'+str(MNBpmax)+""+';'+str(MNBrmax)+""+';'+str(MNBfmax)+'\n')
        csv_file.writelines('Mínimo;'+ str(MNBamin)+""+';'+str(MNBpmin)+""+';'+str(MNBrmin)+""+';'+str(MNBfmin)+'\n')
        csv_file.writelines('=================='+'\n')
        csv_file.writelines('Algoritmo'+';'+'Regressão Linear'+'\n')
        csv_file.writelines('Iteração'+';'+'Acurácia'+';'+'Precisão parcial'+';'+'Precisão total'+';'+'revocação parcial'+';'+'revocação total'+';'+'f-medida parcial'+';'+'f-medida total'+'\n')
        csv_file.writelines('1;'+ str(Ra1)+';'+str(Rpp1)+';'+str(Rpt1)+';'+str(Rrp1)+';'+str(Rrt1)+';'+str(Rfp1)+';'+str(Rft1)+'\n')
        csv_file.writelines('2;'+ str(Ra2)+';'+str(Rpp2)+';'+str(Rpt2)+';'+str(Rrp2)+';'+str(Rrt2)+';'+str(Rfp2)+';'+str(Rft2)+'\n')
        csv_file.writelines('3;'+ str(Ra3)+';'+str(Rpp3)+';'+str(Rpt3)+';'+str(Rrp3)+';'+str(Rrt3)+';'+str(Rfp3)+';'+str(Rft3)+'\n')
        csv_file.writelines('4;'+ str(Ra4)+';'+str(Rpp4)+';'+str(Rpt4)+';'+str(Rrp4)+';'+str(Rrt4)+';'+str(Rfp4)+';'+str(Rft4)+'\n')
        csv_file.writelines('=================='+'\n')
        csv_file.writelines('Total'+'\n')
        csv_file.writelines('Média;'+ str(Ra)+';'+str(Rpp)+';'+str(Rpt)+';'+str(Rrp)+';'+str(Rrt)+';'+str(Rfp)+';'+str(Rft)+'\n')
        csv_file.writelines('Máximo;'+ str(Ramax)+""+';'+str(Rpmax)+""+';'+str(Rrmax)+""+';'+str(Rfmax)+'\n')
        csv_file.writelines('Mínimo;'+ str(Ramin)+""+';'+str(Rpmin)+""+';'+str(Rrmin)+""+';'+str(Rfmin)+'\n')
        csv_file.writelines('=================='+'\n')
        csv_file.writelines('Algoritmo'+';'+'SVC Linear'+'\n')
        csv_file.writelines('Iteração'+';'+'Acurácia'+';'+'Precisão parcial'+';'+'Precisão total'+';'+'revocação parcial'+';'+'revocação total'+';'+'f-medida parcial'+';'+'f-medida total'+'\n')
        csv_file.writelines('1;'+ str(La1)+';'+str(Lpp1)+';'+str(Lpt1)+';'+str(Lrp1)+';'+str(Lrt1)+';'+str(Lfp1)+';'+str(Lft1)+'\n')
        csv_file.writelines('2;'+ str(La2)+';'+str(Lpp2)+';'+str(Lpt2)+';'+str(Lrp2)+';'+str(Lrt2)+';'+str(Lfp2)+';'+str(Lft2)+'\n')
        csv_file.writelines('3;'+ str(La3)+';'+str(Lpp3)+';'+str(Lpt3)+';'+str(Lrp3)+';'+str(Lrt3)+';'+str(Lfp3)+';'+str(Lft3)+'\n')
        csv_file.writelines('4;'+ str(La4)+';'+str(Lpp4)+';'+str(Lpt4)+';'+str(Lrp4)+';'+str(Lrt4)+';'+str(Lfp4)+';'+str(Lft4)+'\n')
        csv_file.writelines('=================='+'\n')
        csv_file.writelines('Total'+'\n')
        csv_file.writelines('Média;'+ str(La)+';'+str(Lpp)+';'+str(Lpt)+';'+str(Lrp)+';'+str(Lrt)+';'+str(Lfp)+';'+str(Lft)+'\n')
        csv_file.writelines('Máximo;'+ str(Lamax)+""+';'+str(Lpmax)+""+';'+str(Lrmax)+""+';'+str(Lfmax)+'\n')
        csv_file.writelines('Mínimo;'+ str(Lamin)+""+';'+str(Lpmin)+""+';'+str(Lrmin)+""+';'+str(Lfmin)+'\n')

print('Suavização+Stopwords')

print('\t\t\t NB \t\t RL \t\t SVC')
executar('experimento5/','sce/unbalanced/colecao_dourada_2_class_unbalanced.csv',False)
executar('experimento5/','sce/unbalanced/colecao_dourada_3_class_unbalanced.csv',False)
executar('experimento5/','sce/balanced/colecao_dourada_2_class_balanced.csv',False)
executar('experimento5/','sce/balanced/colecao_dourada_3_class_balanced.csv',False)
executar('experimento6/','sce/unbalanced/colecao_dourada_2_class_unbalanced.csv',True)
executar('experimento6/','sce/unbalanced/colecao_dourada_3_class_unbalanced.csv',True)
executar('experimento6/','sce/balanced/colecao_dourada_2_class_balanced.csv',True)
executar('experimento6/','sce/balanced/colecao_dourada_3_class_balanced.csv',True)

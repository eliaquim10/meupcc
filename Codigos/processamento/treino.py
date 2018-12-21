from recursos2  import readBase, wordFeature, find_features, remocao_acento, remocao_pontos,  remocao_de_stopwords, stemmiza
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sklearn
import nltk
import csv
import random

# nltk.bigram()
acento = True
remocao_ponto = True ## manteve um certo equilibrio com a pontuação e sem o bigram 81%
remocao_de_stopword = True ## manteve um certo equilibrio com a pontuação e sem o bigram 80-83%

stem = False

def execucao(nome_base,pre_processamento):
    # lendo a base 
    base = readBase(nome_base)    
    tamBase = len(base)
    # aplicando as tecnicas de pre-processamento
    if (pre_processamento):
        # remoção de acentos da base
        if (acento):
            base = remocao_acento(base)
            
        # tokenizando os comentarios
        i=0
        documents = []
        tknzr = nltk.tokenize.TweetTokenizer()
        while (i<tamBase):
            w = tknzr.tokenize(base[i][0])
            conteudoLista = (w,base[i][1])
            documents.append(conteudoLista)
            i += 1
        #remoção das urls dos comentarios
        # remoção dos pontos da base
        if (remocao_ponto):
            documents = remocao_pontos(documents)
        # remoção das stopwords dos comentarios
        if (remocao_de_stopword):
            documents = remocao_de_stopwords(documents)
        # stemming das palavras  
        if (stem):
            documents =  stemmiza(documents)
    else:
         # tokenizando os comentarios
         i=0
         documents = []
         tknzr = nltk.tokenize.TweetTokenizer()
         while (i<tamBase):
             w = tknzr.tokenize(base[i][0])
             conteudoLista = (w,base[i][1])
             documents.append(conteudoLista)
             i += 1

    random.shuffle(documents)

    # coletando todas os tokens 
    word_features = wordFeature(documents)


    # transformando a base em bag-of-word
    featuresets = [(find_features(rev,word_features), category) for (rev, category) in documents]

    # dividindo a base de treino e a de teste
    kfold = 4
    baseInteira = featuresets
    tamT = len(featuresets)
    divisor = tamT//kfold

    training_set = featuresets[0:(divisor*3)]
    testing_set = featuresets[(divisor*3):tamT]

    MNBa = [ ]
    Ra = [ ]
    i = 0
    while i<kfold:
        # print('rodada : '+str(i+1) )

        # fazendo o treinamento  do classificador Naive Bayes

        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(training_set)


         # fazendo o teste do classificador

        testclas = MNB_classifier.classify_many([fs for (fs, l) in testing_set])
        testgold = [l for (fs, l) in testing_set]
        matrix = sklearn.metrics.confusion_matrix(testgold, testclas)

        # calculo das metricas
        
        #acuracia
        MNBa.append( (sklearn.metrics.accuracy_score(testgold, testclas))*100)

        # fazendo o treinamento  do classificador Regressao Logistica

        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)

        # fazendo o teste do classificador 

        testclas = LogisticRegression_classifier.classify_many([fs for (fs, l) in testing_set])
        testgold = [l for (fs, l) in testing_set]

        # calculo das metricas

        #acuracia
        Ra.append((sklearn.metrics.accuracy_score(testgold, testclas))*100) 

        i+=1
    # resultado do k-fold
    MNBm = (sum(MNBa))/(len(MNBa))
    Rm = (sum(Ra))/(len(Ra))
    return MNBm,Rm

N_gram = False
j=0
m =0
l = 10
MNBs1 = 0
Rs1 = 0

MNBs2 = 0
Rs2 = 0


with open('resultado_taia_palavra_aleatorio_ca.csv',mode='w') as csv_file:
    csv_file.writelines(';Multinomial Naive Bayes;;Regressão Linear;;\n')
    csv_file.writelines('Pre-processamento;Sem;Com;Sem;Com;\n')
    print('\t\t\t\tMultinomial Naive Bayes\t\tRegressão Linear')
    print('Pre-processamento\t\tSem\t\tCom\t\tSem\t\tCom')
    while (j<l) :
        # MNBm1, Rm1, Sm1 = execucao('dataset.csv',False)
        # MNBm2, Rm2, Sm2 = execucao('dataset.csv',True)
        MNBm1,Rm1 = execucao('colecao_dourada_2_class_unbalanced.csv',False)
        MNBm2,Rm2 = execucao('colecao_dourada_2_class_unbalanced.csv',True)
        MNBs1+=MNBm1
        Rs1 += Rm1

        MNBs2+=MNBm2
        Rs2 += Rm2

        print(str(j+1)+'\t\t\t\t'+str(MNBm1)+'\t'+str(MNBm2)+'\t'+str(Rm1)+'\t'+str(Rm2))
        csv_file.writelines(str(j+1)+';'+str(MNBm1)+';'+str(MNBm2)+';'+str(Rm1)+';'+str(Rm2)+'\n')
        j+=1
    MNBs1/=l
    Rs1 /= l

    MNBs2/=l
    Rs2 /= l

    print('media\t\t\t\t'+str(MNBs1)+'\t'+str(MNBs2)+'\t'+str(Rs1)+'\t'+str(Rs2))
    csv_file.writelines('media;'+str(MNBs1)+';'+str(MNBs2)+';'+str(Rs1)+';'+str(Rs2)+'\n')
 # ''''''

# coding=utf-8
# from Util import readBase
import csv

import nltk
from unicodedata import normalize
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

def readBase(csvFile = str):##ler o arquivo
    based = []
    with open( csvFile, newline='\n') as csvfile:
        # print (csvfile)
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        # print (reader)
        

        for row in reader:
            try:
                #print(row)
                #sem caracteres especiais
                temp1 = row[2].lower()
                temp2 = row[1]
                # print(temp1)

                #com caracteres especiais
                #temp1 = row[14].lower()
                #temp2 = row[0].lower()

                based.append(tuple([temp1,temp2]))
            except IndexError:
                pass   
        return based
def remocaoacento(base):
    vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
    base2 = []
    for row in base:
        temp1 = ''
        for w in row[0]:
            if (w not in vida):
                temp1 +=w
        base2.append(tuple([temp1,row[1]]))
    return base2
def remocaopontos(documents):
    vida = '.,!?<>][*()+-;:'
    documentos = []

    for row in documents:
        linha = []
        for r in row[0]:
            # temp1 = ''
            if (r not in vida):
                linha.append(r)
        documentos.append(tuple([linha,row[1]]))
    return documentos
def remocao_de_stopwords(documents):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    documentos = []
    for d in documents:
        linha = []
        for w in d[0]: 
            if not w in stopwords:
                linha.append(w)
        documentos.append(tuple([linha,d[1]]))
    return documentos

def tokenizacao(base):
    tknzr = nltk.tokenize.TweetTokenizer()
    documents = []
    i=0
    tamBase = len(base)
    while (i<tamBase):
        conteudoLista = (tknzr.tokenize(base[i][0]),base[i][1])
        documents.append(conteudoLista)
        i += 1
    return documents
def Stemmiza(documents):
    documentos = []
    ps = nltk.stem.RSLPStemmer()
    for row in documents:
        linha = []
        for r in row[0]:
            linha.append(ps.stem(r))
        documentos.append(tuple([linha,row[1]]))
    return documentos

nomeBase = 'colecao_dourada_2_class_balanced.csv'
base = readBase(nomeBase)
base = remocaoacento(base)
tamBase = len(base)
i=0
all_words = []
documents = []



#vericacao com acento
documents = tokenizacao(base)
# tokenização

k=0
l=len(documents)
while (k<l):
    m=len(documents[k][0])
    n=0
    while(n<m):
        all_words.append(documents[k][0][n])
        n+=1
    k += 1

print('quantidade de tokens: '+str(len(all_words)))

documents = remocao_de_stopwords(documents)
documents = remocaopontos(documents)
# remoção de stopwords
documents = Stemmiza(documents)
# Stemming
#print ([w for w in all_words])

print(all_words)
print('quantidade de tokens: '+str(len(all_words)))
for w in documents:
    print(w[0])

polaridade = [w[1] for w in documents]
dataset = [' '.join(str(i) for i in w[0]) for w in documents]
dados_treino, dados_val, pols_treino, pols_val = train_test_split(dataset, polaridade, test_size=0.25)


print("**********************")
print("Conjunto de Treinamento")
print(dados_treino)
print("Polaridades do Conjunto de Treinamento")
print(pols_treino)
print("\n---------------------------------------------\n")

#Print do conjunto de validaĂ§ĂŁo e suas respectivas polaridades
print("Conjunto de Validação")
print(dados_val)
print("Polaridades do Conjunto de Validação")
print(pols_val)

bag = CountVectorizer()

#Método fit_transform: 
#fit = cria e aprende a bag
#transform = cria a matriz termo-documento
bag_treino = bag.fit_transform(dados_treino)

#Printa o vocabulário da bag-of-words   
print("Vocabulário da bag-of-words")
print(sorted(bag.vocabulary_))
print("\n---------------------------------------------\n")

# Printa a bag-of-words    
print("Bag-of-words de treino")
print(bag_treino)
print("\n---------------------------------------------\n")



'''

#all_words = nltk.LidstoneProbDist(nltk.FreqDist(all_words), 0.1)
# all_words = nltk.FreqDist(nltk.FreqDist(w.lower() for w in all_words if w not in stopwords))

# suavização/smoothing


# all_words = nltk.PorterStemmer().stem(all_words)


#print all_words.samples()
 '''
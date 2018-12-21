# ler a base 
import csv
import nltk


def readBase(csvFile = str):
    base = []
    with open(csvFile) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            try:
                temp1 = row[2].lower()
                temp2 = row[1].lower()
                base.append(tuple([temp1, temp2]))
            except IndexError:
                pass
    return base
#cria um dicionario de dados, ou seja, procura todas as palavras existentes nos comentarios
def wordFeature(documents):
    dicionario = []
    for w in documents:
        for q in w[0]:
            # if(not q in dicionario):
            dicionario.append(q)
    return dicionario

def find_features(document,word_features):
    features = {}
    i=0
    l = len(word_features)
    while (i<l):
        features[str(i)] = (word_features[i] in document)
        i+=1
    return features

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
    #(lambda x,g=lambda x, f=lambda x: 3*x-x:f(x**x):g(x))(5)


def stemmiza(documents):
    documentos = []
    ps = nltk.stem.RSLPStemmer()
    for row in documents:
        linha = []
        for r in row[0]:
            linha.append(ps.stem(r))
        documentos.append(tuple([linha,row[1]]))
    return documentos

def remocao_acento(base):
    vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
    base2 = []
    for row in base:
        temp1 = ''
        for w in row[0]:
            if (w not in vida):
                temp1 +=w
        base2.append(tuple([temp1,row[1]]))
    return base2

def remocao_pontos(documents):
    vida = ['.',',','!','?','<','>',']','[','*','(',')','+','-',';',':']
    documentos = []
    for row in documents:
        linha = []
        for r in row[0]:
            # temp1 = ''
            if (r not in vida):
                linha.append(r)
        documentos.append(tuple([linha,row[1]]))
    return documentos
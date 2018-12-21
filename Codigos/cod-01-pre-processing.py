# -*- coding: utf-8 -*-
def readBase(csvFile = str):##ler o arquivo
    based = []
    with open( csvFile, newline='\n') as csvfile:
        # print (csvfile)
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        # print (reader)

        for row in reader:
            try:
                print(row)
                #sem caracteres especiais
                temp1 = unicode(row[2].lower())

                #com caracteres especiais
                #temp1 = row[14].lower()
                #temp2 = row[0].lower()

                based.append(tuple([temp1, temp2]))
            except IndexError:
                pass
        return based

############## tokenization
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
tknzr = TweetTokenizer()

texto = "@EikeBatiiista Como podemos ajudar o senhor? ;-)"

print(sent_tokenize(texto))

print(word_tokenize(texto))

print(tknzr.tokenize(texto))

'''

############## stopwords

from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

## build stop word list for brazilian portuguese
pt_br_stop_words = set(stopwords.words('portuguese'))
#print(pt_br_stop_words)

pt_br_stop_words.update('@', '#','.',':')
#print(pt_br_stop_words)

texto = "Exemplo de tokenização de um tweet. #DarBobeira É achar que o BB envia e-mail pedindo a atualização dos seus dados cadastrais ou com resultado de sorteio: http://t.co/mufifteIdW."
word_tokens = word_tokenize(texto)

print(word_tokenize(texto))

texto_filtrado = [w for w in word_tokens if not w in pt_br_stop_words]
print(texto_filtrado)


############## stemming
from nltk.stem import RSLPStemmer

## Stemmer para a língua portuguesa
stemmerPT = RSLPStemmer()

palavras = ["comprei","compramos","comprou","comprado","compraria"]

for w in palavras:
    print(stemmerPT.stem(w))


############## pos tagger

import pickle as pickle
from nltk.tokenize import sent_tokenize

tagger = pickle.load(open('taggerUnigramPT.pickle'))

texto = "Exemplo de tokenização de um tweet. #DarBobeira É achar que o BB envia e-mail pedindo a atualização dos seus dados cadastrais ou com resultado de sorteio: http://t.co/mufifteIdW."
word_tokens = word_tokenize(texto)

for s in sent_tokenize:
    print(tagger.tag(s))

'''

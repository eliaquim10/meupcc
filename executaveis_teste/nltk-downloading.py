import nltk

import random

from Util import readBase,trata
lista = readBase('colecao_dourada_2_class_unbalanced.csv')

trata(lista,0.7)

# if (not nltk.download()):
#     exit()
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:05:18 2017

@author: Usuário
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

#Armazenamento de documentos (frases) na lista dataset
dataset = [("@EikeBatiiista Como podemos ajudar o senhor? ;-)"),
           ("Alô, @BancodoBrasil! O app de vocês para Android tá muito louco! Entrei na minha conta e vi o extrato de OUTRO CLIENTE! O.o"),
           ("que deus te de em dobro, toda a velocidade de internet q vc me deu @vivo"),
           ("#DarBobeira É achar que o BB envia e-mail pedindo a atualização dos seus dados cadastrais ou com resultado de sorteio: http://t.co/mufifteIdW.")]

print(dataset)

#Armazenamento das polaridades de cada documento (frase) na lista polaris
polaridade = [0, -1, 1, 0]

#DivisĂŁo dos dados das listas dataset e polaris em conjuntos de treinamento e validaĂ§ĂŁo
dados_treino, dados_val, pols_treino, pols_val = train_test_split(dataset, polaridade, test_size=0.25)

#Print do conjunto de treinamento e suas respectivas polaridades
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

#Cria uma instĂ˘ncia para a bag-of-words   
bag = CountVectorizer()

#Método fit_transform: 
#fit = cria e aprende a bag
#transform = cria a matriz termo-documento
bag_treino = bag.fit_transform(dados_treino)

#Printa o vocabulário da bag-of-words   
print("Vocabulário da bag-of-words")
print(sorted(bag.vocabulary_))
print("\n---------------------------------------------\n")

#Printa a bag-of-words    
print("Bag-of-words de treino")
print(bag_treino)
print("\n---------------------------------------------\n")


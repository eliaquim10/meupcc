from Recurso import *
from sklearn.metrics import cohen_kappa_score


#para a classificacao precisa apenas está na coluna 1 a classificacao do primeiro e do segundo na terceira coluna

#Nome da base e ler ela


nomeBase = 'colecao_dourada.csv'
# nomeBase = 'colecao_dourada_teste.csv'
#nomeBase = 'base.csv'
base = read_date_base( nomeBase )
print ('nome da base : '+nomeBase)

#categorias das opiniões exceção do 'total' que precisa ficar fixo
# categorias = ['Negativo','Neutro','Positivo','Total']
categorias = ['1','2','3','4','5','6','Total']

#transformando as categorias em todos de letra minuscula
l = len(categorias)-1
c = categorias[0:l]
c = [ w.lower() for w in c]

#criando o cabeçalho que vai ser mostrado na tela
categoria = '\t\t'
for w in categorias:
	categoria += w+'\t'

#criando a matriz de confucao
matrizconfucao = matriz_de_confucao( base ,c)

matrizdados = matriz_avalia( base , c)
tamanho = len(base) - 1

#avaliador 1 na segunda coluna
dados1 = cal_avaliar1(matrizconfucao)
print(str(dados1))
#avaliador 2 na terceira coluna
dados2 = cal_avaliar2(matrizconfucao)
print(str(dados2))


print ()
print ()
print ("Coeficiente Kappa/Cohen's: " + str( calculo_kappa( matrizconfucao ) ) + "%")

v = vector(base)
print ()
print (cohen_kappa_score(v[0],v[1]))
print ()
print ("Coeficiente Kappa/Fleiss': " + str( compute_kappa( matrizdados ) ) + "%")
print ()


dados1 = cal_percent(dados1,tamanho,c)
dados2 = cal_percent(dados2,tamanho,c)

avaliador1 = 'Avaliador 1   '
total = 0
for dado in dados1:
	total+=dado
	avaliador1 += str(dado) + '%\t'
avaliador1 += str(tamanho)

avaliador2 = 'Avaliador 2   '


for dado in dados2:	
	avaliador2 += str(dado) + '%\t'
avaliador2 += str(tamanho)

print()
print (categoria)
print (avaliador1)
print (avaliador2)
#print 'Kappa'

from Recurso import *

# nomeBase = 'base.csv'
nome_base = 'dados_Corrigido.csv'

base = read_date_base( nome_base )

i=1
l = len(base)
menorQuantword =100000
mediaWord = 0
maiorQuantword = 0  
totalQuantword = 0
palavras = []
while i < l:
	palavras.append(str(base[i][2]).split())
	i+=1

i=0
while i<l:
	try:
		# print(str(i)+' : '+str(palavras[i]))
		aux = len(palavras[i])
		totalQuantword += aux
		if (menorQuantword > aux):
			menorQuantword = aux
		if (maiorQuantword < aux):
			maiorQuantword = aux
	except Exception as e:
		pass	
	i+=1

mediaWord = totalQuantword/l

print ('Media: '+str(mediaWord))
print ('Total: '+str(totalQuantword))
print ('Maior: '+str(maiorQuantword))
print ('Menor: '+str(menorQuantword))



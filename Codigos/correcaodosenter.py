#correcao dos \n 
arq = open('dados_tabela.csv', 'r')
texto = arq.read()

palavras = ['Denunciar','denuncias','denuncia']
errado =0
j=0
indice=0
dados =''
for i in texto:
	if(i=='\n'):
		j=indice
		s=''
		while(j>indice-10):
			j-=1
			if((texto[j]!=' ') and (texto[j]!=';')):
				s +=texto[j]
		s=s[::-1]
		print s
		if not (s in palavras):
			errado+=1
			dados+= ' '
		else:
			dados += i
	else:
		dados += i
	indice += 1

arq.close()

arq = open('dados_tabela.csv', 'w')
arq.write(dados)
arq.close
print errado
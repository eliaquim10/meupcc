#correcao dos virgula 
from datetime import datetime
def verfica(s,w):
	try:
		datetime_object = datetime.strptime(s, w)
		# print 'entrou'
		return True
	except Exception:
		return False
def dia(d):
	return ((d>=1) and (d<=31))
def mes(m):
	return m in meses
def ano(a):
	return ((a>=1) and (a<=3001))

errado = 1

while(errado!=0):
	errado = 0
	arq = open('dados_tabela3.csv', mode='r',encoding='utf-8')
	texto = arq.read()


	j=0
	w=0
	indice=0
	dados =''
	add=False
	meses=['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
	for i in texto:
		if(i=='\n'):
			w=0
		if(i==';'):
			w+=1
		if((w==3) and (i==';')):
			j=indice
			s=''		
			while(j<indice+14):
				j+=1
				if(texto[j]==';'):
					break
				# if(texto[j]!=' '):
				s += texto[j]
			
			# s=s[::-1]
			# print s
			p = []
			for q in s.split():
				if (q!=' '):
					p.append(q)
			# p = [q for q in s.split() if (q!=' ')]
			
			# print palavra
			#print str(p)
			
			# print str(dia(p[0]))+str(mes(p[1]))+str(ano(p[2]))
			try:
				add = not (dia(int(p[0])) and mes(p[1]) and ano(int(p[2])))
			except Exception:
				# print (str(p))
				add = True
			# print add
			#verfica(p[1],' %d %b %Y')
			if(add):
				# print (str(p))
				# print (s)
				dados += ' '
				errado += 1
			else:
				dados += i

			# if(errado==2):
			# 	print (s)
			# 	print ('tamanho' + ' : ' + str(len(s)))
		else:
			dados += i
		indice+=1	

	arq.close()
	print (errado)
	arq = open('dados_tabela3.csv',mode='w',encoding='utf-8')
	arq.write(dados)
	arq.close()
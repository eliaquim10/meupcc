import csv

def read_base(csvFile = str):
	base = []
	p=0
	n=0
	N=0
	with open(csvFile, newline = '\n') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in reader:
			try:
				temp1 = row[0]
				temp2 = row[1]
				if(temp2=='1'):
					p+=1
				if(temp2=='2'):
					n+=1
				if(temp2=='4'):
					N+=1
				temp3 = row[2]
				base.append([temp1, temp2, temp3])
			except Exception:
				pass
	return p,n,N,base

p,n,N,base = read_base('colecao_dourada.csv')

nome_base = 'colecao_dourada_2_class_unbalanced.csv'
print('nome base:'+nome_base)
arq = open(nome_base,mode='w',encoding='utf-8')

s = ''
k=10000
# if(p<k):
# 	k=p
# if(n<k):
# 	k=n
# if(N<k):
# 	k=N
print('Limite: '+str(k))
print('Positivo: '+str(p))
print('Negativo: '+str(n))
print('Neutro: '+str(N))
a=0
b=0
c=0

for dado in base:
	if((dado[1]=='1' or dado[1]==1) and (a<k)):
		s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
		a+=1
	if((dado[1]=='2' or dado[1]==2) and (b<k)):
		s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
		b+=1
	# if((dado[1]=='4' or dado[1]==4) and (c<k)):
	# 	s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
	# 	c+=1

arq.write(s)
arq.close

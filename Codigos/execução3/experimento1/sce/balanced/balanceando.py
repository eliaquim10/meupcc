import csv

def read_base(csvFile = str):
	base = []
	i=0
	with open(csvFile, newline = '\n') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='|')
		for row in reader:
			try:
				#sem caracteres especiais
				# print(str(row))
				if(i!=0):
					temp1 = str(row[0])
					temp2 = str(row[1])
					temp3 = str(row[2])
					# print(temp1)
					base.append(tuple([temp1, temp2, temp3]))
				else:
					c=row[0]
					i=1
			except Exception:
				pass
		return c,base

c,dados = read_base('colecao_dourada.csv')
i = 0

arq = open('colecao_dourada_2_class_balanced.csv',mode='w',encoding='utf-8')
s = ''
k=99
a=0
b=0
d=0

for dado in dados:
	# print(str(dado))
	if(i!=0):
		if((dado[1]=='1' or dado[1]==1) and (a<k)):
			s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
			a+=1
		if((dado[1]=='2' or dado[1]==2) and (b<k)):
			s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
			b+=1
		# if((dado[1]=='4' or dado[1]==4) and (d<k)):
		# 	s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
		# 	d+=1
	# if((dado[1]=='1' or dado[1] =='2' or dado[1]=='3' or dado[1]==1 or dado[1]==2 or dado[1]==3) and (i!=0)):
		# print('entrou')
	if(i==0):
		s += c + '\n'
		s += dado[0]+';'+dado[1]+';'+dado[2]+'\n'
		i=1

arq.write(s)
arq.close

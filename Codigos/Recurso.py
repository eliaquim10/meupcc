# coding=utf-8
# from Util import readBase
import csv

import math


def read_date_base(csvFile = str):##ler o arquivo
	based = []
	a=0
	with open( csvFile) as csvfile:
	# with open( csvFile, newline='\n') as csvfile:
		#print csvfile
		reader = csv.reader(csvfile , delimiter=';', quotechar='|')
		# print (reader)
		for row in reader:
			try:
				# row = reader.next() 
				# while(row!=None):
				#sem caracteres especiais
				#temp2 = unicode(row[14].lower(), errors='ignore').encode('utf-8')
				#print(temp2)
				#temp1 = unicode(row[0].lower(), errors='ignore')
				#temp3 = unicode(row[1].lower(), errors='ignore')
				#temp4 = unicode ( row[2].lower ( ), errors='ignore' )

				enter = True
				temp1 = str(row[0])
				temp2 = str(row[1]) 
				temp3 = str(row[2])
				temp4 = (str(row[3]) if(row[3]!=None) else '')

				#com caracteres especiais
				#temp1 = row[14].lower()
				#temp2 = row[0].lower()
				if(enter):
					based.append([temp1, temp2, temp3,temp4])
				# row = reader.next()
			except Exception:
				pass
		return based

def matriz_de_confucao(base,categorias):
	# 0 positivc, 1 neutro, 2 negativo
	i=0
	l = len(categorias)
	matriz=[]
	j=0
	while (i < l):
		matriz.append([0]*l)
		i+=1
	i=1

	tamanho = len(base)
	count =0
	while i < tamanho:
		try:
			c=0
			for w in categorias:
				if(equals(base[i][1],w)): 
					a = c  
				if(equals(base[i][2],w)): 
					b = c
				c+=1
			count+=1
			matriz[b][a]+=1
		except Exception:
			pass
		i+=1
	return matriz

def equals(frase1 = str,frase2 = str):
	b = 0
	aux = 0
	tamanho = len(frase1)

	if (not (tamanho == len(frase2))):
		return 0
	while (b < tamanho):
		if frase1[b] == frase2[b]:
			aux+=1
		else:
			return 0
		b+=1
	if b == aux :
		return 1
	return 0

def calculo_kappa(matriz):
	total=0.
	diagonal=0.
	proColLin = 0.
	i=0
	
	tamanho = len(matriz)
	while i < tamanho:
		coluna=0
		j=0
		while j < tamanho:
			coluna += matriz[j][i] 
			j+=1
		linha = sum(matriz[i])
		proColLin += (linha)*(coluna)
		i+=1
	i=0
	while i < tamanho:
		for w in matriz[i]:
			total += w
		diagonal += matriz[i][i]
		i += 1
	x = (total * diagonal) - proColLin
	y = math.pow(total,2) - proColLin

	#x = x*10000

	#IKappa = int(x / y)
	#return float((IKappa))/100
	return cal_copercent(x,y)



def compute_kappa(mat): #Fleiss'
	n = sum(mat[0])  # PRE : every line count must be equal to n
	N = len ( mat )
	k = len ( mat[0] )

	#if DEBUG:print n, "raters."print N, "subjects."print k, "categories."

	# Computing p[]
	p = [0.0] * k
	for j in range ( k ):
		p[j] = 0.0
		for i in range ( N ):
			p[j] += mat[i][j]
		p[j] /= N * n
	#if DEBUG: print "p =", p

	# Computing P[]
	P = [0.0] * N
	for i in range ( N ):
		P[i] = 0.0
		for j in range ( k ):
			P[i] += mat[i][j] * mat[i][j]
		P[i] = (P[i] - n) / (n * (n - 1))
	#if DEBUG: print "P =", P

	# Computing Pbar
	Pbar = sum ( P ) / N
	#if DEBUG: print "Pbar =", Pbar

	# Computing PbarE
	PbarE = 0.0
	for pj in p:
		PbarE += pj * pj
	#if DEBUG: print "PbarE =", PbarE

	#kappa = int(((Pbar - PbarE)*10000) / (1 - PbarE))
	kappa = ( ((Pbar - PbarE) ) / (1 - PbarE) )
	#if DEBUG: print "kappa =", kappa

	kappa = cal_copercent(kappa,1)

	#return float(kappa)/100
	return kappa



def matriz_avalia(base,categorias):
	# positivo neutro negativo
	matriz= []
	l = len(categorias)
	aux = [0]*l
	
	i=1
	while i < len(base):
		j=0
		for w in categorias:
			if(base[i][1] == w):
				aux[j] += 1
			if(base[i][2] == w):
				aux[j] += 1
			j+=1
		matriz.append(aux)
		aux = [0]*l
		i+=1
	# print (str(matriz))
	return matriz

def compute_kappa2(matriz): #Cohen's
	total = 0
	kappa=0.
	pa = 0.
	pe = 0.
	somaquadro=0 # soma dos quadrados
	i=0
	tamanho = len(matriz) - 1
	tamPart= 0
	for w in matriz[0]:
		tamPart+=w
	# print tamPart
	while i < tamanho+1:
		for w in matriz[i]:
			total += w
			somaquadro += w*w
		i += 1
	x = ((somaquadro) - (total*tamanho))
	y = (tamanho*total*(total-1))
	#print (xrange(total))
	#print y
	pa = x / y
	return pa
	# print (pa)

def cal_percent(dados,tamanhobase,categorias):
	i = 0
	l = len(dados)
	date = [0]*l
	aux = []
	x = 0
	while i < l:
		#x = int((dados[i] * 10000)/(tamanhobase))
		#dados[i] = float(x)/ 100
		date[i] = cal_copercent(dados[i],tamanhobase)
		i+=1
	return date

def cal_copercent(x, y):
	try:
		aux = int ( (x * 10000) / (y) )
		result = (float ( aux )) / 100
		return result
	except Exception:
		return 0
		pass

def cal_rel(matriz):

	correto = 0
	total = 0
	c = len(matriz)
	l = len(matriz[0])
	for i in range (c) :
		for j in range (l) :
			if matriz[i][j]==2:
				correto+=2

def cal_avaliar1(matrizconfucao):
	l = len(matrizconfucao)
	a = [0.00]*l
	i = 0
	while i<l:
		j=0
		while(j<l):
			a[j] += matrizconfucao[i][j]
			j+=1
		i+=1
	return a

def cal_avaliar2(matrizconfucao):
	l = len(matrizconfucao)
	a = [0.00]*l
	j=0
	
	while(j<l):
		i = 0
		while i<l:
			a[j] += matrizconfucao[j][i]
			i += 1
		j+=1
	return a
def vector(base):
	v1 = []
	v2 = []
	for w in base:
		v1.append(w[1])
		v2.append(w[2])
	return [v1,v2]
	

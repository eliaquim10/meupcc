# E1.1) Faça um programa que apresente no console o seu nome completo.

print ('Eliaquim Moreira do Nascimento')
# E2.1) Faça um programa com três variáveis para armazenar seu nome, sobrenome e idade.
# Apresente no console seu nome completo e sua idade convertida para String.

nome = 'Eliaquim'
sobrenome = 'Moreira do Nascimento'
idade = 22
print('nome :' + nome + ' ' + sobrenome)
print('idade :' + str(idade) +' anos')

# E3.1) Faça um programa que receba duas Strings como entrada (seu nome e sobrenome, por
# exemplo) e apresente no console as Strings concatenadas.

nome = input('digite seu primeiro nome: ')
sobrenome = input('digite seu sobrenome: ')
print('nome :' + nome + ' ' + sobrenome)

# E3.2) Faça um programa que receba dois números de entrada e calcule as operações de adição,
# subtração, multiplicação, divisão, divisão inteira, módulo e exponenciação. Apresente os resultados
# no console.

def adicao(x,y):
    return x+y
def subtracao(x,y):
    return x-y
def multiplicacao(x,y):
    return x*y
def divisao(x,y):
    if( y != 0 ):
        return x/y
    else:
        return 'erro'

def divisaointeira(x,y):
    if( y != 0 ):
        return (x//y)
    else:
        return 'erro'

def modulo(x,y):
    return x%y

def exponenciacao(x,y):
    return x**y

s = input('digite o 1º numero: ')
x = float(s)

s = input('digite o 2º numero: ')
y = float(s)

print (' 1 - adição')
print (' 2 - subtração')
print (' 3 - multiplicação')
print (' 4 - divisão')
print (' 5 - divisão inteira')
print (' 6 - módulo ')
print (' 7 - exponenciação ')
s = input('digite a opção: ')
opcao = int(s) 
if(opcao==1):
    print (' Adição ')
    print ('Resultado: ' + str(adicao(x,y)))
if(opcao==2):
    print (' Subtração ')
    print ('Resultado: ' + str(subtracao(x,y)))
if(opcao==3):
    print (' Multiplicação ')
    print ('Resultado: ' + str(multiplicacao(x,y)))
if(opcao==4):
    print (' Divisão ')
    print ('Resultado: ' + str(divisao(x,y)))
if(opcao==5):
    print (' Divisão inteira ')
    print ('Resultado: ' + str(divisaointeira(x,y)))
if(opcao==6):
    print (' Módulo ')
    print ('Resultado: ' + str(modulo(x,y)))
if(opcao==7):
    print (' Exponenciação ')
    print ('Resultado: ' + str(exponenciacao(x,y)))

# E3.3) Faça um programa que receba como entrada as quatro notas que um aluno obteve. Calcule a
# média aritmética e informe no console se este aluno foi aprovado ou não. Considere 6.0 a média
# para aprovação.

def media(x,y,w,z):
    return (x+y+w+z)/4

s = input('digite o 1º nota: ')
x = float(s)

s = input('digite o 2º nota: ')
y = float(s)

s = input('digite o 3º nota: ')
w = float(s)

s = input('digite o 4º nota: ')
z = float(s)

media = media(x,y,w,z)

if(media>=6.0):
    status = 'Aprovado'
else:
    status = 'Reprovado'

print ('Situação: ' + status + '\nnota: ' +str(media))

# E4.1) Faça um programa que receba um da idade de uma pessoa e apresente no console se esta
# pessoa é maior ou menor de idade.

s = input('digite sua idade: ')
idade = int(s)
if(idade>=18):
    print('maior de idade')
else:
    print('menor de idade')

# E4.2) Faça um programa que receba dois números de entrada. Apresente no console se os números
# são iguais ou diferentes.

s = input('digite o 1º numero: ')
x = float(s)

s = input('digite o 2º numero: ')
y = float(s)

if(x==y):
    print('números são iguais')
else:
    print('números são diferentes')

# E4.3) Faça um programa que receba dois números de entrada. Se os números forem diferentes,
# informe no console qual deles é o maior número. Do contrário, informe que os números são iguais.

s = input('digite o 1º numero: ')
x = float(s)

s = input('digite o 2º numero: ')
y = float(s)

if(x==y):
    print('números são iguais')
else:
    if(x>y):
        print('1º número é maior')
    else:
        print('2º número é maior')

# E4.4) Faça um programa que simule uma operação de login. O programa deverá receber o nome do
# usuário e a senha. Caso os valores sejam “prognit” e “1234”, para usuário e senha, apresente no
# console a mensagem “Acesso Permitido”. Do contrário, apresente a mensagem “Acesso Negado”.

login = input('Login: ')

senha = input('Senha: ')

if ( (login=='prognit') & (senha=='1234') ):
    print('Acesso Permitido')
else:
    print('Acesso Negado')

# E4.5) Faça um programa que receba quatro valores referentes aos lados de uma figura geométrica.
# Apresente no console se estes valores correspondem aos lados de um quadrado, retângulo ou de
# uma outra figura geométrica.

def figura(x,y,w,z):
    if ( (x==y) & (y==w) & (w==z) ):
        return 'quadrado'
    if ( ((x==y) & (w==z)) | ((x==w)& (y==z)) ):
        return 'retângulo'
    return 'uma outra figura geométrica'

s = input('digite o 1º valor: ')
x = float(s)

s = input('digite o 2º valor: ')
y = float(s)

s = input('digite o 3º valor: ')
w = float(s)

s = input('digite o 4º valor: ')
z = float(s)

figura = figura(x,y,w,z)

print ('Tipo: ' + figura )


# E5.1) Faça um programa que repita o seu nome no console 10 vezes. Utilize a estrutura de repetição
# de sua preferência.

for c in range(10):
    print ('Eliaquim Moreira do Nascimento')

# E5.2) Faça um programa que apresente no console todos os números presentes no intervalo entre 1
# e 100. Utilize a estrutura de repetição de sua preferência.

for c in range(1,101):
    print (c)

# E5.3) Coloque o programa do exercício (E4.4) em uma estrutura de repetição. Dê ao usuário três
# tentativas para ter acesso permitido no sistema de login. Utilize a estrutura de repetição de sua
# preferência.

for c in range(1,4):
    login = input('Login: ')

    senha = input('Senha: ')

    if ( (login=='prognit') & (senha=='1234') ):
        print('Acesso Permitido')
    else:
        print('Acesso Negado')
        if(c==3):
            print ('Usuário bloqueado')
            break
        else:
            print ('Tentativa: '+str(c))

# E5.4) Coloque o programa do exercício (E4.5) em uma estrutura de repetição. Permita que o
# usuário utilize o programa para identificação de figuras geométricas quantas vezes desejar. Você
# pode utilizar como condição de parada a identificação de um caracter (por exemplo “Digite 0 para
# encerrar”). Utilize a estrutura de repetição de sua preferência e o comando break para forçar a
# saída do laço.

def figura(x,y,w,z):
    if ( (x==y) & (y==w)& (w==z) ):
        return 'quadrado'
   if ( ((x==y) & (w==z)) | ((x==w)& (y==z)) ):
       return 'retângulo'
   return 'uma outra figura geométrica'
while(True):
    x = float (input('digite o 1º valor: '))

    y = float (input('digite o 2º valor: '))

    w = float (input('digite o 3º valor: '))

    z = float (input('digite o 4º valor: '))

    r = figura(x,y,w,z)

    print ('Tipo: ' + r )

    print (' 1 - Continua')
    print (' 0 - Encerrar')
    s = input('digite a opção: ')
    opcao = int(s)
    if(opcao==0):
        break

# E6.1) Faça um programa para realizar a adição entre dois números. Para tanto, crie uma função
# chamada “adição” que realize a operação. Faça a chamada externa da função para executar o
# programa.

def adicao(x,y):
   return x+y

s = input('digite o 1º numero: ')
x = float(s)

s = input('digite o 2º numero: ')
y = float(s)

print (' Adição ')
print ('Resultado: ' + str(adicao(x,y)))

# E6.2) Amplie o seu programa do exercício (E6.1). Crie funções para as operações de subtração,
# multiplicação e divisão. Faça a chamada externa das funções para executar o programa.

def subtracao(x,y):
   return x-y
def multiplicacao(x,y):
   return x*y
def divisao(x,y):
   if( y != 0 ):
       return x/y
   else:
       return 'erro'

s = input('digite o 1º numero: ')
x = float(s)

s = input('digite o 2º numero: ')
y = float(s)

print (' 1 - adição')
print (' 2 - subtração')
print (' 3 - multiplicação')
print (' 4 - divisão')
s = input('digite a opção: ')
opcao = int(s)
if(opcao==1):
    print (' Adição ')
    print ('Resultado: ' + str(adicao(x,y)))
if(opcao==2):
    print (' Subtração ')
    print ('Resultado: ' + str(subtracao(x,y)))
if(opcao==3):
    print (' Multiplicação ')
    print ('Resultado: ' + str(multiplicacao(x,y)))
if(opcao==4):
    print (' Divisão ')
    print ('Resultado: ' + str(divisao(x,y)))

# E6.3) Coloque o seu programa do exercício (E6.2) em uma estrutura de repetição. Permita que o
# usuário selecione a operação aritmética deseja utilizar e que o programa execute enquanto o usuário
# desejar.

while(True):
    s = input('digite o 1º numero: ')
    x = float(s)

    s = input('digite o 2º numero: ')
    y = float(s)

    print (' 1 - adição')
    print (' 2 - subtração')
    print (' 3 - multiplicação')
    print (' 4 - divisão')
    s = input('digite a operação: ')
    opcao = int(s)
    if(opcao==1):
        print (' Adição ')
        print ('Resultado: ' + str(adicao(x,y)))
    if(opcao==2):
        print (' Subtração ')
        print ('Resultado: ' + str(subtracao(x,y)))
    if(opcao==3):
        print (' Multiplicação ')
        print ('Resultado: ' + str(multiplicacao(x,y)))
    if(opcao==4):
        print (' Divisão ')
        print ('Resultado: ' + str(divisao(x,y)))
    
    print (' 1 - Continua')
    print (' 0 - Encerrar')
    opcao = int(input('digite a opção: '))
    if(opcao==0):
        break
# E6.4) Crie uma segunda versão do programa do exercício (E6.3). Implemente parâmetros e retorno
# nas funções e passe via argumento os valores informados pelo usuário.

while(True):
    x = float(input('digite o 1º numero: '))

    y = float(input('digite o 2º numero: '))
    

    print (' 1 - adição')
    print (' 2 - subtração')
    print (' 3 - multiplicação')
    print (' 4 - divisão')
    opcao = int(input('digite a operação: '))
    if(opcao==1):
        print (' Adição ')
        print ('Resultado: ' + str(adicao(x,y)))
    if(opcao==2):
        print (' Subtração ')
        print ('Resultado: ' + str(subtracao(x,y)))
    if(opcao==3):
        print (' Multiplicação ')
        print ('Resultado: ' + str(multiplicacao(x,y)))
    if(opcao==4):
        print (' Divisão ')
        print ('Resultado: ' + str(divisao(x,y)))
    
    print (' 1 - Continua')
    print (' 0 - Encerrar')
    opcao = int(input('digite a opção: '))
    if(opcao==0):
        break

# E7.1) Faça um programa que receba 5 nomes como entrada. Armazene-os em uma lista e depois
# apresente no console os nomes que foram informados. Obs.: Inicie a sua lista com colchetes vazios.
# Exemplo: listaNomes = []

lista_nomes = []
i=0

while(i<5):
    s = input('digite o nome: ')
    lista_nomes.append(s)
    i+=1
for s in lista_nomes:
    print (s)


# E7.2) Faça um programa que receba nomes como entrada. Armazene-os em uma lista e depois
# apresente no console a quantidade e os nomes que foram informados em ordem crescente
# (alfabética) e decrescente. Execute o programa em um laço que repita enquanto o usuário desejar.

while(True):
    print('\tNomes\t')
    while(True):
        s = input('digite o nome: ')
        lista_nomes.append(s)
        print (' 1 - Continua')
        print (' 0 - Encerrar')
        s = input('digite a opção: ')
        opcao = int(s)
        if(opcao==0):
            break
    print ('Acabou')

    lista_nomes.sort()

    for s in lista_nomes:
        print (s)

    lista_nomes.sort(reverse=True)

    for s in lista_nomes:
        print (s)

    print (' 1 - Continua')
    print (' 0 - Encerrar')
    s = input('digite a opção: ')
    opcao = int(s)
    if(opcao==0):
        break


# E7.3) Faça um programa que crie um cardápio com nomes de pratos que são servidos em um
# estabelecimento. Você deverá armazenar os nomes em uma lista e permitir que o usuário faça
# pesquisas para saber se o prato que deseja pedir existe ou não no cardápio. Obs.: Você pode
# utilizar a linha if item in lista:

while(True):
    print('\tCardapio\t')
    while(True):
            s = input('digite o nome do prato: ')
            lista_nomes.append(s)
            print (' 1 - Continua')
            print (' 0 - Encerrar')
            s = input('digite a opção: ')
            opcao = int(s)
            if(opcao==0):
                break
    print ('*********************************')

    p = input ('Pesquisa: ')
    if (p in lista_nomes):
         print('Tem no cardápio')
    else:
         print('Não tem no cardápio')

    print (' 1 - Continua')
    print (' 0 - Encerrar')
    s = input('digite a opção: ')
    opcao = int(s)
    if(opcao==0):
         break

# E7.4) Faça um programa que armazene em uma lista as tarefas a serem executadas durante um dia.
# O seu programa deverá executar as operações abaixo:
# a) Apresentar todas as tarefas
def listar(l):
    for s in l:
        print (s)
# b) Adicionar uma nova tarefa
def add(l,s):
    l.append(s)
    print ('Adicionado com sucesso')
# c) Excluir uma tarefa realizada
def rm(l,i):
    l.remove(i)
    print ('Removido com sucesso')
# d) Apresentar o nome de uma tarefa de acordo com o índice (ex.: qual a segunda tarefa
# (índice 1) do dia?)
def get(l,i):
    print ('Tarefa indice:'+str(i)+','+l[i])
# e) Apresentar nomes de tarefas de acordo com uma fatia de índices
def gets(l,i,f):
    for c in range(i,f):
        print ('Tarefa indice:'+str(c)+','+l[c])
# f) Apresentar a primeira tarefa da lista
def inicio(l):
    print ('Tarefa inicial:'+l[0])
# g) Apresentar a última tarefa da lista
def final(l):
    i=len(l)-1
    print ('Tarefa final:'+l[i])
# h) Apresentar o total de tarefas a serem realizadas.
def total(l):
    print ('Total de tarefa: '+str(len(l)))

l=[]
while(True):    
    print (' 1 - Listar todas as tarefas')
    print (' 2 - Adicionar tarefa por indice')
    print (' 3 - Excluir tarefa por indice')
    print (' 4 - Procurar tarefa por indice')
    print (' 5 - Procurar tarefas por periodo')
    print (' 6 - Primeira tarefa')
    print (' 7 - Ultima tarefa')
    print (' 8 - Total de tarefas')
    print (' 0 - Sair')
    opcao =  int(input('digite a opção: '))
    if(opcao==0):
        break
    if(opcao==1):
        listar(l)
    if(opcao==2):
        tarefa = input (' Nova tarefa: ')
        add(l,tarefa)
    if(opcao==3):
        tarefa = input (' Excluir tarefa: ')
        rm(l,tarefa)
    if(opcao==4):
        tarefa = input (' Indice da tarefa: ')
        get (l,tarefa)
    if(opcao==5):
        i = input ('Primeiro indice da pesquisa: ')
        f = input ('Primeiro indice da pesquisa: ')
        gets (l,i,f)
    if(opcao==6):
        inicio(l)
    if(opcao==7):
        final(l)
    if(opcao==8):
        total(l)

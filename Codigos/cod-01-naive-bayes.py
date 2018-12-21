#Conjunto de importações
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import nltk
from Util  import readBase,remocaoacento,remocaopontos


#Armazenamento de documentos (frases) na lista dataset
dataset = [("@luciogirotto: TERCEIRA RECLAMAÇAO bradesco alobradesco bancocentralbr BandNews datenaoficial ReclameAQUI http://t.co/v9SgaBB27A"),
           ("@RadioRockPuro: @Nilson_de_Vix @claro um pior que outro, não tem serviço que preste// nenhum deles, @Claro, presta. E, são caríssimos. .."),
           ("As melhores e verdadeiras canções compomos quando temos certeza do caminho a seguir. Falo do que vivo, vivo do... http://t.co/6pVMQFg0LM"),
           ("Boa sorte pra querida Cris Duclos da vivo que ta concorrendo ao Caboré 2013. Bacana participar do… http://t.co/xvtwR8UDDB"),
           ("Conte uma História para uma criança Conte uma história pra Mim Itaú Itau Itaú Cultural Itaú Personnalité... http://t.co/TOir8zL4wE"),
           ("Péssimo é a palavra que melhor define o serviço 3G da @ClaroBrasil. O SAC não funciona e a rede vive indisponível"),
           ("Quero q todo mundo tenha dois chocolates: um pra comer e outro pra quem ta pedindo. ^-^ ^-^ Bradesco QueroQueTodoMundoTenha"),
           ("SE FOR FRACO NAO FICA DE PEH,VIVO NO LIMITE,SOH Q NAO SOU LIMITADO. Vivo"),
           ("ADSL OFF, Oi Velox Off novamente... que maravilha!"),
           ("AloBradesco chega de desculpas quero soluções!!! Mais uma vez os caixas do auto-atendimento fora do ar as 11:15 de 13/12/13 SOLUÇÕES!!!"),
           ("alobradesco mais de 30 minutos e não fui atendido ag 2178"),
           ("bancodobrasil boa tarde, não consigo acessar a conta pela net. O que está acontecendo? "),
           ("bancodobrasil fdp. Paguei minha multa já faz tempo e olho no site ta como não paga"),
           ("bancodobrasil me impressionando a cada dia: abrem extratos aleatorios no app pra mobile e dizem q agr ta td bem e q nao foi serio! "),
           ("BancodoBrasil,  incompetência define! "),
           ("BancoItau ItauBanco Itau Horrível ficar esperando por um longo período sem explicação. Acho digno, ao menos uma informação. "),
           ("bradesco, o banco mais retrógrado do brasil!!!!! "),
           ("Bradesco, sempre fudendo com a vida das pessoas! http://t.co/js3pF4n2Yc"),
           ("Cine EmChamas JogosVorazes @Bb' AmorMeu Lôra Branquelo Perfeito :) @Betinho_melo http://t.co/0fzpDxHNqM"),
            ("cineart itaucard paga meia agora. Muito bom http://t.co/uIMWqfinIQ")]
							                  
#Armazenamento das polaridades de cada documento (frase) na lista polaris
polaris = [-1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1,-1, -1, -1, -1, -1, -1, 1, 1]
						        
#Divisão dos dados das listas dataset e polaris em conjuntos de treinamento e validação

 
documents = []
acento = False
#print base[0][0].split()
tknzr = nltk.tokenize.TweetTokenizer()
i=0
l = len(dataset)
while (i<l):
    if(acento):
        w = remocaoacento(tknzr.tokenize(dataset[i]))
    else:
        w = tknzr.tokenize(dataset[i])
    w = remocaopontos(w)
    conteudoLista = (w)
    documents.append(conteudoLista)
    i += 1

    ################################ Pre Processamento
    stopwords = nltk.corpus.stopwords.words('portuguese')    

    stemmer = nltk.stem.RSLPStemmer()

    # h=0
    # j=len(documents)
    # while (h<j):
    #    g=len(documents[h][0])
    #    f=0
    #    while(f<g):
    #        stemmer.stem(documents[h][0][f])
    #        f+=1
    #    h += 1

    documento  = []
    for linha in dataset:
      documento.append( [stemmer.stem(w.lower()) for w in linha if w not in stopwords])
    # documents = [[removerpalavras(freq,w[0]),w[1]] for w in documents]
'''
'''
dados_treino, dados_val, pols_treino, pols_val = train_test_split(documento, polaris, test_size=0.30)

#Print do conjunto de treinamento e suas respectivas polaridades
print("Conjunto de Treinamento")
print(dados_treino)
print("Polaridades do Conjunto de Treinamento")
print(pols_treino)
print("\n---------------------------------------------\n")
#Print do conjunto de validação e suas respectivas polaridades
print("Conjunto de Validação")
print(dados_val)
print("Polaridades do Conjunto de Validação")
print(pols_val)

#Cria uma instância para a bag-of-words   
bag = CountVectorizer()

#Método fit_transform: 
#fit = cria e aprende a bag
#transform = cria a matriz termo-documento
bag_treino = bag.fit_transform(dados_treino)

#A função sorted() ordena o vocabulário da bag-of-words   
print("Vocabulário da bag-of-words")
print(sorted(bag.vocabulary_))
print("\n---------------------------------------------\n")

#Printa a bag-of-words    
print("Bag-of-words de treino")
print(bag_treino)
print("\n---------------------------------------------\n")

#Cria a matriz termo-documento para o conjunto de validação com a bag já treinada
bag_val = bag.transform(dados_val)

#Printa a matriz termo-documento criada para o conjunto de validação    
print("Bag-of-words de validação")
print(bag_val)
print("\n---------------------------------------------\n")

#Cria uma instância para o algoritmo Multinomial Naive Bayes    
nb_modelo = MultinomialNB()
#O método fit treina o modelo utilizando o algoritmo Multinomial Naive Bayes
#O argumento da bag deve ser passado no formato array
nb_modelo.fit(bag_treino.toarray(), pols_treino)

#Realiza as predições para o conjunto de treinamento  
pols_pred_treino = nb_modelo.predict(bag_treino.toarray())
#Realiza as predições para o conjunto de validação
pols_pred_val = nb_modelo.predict(bag_val.toarray())
#Printa as predições calculadas para ambos os conjuntos
print("Polaridades preditas para o conjunto de treinamento")
i = 0
treino = bag_treino.toarray()
l = len(pols_val)
count  = 0
while i<l:
  if pols_val[i]==pols_pred_val[i] :
    count+=1
print('calculo:'+str(count/l))


print(pols_pred_treino)
print("Polaridades preditas para o conjunto de validação")
print(pols_pred_val)

#Armazenamento da frase teste na variável frase_teste
frase_teste = [("Eu amo muito este visual")]

#Cria a bag-of-words para a frase_teste
bag_teste = bag.transform(frase_teste)

#Aplica o modelo Multinomial Naive Bayes aprendido na bag criada
pol_pred_teste = nb_modelo.predict(bag_teste.toarray())
                
#Estrutura de decisão para apresentar o resultado como String
#Resultado = 1 ==> Polaridade POSITIVO
#Resultado = -1 ==> Polaridade NEGATIVO
if pol_pred_teste == 1:
   print("POSITIVO")
else:
   print("NEGATIVO")
print("\n---------------------------------------------\n")


'''import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)'''
import csv
import sklearn.metrics as metricas
import Codigos.Recurso as Re

def readBase1(csvFile = str):
    classes = [0]*6
    base = []
    i= False
    with open(csvFile, newline='\n',encoding='utf-8') as csvFile:

        spamreader = csv.reader(csvFile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                if(i):
                    temp = int(row[0])
                    classes[temp - 1]+=1
                    base.append(classes.copy())
                else:
                    i = True
            except Exception:
                pass
    return base
def readBase2(csvFile = str):
    base = []
    i= False
    j=0
    with open(csvFile, newline='\n',encoding='utf-8') as csvFile:

        spamreader = csv.reader(csvFile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                if(i):
                    temp = int(row[0])
                    base.append(temp)
                    # print(j)
                    j+=1
                else:
                    i = True
            except Exception:
                print(j)
                pass
    return base

def writes(path,base):
    with open(path,mode='w', encoding='utf-8') as csv_file:
        #writer = csv.writer(csv_file)
        for w in base:
            csv_file.writelines(w[0]+';'+w[1]+'\n')

def coluna(base, c =int):
    column = []
    for w in base:
        column.append(w[c-1])
    return column

def mostra_grafico(base,path):
    import matplotlib.pyplot as plt

    pos = coluna(base,1)
    neg = coluna(base,2)
    amb = coluna(base,3)
    neu = coluna(base,4)
    des = coluna(base,5)
    iro = coluna(base,6)

    tempos = range(1, len(base) + 1)
    plt.plot(tempos, pos, 'k', label='Positivo')
    plt.plot(tempos, neg, 'b', label='Negativo')
    plt.plot(tempos, amb, 'r', label='Ambos')
    plt.plot(tempos, neu, 'p', label='Neutro')
    plt.plot(tempos, des, 'g', label='Descarte')
    plt.plot(tempos, iro, 'y', label='Ironia')
    plt.title(path)
    plt.xlabel('opiniao')
    plt.ylabel('classes')
    plt.legend()

    plt.show()
    plt.clf()   # clear figure

def mostra_graficos(bases,paths):
    i = 0
    l = len(bases)
    while(i<l):
        mostra_grafico(bases[i],paths[i])
        i+=1

def bases(paths,grafico):
    if(grafico):
        base1 = readBase1(paths[0])
        base2 = readBase1(paths[1])
        base3 = readBase1(paths[2])
        base4 = readBase1(paths[3])
    else:
        base1 = readBase2(paths[0])
        base2 = readBase2(paths[1])
        base3 = readBase2(paths[2])
        base4 = readBase2(paths[3])

    return base1 ,base2 ,base3 ,base4
def helper(base1,base2):
    b1 = []
    b2 = []
    pol = [1,2]
    i = 0
    while(i<len(base1)):
        if(base1[i] in pol):
            if(base2[i] in pol):
                b1.append(base1[i])
                b2.append(base2[i])
        i+=1
    return b1,b2

paths = ['data_set\data_base_d_q_1000.csv',
         'data_set\data_base_s_1000.csv',
         'data_set\data_base_v_a_2000.csv',
         'data_set\data_base_v_o_2000_1.csv']

base1 ,base2 ,base3 ,base4 = bases(paths,False)

base1,base2 = helper(base1,base2)
base3,base4 = helper(base3,base4)
print('meninas')
print(len(base1))
print('meninos')
print(len(base3))
print('\n')
# print(base3)

kappa_meninas = metricas.cohen_kappa_score(base1,base2)
print(kappa_meninas)
kappa_meninos = metricas.cohen_kappa_score(base3,base4)
print(kappa_meninos)


# mostra_graficos([base1 ,base2 ,base3 ,base4],paths)

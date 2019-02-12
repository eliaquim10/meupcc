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

def readBase(csvFile = str):
    classes = [0]*6
    base = []
    i= False
    j=0
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

path = 'data_set\data_base_d_q_1000.csv'
base = readBase(path)

mostra_grafico(base,path)

path = 'data_set\data_base_v_a_2000.csv'
base = readBase(path)

mostra_grafico(base,path)

path = 'data_set\data_base_s_1000.csv'
base = readBase(path)

mostra_grafico(base,path)



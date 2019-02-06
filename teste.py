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

def readBase(csvFile1 = str,csvFile2 = str):
    base1 = []
    base2 = []
    i= True
    j=0
    with open(csvFile1, newline='\n',encoding='utf-8') as csvFile1:

        spamreader = csv.reader(csvFile1, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                temp1 = row[0].lower()
                base1.append(temp1[0])
            except IndexError:
                pass

    with open(csvFile2, newline='\n',encoding='utf-8') as csvFile2:

        spamreader = csv.reader(csvFile2, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                print(row)
                if(i):
                    temp1 = row[0].lower()
                    temp2 = row[1].lower()
                    base2.append([temp1, temp2])
                    i = False
                else:
                    temp1 = row[1].lower()
                    base2.append([base1[j], temp1])
            except IndexError:
                pass
            finally:
                j+=1
        return base2


def writes(path,base):
    with open(path,mode='w', encoding='utf-8') as csv_file:
        #writer = csv.writer(csv_file)
        for w in base:
            csv_file.writelines(w[0]+';'+w[1]+'\n')

path = 'data_set\data_base_vinicius_a_2000'
base = readBase(path + '_1.csv',path + '.csv')
writes(path + '_2.csv',base)
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
    base = [0]*6
    i= False
    j=0
    with open(csvFile, newline='\n',encoding='utf-8') as csvFile:

        spamreader = csv.reader(csvFile, delimiter=';', quotechar='|')

        for row in spamreader:
            try:
                if(i):
                    temp = int(row[0])
                    if(temp==4):
                        print(row[1])
                    base[temp - 1]+=1
                else:
                    print(row)
                    i = True
            except IndexError:
                pass
        return base


def writes(path,base):
    with open(path,mode='w', encoding='utf-8') as csv_file:
        #writer = csv.writer(csv_file)
        for w in base:
            csv_file.writelines(w[0]+';'+w[1]+'\n')

path = 'data_set\data_base_dilva_1000.csv'
base = readBase(path)
print(base)
from six.moves import zip as izip
import os

def comparar(text,senteca):
    len_text = len(text)
    len_senteca = len(senteca)
    i = 0
    while(i<len_text):
        # if(len_text-i<len_senteca):
        #     return False
        # s = 'but a little too smugly superior to like'
        # # f = 'Too smart to ignore but a little too smugly superior to like , this could be a movie that ends up slapping its target audience in the face by shooting itself in the foot .'
        # f = 'Too smart to ignore but a little too smugly superior to like'
        # print(len(f))
        # print(i,i+len_senteca)
        if(senteca == text[i:i+len_senteca]):
            return True
        i+=1
    return False


def compare(dataset,senteca):
    for text in dataset:
        if(comparar(text,senteca)):
            return True
    return False

def normatize(dataset,phrases,sentiments,dictionary , sentiment_labels):
    aux_phrases = dictionary.copy()
    aux_sentiment = sentiment_labels.copy()
    for phrase,sentiment in izip(phrases,sentiments):
        if(phrase not in aux_sentiment):
            if (compare(dataset,phrase)):
                aux_phrases.append(phrase)
                aux_sentiment.append(sentiment)
    return aux_phrases,aux_sentiment

def read_dictionary(dirname,file):
    data = []
    with open(os.path.join(dirname, file), 'r') as reader:
        for line in reader:
            (text, id) = line.split('|')
            data.append(text)
    return data

def read_sentiment(dirname,file):
    data = []
    with open(os.path.join(dirname, file), 'r') as reader:
        next(reader)
        for line in reader:
            (id ,text) = line.split('|')
            data.append(text)
    return data

def summary_data(dirname,dictionary,sentiment_labels):
    dictionary = read_dictionary(dirname,dictionary)
    sentiment_labels = read_sentiment(dirname,sentiment_labels)

    return dictionary ,sentiment_labels,len(dictionary)
def ordernar(dirname,file):
    legend = ''
    dataset = {}
    with open(os.path.join(dirname, file), 'r') as sentences:
        # next(sentences)  # legend
        for sentence_line in sentences:
            (text, id) = sentence_line.split('|')
            dataset[int(id)] = text

    dataset = sorted(dataset.items())
    # print(datasource)
    # exit()
    with open(os.path.join(dirname, file), mode='w', encoding='utf-8') as dictionary:
            for id,text in dataset:
                dictionary.writelines( text + '|' + str(id) + '\n')
def recorrente(li = 0,lo = 11855):
    dirname = 'data/stanfordSentimentTreebank'

    # path = 'data\stanfordSentimentTreebank\dictionary.txt'
    print("loading corpus from writer")

    dictionary , sentiments,i = summary_data(dirname,'dictionary.txt','sentiment_labels.txt')

    dataset = []
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
            next(sentences)  # legend
            j = 0
            for sentence_line in sentences:
                if(li<=j):
                    if(lo>j):
                        (id, text) = sentence_line.split('\t')
                        dataset.append(text)
                    else:
                        break
                j+=1

    # read all phrase text
    copy_phrases_data = read_dictionary(dirname,'dictionary_copy.txt')# known size of phrases
    copy_sentiment = read_sentiment(dirname,'sentiment_labels_copy.txt')


    copy_phrases_data,copy_sentiment = normatize(dataset ,copy_phrases_data ,copy_sentiment, dictionary, sentiments)

    # # with open(path, mode='w', encoding='utf-8') as csv_file2:
    i = 0
    with open(os.path.join(dirname, 'dictionary.txt'), mode='w', encoding='utf-8') as dictionary_file:
        with open(os.path.join(dirname, 'sentiment_labels.txt'), mode='w', encoding='utf-8') as sentiment_file:
            for p_d,s in izip(dictionary,sentiments):
                i +=1
                dictionary_file.writelines( p_d + '|' + str(i) + '\n')
                sentiment_file.writelines( str(i) + '|' + s)

            for p_d,s in izip(copy_phrases_data,copy_sentiment):
                i +=1
                dictionary_file.writelines( p_d + '|' + str(i) + '\n')
                sentiment_file.writelines( str(i) + '|' + s)

recorrente(lo = 3)
# for w in range(0,5):
#     recorrente(w*10,(w+1)*10)
#     print('process',w)

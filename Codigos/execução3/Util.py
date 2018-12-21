# -*- coding: utf-8 -*-
import csv
import numpy as np

def readBase(csvFile = str):
    base = []
    with open(csvFile, newline='\n') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            try:
                #sem caracteres especiais
                # vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
                # print(row[2])
                # temp1 = ''
                # for w in row[2]:
                #     if (w not in vida):
                #         temp1 +=str(w).lower()
                # temp2 = row[1]

                #com caracteres especiais
                temp1 = row[2].lower()
                temp2 = row[1].lower()
                base.append(tuple([temp1, temp2]))
            except IndexError:
                pass
        return base

# def printResults(classifier = str, results = [], file = file):
#     file.write(classifier+ " \n")
#     file.write("Iterations:\n")
#     for z in range(len(results)):
#         file.write(str(results[z])+"\n")
#     file.write("Average:{}\n".format(np.average(results)))
#     file.write("Standard Deviation:{}\n\n\n\n".format(np.std(results)))

def remocaoacento(base):
    vida = '^áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ'
    base2 = []
    for row in base:
        temp1 = ''
        for w in row:
            if (w not in vida):
                temp1 +=w.lower()
        base2.append(temp1)
    return base2

def remocaopontos(base):
    vida = '.,!?<>][*()+-;:'
    base2 = []
    for row in base:
        if (row not in vida):
            base2.append(row)
        else:
            base2.append(' ')
    return base2

def precisionRecallFmeasure(classifier, gold):
    #results = classifier.classify_many([fs for (fs, l) in gold])
    #correct = [l == r for ((fs, l), r) in zip(gold, results)]

    testclas = classifier.classify_many([fs for (fs, l) in gold])
    testgold = [l for (fs, l) in gold]

    h=0
    #cont = 0
    tpos=0
    fpos=0
    tneg=0
    fneg=0
    tneu=0
    fneu=0
    fneuPos=0
    fneuNeg=0
    fnegPos=0
    fnegNeu=0
    fposNeg=0
    fposNeu=0
    while(h<len(testgold)):
        if (testgold[h]==testclas[h])and (testclas[h]== u"positivo"):
            tpos=tpos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo"):
            fpos=fpos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo") and (testgold[h]==u'negativo'):
            fposNeg=fposNeg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"positivo") and (testgold[h]==u'neutro'):
            fposNeu=fposNeu+1
        if (testgold[h]==testclas[h])and (testclas[h]== u"negativo"):
            tneg=tneg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo"):
            fneg=fneg+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo") and (testgold[h]==u'positivo'):
            fnegPos=fnegPos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"negativo") and (testgold[h]==u'neutro'):
            fnegNeu=fnegNeu+1
        if (testgold[h]==testclas[h])and (testclas[h]== u"neutro"):
            tneu=tneu+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro"):
            fneu=fneu+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro") and (testgold[h]==u'positivo'):
            fneuPos=fneuPos+1
        if (testgold[h]!=testclas[h])and (testclas[h]== u"neutro") and (testgold[h]==u'negativo'):
            fneuNeg=fneuNeg+1
        h=h+1

#-------------------------
    precisionPos = float(tpos)/(tpos+fpos)
    precisionNeg = float(tneg)/(tneg+fneg)
    if((len(set([l for (fs, l) in gold])))==3):
        precisionNeu = float(tneu)/(tneu+fneu)
    else:
        precisionNeu = 0
    precision = float(precisionPos+precisionNeg+precisionNeu)/(len(set([l for (fs, l) in gold])))
#--------------------------
    recallPos = float(tpos)/(tpos+fnegPos+fneuPos)
    recallNeg = float(tneg)/(tneg+fposNeg+fneuNeg)
    if((len(set([l for (fs, l) in gold])))==3):
        recallNeu = float(tneu)/(tneu+fnegNeu+fposNeu)
    else:
        recallNeu = 0
    recall = float(recallPos+recallNeg+recallNeu)/(len(set([l for (fs, l) in gold])))
#---------------------------
    if((len(set([l for (fs, l) in gold])))==3):
        mc = '''
             Pos   Neg   Neu\n
        Pos   '''+str(tpos)+"    "+str(fposNeg)+"    "+str(fposNeu)+'''\n
        Neg   '''+str(fnegPos)+"    "+str(tneg)+"    "+str(fnegNeu)+'''\n
        Neu   '''+str(fneuPos)+"    "+str(fneuNeg)+"    "+str(tneu)
    if((len(set([l for (fs, l) in gold])))==2):
        mc ='''
               Pos   Neg \n
        Pos    '''+str(tpos)+"   "+str(fpos)+'''\n
        Neg    '''+str(fneg)+'''   '''+str(tneg)
#---------------------------

    fmeasure = (2*precision*recall)/(precision+recall)
    fmeasurePos = (2*precisionPos*recallPos)/(precisionPos+recallPos)
    fmeasureNeg = (2*precisionNeg*recallNeg)/(precisionNeg+recallNeg)
    fmeasureNeu = (2*precisionNeu*recallNeu)/(precisionNeu+recallNeu)
#---------------------------

    string = "precision: "+str(precision)+"  precisionPos: "+str(precisionPos)+'  precisionNeg: '+str(precisionNeg)+'  precisionNeu: '+str(precisionNeu)+"" \
             "\n recall: "+str(recall)+"  recallPos: "+str(recallPos)+"  recallNeg: "+str(recallNeg)+"  recallnNeu: "+str(recallNeu)+"" \
             "\n F-measure: "+str(fmeasure)+"  f-measurePos: "+str(fmeasurePos)+"  f-measureNeg: "+str(fmeasureNeg)+"  f-measureNeu: "+str(fmeasureNeu)+"" \
                                                                                                                                                                                "\n"+str(len(set([l for (fs, l) in gold])))+"\nf-measure: "+str(fmeasure)
    return string
'''
    if correct:
        return float(sum(correct))/len(correct)
    #acuracia = float(sum(x == y for x, y in izip(reference, test))) / len(test)
    #precision = float(len(reference.intersection(test)))/len(test)
    #recal    = float(len(reference.intersection(test)))/len(reference)
    #fmeasure =
    else:
        return 0
'''

def write_results(resultados =[], times =[], silhoetes =[], path = str):

    '''
    Função responsável por escrever um arquivo csv com todos os resultados obtidos
    :param resultados: resultados de cada iteração do PSO
    :param times: o tempo de execução para cada iteração do PSO
    :param silhoetes: Avaliação da semelhança dos clusters antes e depois da execução do PSO
    em cada execução
    :param path: caminho e nome do arquivo a ser salvo
    '''


    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([" ", "Classe 0", "Classe 1","Classe 2", "All Class"])
        writer.writerow(["Execution Number","Precision/Recall/F-Score",
        "Precision/Recall/F-Score","Precision/Recall/F-Score",
        "Precision/Recall/F-Score/Support",
        "Silhouette Coefficient","Time (seconds)"])

    print_vetor = [[z+1, a[0], a[1], a[2],np.mean(a, axis=1), b, c]for z, a, b, c in zip(range(0,len(resultados)),resultados,silhoetes,times)]

    for i in range(len(print_vetor)):
        writer.writerow(print_vetor[i])

    media = np.array(reduce(operator.add, np.mean(np.array(resultados), axis=2)))/len(resultados)
    writer.writerow([" ","Precison", "Recall", "F-Score"])
    writer.writerow(["Media Final:",media[0], media[1], media[2]])
# coding:utf-8  
'''
Created on 2019年7月2日

@author: cvter
'''
import argparse
import jieba
from gensim.models import doc2vec
import csv
import random
import warnings

warnings.filterwarnings("ignore")#warning off
def parse_args():#define the paramter of program
    parser = argparse.ArgumentParser(description="dutyModel.")
    parser.add_argument('--fileName', nargs='?', default='med.csv',help='fileName.')
    return parser.parse_args(args=[])

def get_stop_words():#load the stopwords 
    spath = '../data/stopword.txt'
    stopwords = [line.strip() for line in open(spath, 'r', encoding='GBK').readlines()]  
    return stopwords

def get_lineText(textpath): #get the data and tokenize
    rows = csv.reader(open(textpath,'r',encoding='utf-8'))
    lineText = []
    rawText = []
    stopwords = get_stop_words()
    for r in rows:
        rawText.append(r[0])
        seg_list = jieba.lcut(r[0].strip()) 
        txt_list = [' '.join(seg) for seg in seg_list if seg not in stopwords]
        lineText.append(txt_list)
    return lineText,rawText

def train_doc2vec_model(tagged_data,max_epochs=10):
    vec_size = 20
    alpha = 0.025
    #If dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW). 
    model = doc2vec.Doc2Vec(size=vec_size,alpha=alpha, min_alpha=0.00025,min_count=1,dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha 
    return model 

if __name__ == '__main__':
    #1.读取参数
    args = parse_args()
    fileName = args.fileName
    #2.加载文本，utf-8格式
    lineText,rawText = get_lineText('../data/'+fileName)
    print ("The texts %d has been loaded successfully in the file %s" % (len(lineText),fileName))
    with open('../data/raw.txt','w',encoding='GBK') as fw:
        lists=[line+"\n" for line in rawText]
        fw.writelines(lists)
    #3.词典生成和训练模型
    tagged_data = [doc2vec.TaggedDocument(words=line, tags=[str(i)]) for i, line in enumerate(lineText)]
    model = train_doc2vec_model(tagged_data,10)
    model.save("../data/d2v.model")
    print("Doc2Vec Model Saved")
    '''
    print ("%3s%20s%20s" % ('K','Epochs','Rank Ratio'))
    testTag = random.sample(range(len(rawText)), k=600)#选择约0.2测试集
    for epochs in [5,10,50,100]:
        for k in [5,10,15,20]:
            model = train_doc2vec_model(tagged_data,epochs)
            rankfloat = 0.0
            for i in testTag:
                mostSim=model.docvecs.most_similar(i,topn=k)#not include iteself
                for _ , sim in mostSim: rankfloat=rankfloat+float(sim)
            print ("%3s%20s%20.8f" % (k, epochs, rankfloat/(k*600)))    
    '''
    '''
    print ("%3s%20s%20s%20s" % ('K','Epochs','Hit Ratio','Rank Ratio'))
    testTag = random.sample(range(len(rawText)), k=600)#选择约0.2测试集
    for epochs in [2,5,8,10]:
        for k in [5,10,15,20]:
            model = train_doc2vec_model(tagged_data,epochs)
            hitnum = 0
            rankfloat = 0.0
            for i in testTag:
                mostSim=model.docvecs.most_similar(i,topn=k)#not include iteself
                for j, sim in mostSim:
                    #sims = model.docvecs.similarity(j,j)
                    #print ("The similarity is %f between %s - %s"%(sims,j,j))                    
                    if int(i)==int(j): 
                        hitnum=hitnum+1
                        rankfloat = rankfloat+sim
            hitRatio = hitnum*1.0/k
            rankRatio = rankfloat/k    
            print ("%3s%20s%20.8f%20.8f" % (k, epochs, hitRatio, rankRatio))
    '''
'''
 K              Epochs          Rank Ratio
  5                   5          0.89471015
 10                   5          0.88462182
 15                   5          0.87868902
 20                   5          0.87069205
  5                  10          0.89738335
 10                  10          0.88398079
 15                  10          0.87756086
 20                  10          0.87146660
  5                  50          0.89096438
 10                  50          0.87417758
 15                  50          0.86305277
 20                  50          0.85525491
  5                 100          0.89313091
 10                 100          0.87477955
 15                 100          0.86354683
 20                 100          0.85412271
'''
    
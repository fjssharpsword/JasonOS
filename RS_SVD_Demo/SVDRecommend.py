'''
Created on 2019.6.19
@author: Jason.F
@summary:
SVDRecommend.py: Reommend TopK items to user.
Dependencies: python3.x, numpy, pandas, surprise, sklearn. you can install their by pip tool.
Input: urdList, the format is one uid per line. the datatype is int.
Output: urdList, topk items for specific userid. The format of every line is:uid [itemid1,itemid2,...,itemidk]
Usage:python SVDRecommend.py --TopK 5 --modelPath /data/fjsdata/nursereport/svd.model --uidPath /data/fjsdata/nursereport/uid.list
'''
import pandas as pd
import numpy as np
import surprise as sp
import time
import argparse
import math
import os
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def parse_args():#define the paramter of program
    parser = argparse.ArgumentParser(description="Run SVD.")
    parser.add_argument('--TopK', nargs='?', default=10, help='Recommend K items')
    parser.add_argument('--modelPath', nargs='?', default='/data/fjsdata/nursereport/svd.model',
                        help='Data path of saving model.')
    parser.add_argument('--uidPath', nargs='?', default='/data/fjsdata/nursereport/uid.list',
                        help='The uid will be recommended.')
    return parser.parse_args()

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def load_data(filepath):#read file
    list_uid =[]
    with open(filepath, "r") as f:
        line = f.readline()
        while line != None and line != "":
            list_uid.append(line.strip())
            line = f.readline()
    return list_uid

def write_data(filepath, iidList):#write file
    with open(filepath,"w") as f:
        for iid in iidList:
            f.write(str(iid[0])+" ["+",".join(str(x) for x in iid[1])+"]")
            f.write('\n')
        f.close()

if __name__ == '__main__':
    #1.load data with parameter,dataPath and modelPath
    args = parse_args()
    topK = args.TopK
    modelPath = args.modelPath
    uidPath =args.uidPath
    #2.load the model
    predictions, algo = sp.dump.load(modelPath)
    print ("The model has loaded successfully from the path:%s" % modelPath)
    #3.get the topk items
    top_n = get_top_n(predictions, n=int(topK))
    #4.recommended items for each user
    uidList = load_data(uidPath)
    iidList = []
    for uid in uidList:
        user_ratings = top_n.get(int(uid))
        if user_ratings!=None:
            iid_rec = [int(uid), [iid for (iid, _) in user_ratings]]
            iidList.append(iid_rec)
        else:
            iid_rec = [int(uid),[]]
            iidList.append(iid_rec)
    #5.output the results of recommendation.
    write_data(uidPath,iidList)
    print ("Complete recommendation.")

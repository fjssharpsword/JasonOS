'''
Created on 2019.6.19
@author: Jason.F
@summary:
SVDTrain.py:Training the SVD model.
Dependencies: python3.x, numpy, pandas, surprise, sklearn. you can install their by pip tool.
Input: The format is: userid, itemid, rating, among them the rating denotes behavior records of nurse on items.
       the datatype of userid and itemid is int, the number range from zero to max of users and items. The datatype of rating is float or int, such as 2.5, 3.
Output: The learned SVD model which can recommend topk items to nurse based on the collaborative filtering.
Usage:python SVDTrain.py --dataPath /data/fjsdata/nursereport/ui.rating --modelPath /data/fjsdata/nursereport/svd.model
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
    parser.add_argument('--dataPath', nargs='?', default='/data/fjsdata/nursereport/ui.rating',
                        help='Data path of training file.')
    parser.add_argument('--modelPath', nargs='?', default='/data/fjsdata/nursereport/svd.model',
                        help='Data path of saving model.')
    return parser.parse_args()

def load_data(filepath):#read file
    list_rating =[]
    with open(filepath, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
            list_rating.append([user, item, rating])
            line = f.readline()
    df_rating = pd.DataFrame(list_rating, columns=['u','i','r'])
    #normalize the rating in the range[0,1]
    num_max=df_rating['r'].max()
    num_min=df_rating['r'].min()
    df_rating['r']=df_rating['r'].apply(lambda x: (x-num_min+1)*1.0/(num_max-num_min+1) )
    return df_rating

def calc_dcg(items):#calculate DCG and IDCG
    dcg = 0
    i = 0
    for item in items:
        i += 1
        dcg += (math.pow(2, item) - 1)/ math.log(1 + i, 2)
    return dcg

def index_at_k(predictions, k, threshold=0.1):
   #Return precision and recall at k metrics for each user.
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    ndcgs =dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r > threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est > threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r > threshold) and (est > threshold)) for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        #true ratings of recommended items in top k
        l_rec_k = [true_r for (_,true_r) in user_ratings[:k]]
        dcg = calc_dcg(l_rec_k)
        #l_rec_k.sort(reverse=True)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        l_rel_k = [true_r for (_,true_r) in user_ratings[:k]]
        idcg = calc_dcg(l_rel_k)
        ndcgs[uid]=dcg*1.0/idcg 
    return precisions, recalls, ndcgs
      
if __name__ == '__main__':
    #1.load data with parameter,dataPath and modelPath
    args = parse_args()
    dataPath = args.dataPath
    modelPath = args.modelPath
    df_rating = load_data(dataPath)
    print ('Dataset has loaded and its shape is:%d rows and %d columns'%(df_rating.shape[0],df_rating.shape[1]))
    #2.Transforming into data format of surprise and spliting the train-set and test-set
    # The columns must correspond to user id, item id and ratings (in that order).
    reader = sp.Reader(rating_scale=(0, 1))
    spdata = sp.Dataset.load_from_df(df_rating[['u', 'i', 'r']],reader)
    # sampling random trainset and testset
    trainset = spdata.build_full_trainset()
    testset = trainset.build_testset()
    #3.Training the model and predicting ratings for the testset
    st = time.time()
    algo = sp.SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    et =time.time()
    print ('Model has trained successfully in %s seconds!'%(et - st))
    
    #4.measure the model
    print ("RMSE:%0.8f" % (sp.accuracy.rmse(predictions)))
    print ("%3s%20s%20s%20s" % ('K','Precisions','Recalls','NDCG'))
    for k in [5,10,15,20]:#latent factor
        precisions, recalls, ndcgs = index_at_k(predictions, k=k)
        # Precision and recall can then be averaged over all users
        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        ndcg = sum(ndcg for ndcg in ndcgs.values()) / len(ndcgs)
        print ("%3s%20.8f%20.8f%20.8f" % (k, precision, recall, ndcg))
    
    #5.save the model
    file_name = os.path.expanduser(modelPath)
    sp.dump.dump(file_name, predictions=predictions,  algo=algo)# Dump algorithm
    print ("The model has saved successfully in the path:%s" % file_name)

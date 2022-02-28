import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score
from regression_into_classification import *

def finding_best_candidate_two_obj(y, y_train, y_test,y_val,w_train,w_test,w_val, X_train, X_test,X_val, y_pred , y_std,w_pred_prob,w_std,Model,method):
    if method== "entropy":
        index_new_experiment = entropy(y_pred, w_pred_prob) 
    elif method == "random":
        index_new_experiment = random.randint(0, len(y_pred)-1)
    elif method =="least_confident":
        index_new_experiment = least_confident(y_pred,w_pred_prob)
    elif method == "MU":
        index_new_experiment = MU(y_std)
    elif method == "MLI":
        index_new_experiment = MLI(y_pred, y_std)
    elif method =="margin":
        index_new_experiment = margin(y_pred,w_pred_prob)
    elif method == "distance":
        index_new_experiment = distance(X_train, X_test)
    elif method == "best":
        index_new_experiment = BEST(Model,X_train,y_train,X_test,y_test,X_val,y_val)
    elif method == "least_confident_distance":
        l = least_confident_distance(y_pred,w_pred_prob,X_train,X_test,X_val)
        index_new_experiment = distances_of_selected(X_train, X_test,l)
    elif method == "margin_distance":
        l=margin_distance(y_pred,w_pred_prob,X_train,X_test,X_val)
        index_new_experiment = distances_of_selected(X_train, X_test,l)
    elif method == "entropy_distance":
        l = entropy_distance(y_pred,w_pred_prob,X_train,X_test,X_val)
        index_new_experiment = distances_of_selected(X_train, X_test,l)
        
    X_train, X_test, y_train,y_test,w_train,w_test = building_new_training_and_test_sets_two_obj(X_train, X_test,y_train, y_test,w_train,w_test, index_new_experiment)
    return (X_train,X_test,y_train,y_test,w_train,w_test)
    
def building_new_training_and_test_sets(X_train, y_train, X_test, y_test, index): #, iteration x_times_choosen, x_updated_index_array):
    """prepare the next training and testset for the next iteration"""
    #append to the X_train and y_train
    X_t = np.reshape(X_test[index], (1,len(X_test[index])))
    X_train_new = np.append(X_train,X_t, axis= 0)
    y_train_new = np.append(y_train, y_test[index])
    #drop from X_test and y_test
    X_test_new = np.delete(X_test, obj = index, axis=0)
    y_test_new = np.delete(y_test, obj = index, axis=0)
    return (X_train_new, y_train_new, X_test_new, y_test_new) #, iteration, x_times_choosen, x_updated_index_array_new)

def building_new_training_and_test_sets_two_obj(X_train, X_test,y_train, y_test,w_train,w_test, index):
    X_t = np.reshape(X_test[index], (1,len(X_test[index])))
    X_train_new = np.append(X_train,X_t, axis= 0)
    y_train_new = np.append(y_train, y_test[index])
    w_train_new = np.append(w_train, w_test[index])

    X_test_new = np.delete(X_test, obj = index, axis=0)
    y_test_new = np.delete(y_test, obj = index, axis=0)
    w_test_new = np.delete(w_test, obj = index, axis=0)
    return X_train_new, X_test_new, y_train_new,y_test_new,w_train_new,w_test_new

# methods for selecting the next best candidate "to make an experiment with"  
def MU(y_std):
    """outputs the X_test index with the biggest uncertainty corresponding to it"""
    index_new_experiment = y_std.argmax(axis=0)
    return index_new_experiment

def MLI(y_pred, y_std):
    """outputs the X_test value which gives the maximum likelyhood of improvement --> better use least_confident"""
    index_new_experiment = (abs(0.5 - y_pred)-y_std).argmin(axis=0)
    return index_new_experiment

def BEST(Model,X_train,y_train,X_test,y_test,X_val,y_val):
    score = []
    for index in range(len(y_test)):
        X_t = np.reshape(X_test[index], (1,len(X_test[index])))
        X_train_new = np.append(X_train,X_t, axis= 0)
        y_train_new = np.append(y_train, y_test[index])
        Model.fit(X_train_new, y_train_new)
        y_pred_val = Model.predict(X_val)
        acc_val = balanced_accuracy_score(y_val,y_pred_val)
        score.append(acc_val)
    score = np.asarray(score)
    index_new_experiment = score.argmax(axis=0)
    return index_new_experiment

def entropy(pred_prob1,pred_prob):
   """inputs are probabilities of the predictions"""
   p=- np.sum(np.multiply(pred_prob,np.log2(pred_prob,where=0<pred_prob)),axis=1)
   index_new_experiment = max(np.argwhere(p == np.amax(p)))[0]
   return index_new_experiment

def least_confident(pred_prob1,pred_prob):
   """###(y_pred, w_pred_prob)### inputs are probabilities of the predictions (former MEI)"""
   index_new_experiment = max(np.argwhere(np.amax(pred_prob1,axis=1)  == np.min(np.amax(pred_prob1,axis=1)) ))[0]
   return index_new_experiment

def distance(X_train, X_test):
    dist = []
    for test in X_test:
        d=[]
        for train in X_train:
            #d+= np.linalg.norm((test,train), ord=None, axis=None, keepdims=False)
            d.append(np.sum(np.absolute(np.subtract(test,train)),axis=0))
        dist.append(sum(d))
    dist = np.array(dist)
    index_new_experiment = dist.argmax(axis=0)
    return index_new_experiment


#combination of distance measure and uncertainty sampling strategy
def distances_of_selected(X_train, X_test,indexes):
    dist = []
    for test in X_test:
        d=[]
        for train in X_train:
            #d+= np.linalg.norm((test,train), ord=None, axis=None, keepdims=False)
            d.append(np.sum(np.absolute(np.subtract(test,train)),axis=0))
        #dist.append(min(d))
        dist.append(sum(d))
    dist_indexes = []
    for i in indexes:
        dist_indexes.append(dist[i])
    dist_indexes = np.array(dist_indexes)
    index_new_experiment = dist_indexes.argmax(axis=0) 
    index_new_experiment = indexes[index_new_experiment]
    return index_new_experiment
    
def least_confident_distance(pred_prob1,pred_prob,X_train,X_test,X_val):
   """###(y_pred, w_pred_prob)### inputs are probabilities of the predictions (former MEI)"""
   probabilityvector = np.amax(pred_prob1,axis=1)
   p = probabilityvector
   values = []
   threshold = (len(X_test)*5)//100
   for i in range(threshold):
       try:
           t = np.amin(p)
       except Exception:
           break
       values.append(t)
       s = np.argwhere(p == np.min(p))
       p = np.delete(p, s)
       #break###
   indexes = []  
   for v in values:
       if len(indexes)<threshold:
           indexes = indexes + list(np.ravel(np.argwhere(probabilityvector == v)))
       else:
           break
   return indexes

def entropy_distance(pred_prob1,pred_prob,X_train,X_test,X_val):
   """###(y_pred, w_pred_prob)### inputs are probabilities of the predictions (former MEI)"""
   probabilityvector = -np.sum(np.multiply(pred_prob,np.log2(pred_prob,where= 0<pred_prob)),axis=1)
   p = probabilityvector
   values = []
   threshold = (len(X_test)*5)//100
   for i in range(threshold):
       try:
           t = np.amax(p)
       except Exception:
           break
       values.append(t)
       s = np.argwhere(p == np.amax(p))
       p = np.delete(p, s)
       #break###
   indexes = []  
   for v in values:
       if len(indexes)<threshold:
           indexes = indexes + list(np.ravel(np.argwhere(probabilityvector == v)))
       else:
           break
   return indexes

"""
def margin(pred_prob1,pred_prob):
    if len(pred_prob) != 0:
        pred_prob1_sec = []
        pred_prob_sec = []
        for i in pred_prob1:
            pred_prob1_sec.append(np.unique(i, axis=0)[-2])
        for i in pred_prob:
            pred_prob_sec.append(np.unique(i, axis=0)[-2])
        pred_prob1_sec= np.array(pred_prob1_sec)
        pred_prob_sec= np.array(pred_prob_sec)
        
        index_new_experiment= np.argwhere(np.add(np.add(np.amax(pred_prob,axis=1),np.amax(pred_prob,axis=1) ),np.add(pred_prob1_sec,pred_prob_sec )  ) == np.max(np.add(np.add(np.amax(pred_prob1,axis=1),np.amax(pred_prob,axis=1) ),np.add(pred_prob1_sec,pred_prob_sec )  )) )
    else:
        index_new_experiment= [max(lst)-np.unique(lst)[-2] for lst in pred_prob1].argmax(axis=0)
    return index_new_experiment
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:07:45 2021

@author: kbock
"""
import numpy as np
from sklearn.model_selection import train_test_split

def create_train_test_set(X,y,train_set_size, ran_state=None): #,train_target_bound_level =0.5): 
    x_updated_index_array_test = np.arange(len(y))
    y_train = []
    more_train_set_size = train_set_size
    X_test = X
    X_train_0 = []
    #werkstoffid_train = []
    while  more_train_set_size != 0:
        X_train, X_test, x_updated_index_array_train, x_updated_index_array_test = train_test_split(X_test, x_updated_index_array_test, train_size=more_train_set_size, random_state=ran_state)
        
        try:
            X_train_0 = np.concatenate((X_train_0, X_train))
        except:
            X_train_0 = X_train
           
        for updated_index, j in zip(np.flipud(x_updated_index_array_train), reversed(range(len(x_updated_index_array_train)))):
            for all_indices in range(len(y)):
                if updated_index == all_indices:
                    #if y[updated_index] < train_target_bound_level*y.max(axis=0):
                    y_train.append(y[updated_index])
#                     else:
#                         X_test = np.concatenate((X_test , np.reshape(X_train[j],(1,22))))
#                         x_updated_index_array_test = np.concatenate((x_updated_index_array_test , np.reshape(x_updated_index_array_train[j],(1,))))
#                         X_train_0 = np.delete(X_train_0, obj = j, axis=0)
#                         x_updated_index_array_train = np.delete(x_updated_index_array_train, obj = j, axis=0)
                        
        more_train_set_size = train_set_size - len(y_train)
        X = X_test
        #x_updated_index_array = x_updated_index_array_test
        
    y_test = []
    for updated_index in x_updated_index_array_test:
        for all_indices in range(len(y)):
            if updated_index == all_indices:
                y_test.append(y[all_indices])
#     for updated_index in np.flipud(x_updated_index_array_train):
#         for all_indices in range(len(y)):
#             if updated_index == all_indices:
#                 if y[updated_index] >= train_target_bound_level*y.max(axis=0):
#                     y_test.append(y[updated_index])
    
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
        
    return X_train_0, X_test, y_train, y_test#, x_updated_index_array_train, x_updated_index_array_test
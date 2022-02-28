import numpy as np
import pandas as pd
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling,consensus_entropy_sampling,max_disagreement_sampling
n_splits = 6 #KFold splits
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score


method = ["vote_entropy_sampling"]#"consensus_entropy_sampling","max_disagreement_sampling"]
zoom = [100]
m =RandomForestClassifier()
#m = MLPClassifier( alpha=1e-5, random_state=1,max_iter=100)
train_set_size = [8]
for med in method:
    for epoch in range(0,5): 
        print(epoch)
        kf = KFold(n_splits=n_splits,shuffle=True,random_state=0)
        for z in zoom:
            print("###############################################\n###############################################")
            table = pd.DataFrame(data=[])
            table2 = pd.DataFrame(data=[])
            table3 = pd.DataFrame(data=[])
            table4 = pd.DataFrame(data=[])
            col_names = []
            col_names2 = []
            col_names3 = []
            col_names4 = []
            
            col_names.append("{a}{b}".format(a=m,b=med))
            col_names2.append("{a}{b}".format(a=m,b=med))
            col_names3.append("{a}{b}".format(a=m,b=med))
            col_names4.append("{a}{b}".format(a=m,b=med))
            for tss in train_set_size:
                VGG_path = "C:/Users/kbock/Desktop/Arbeitsordner/data/processed_data/{t}zoom_224_224(12)_clust/{t}zoom_VGG16_conv33_mean_standardized.csv".format(t=z)
                features_and_targets = pd.read_csv(VGG_path)
                X_raw = features_and_targets.drop(["Bruchart-id","Material_cluster"], axis=1).values
                y = features_and_targets[["Bruchart-id"]]
                w = features_and_targets[["Material_cluster"]]
                y = np.array(y)
                w = np.array(w)
                
                X_split = np.arange(len(X_raw)/12)
                for train_index, test_index in kf.split(X_split):
                    train_index = np.multiply(train_index,12)
                    app= []
                    for t in train_index:
                        app.append(t+1)
                        app.append(t+2)
                        app.append(t+3)
                        app.append(t+4)
                        app.append(t+5)
                        app.append(t+6)
                        app.append(t+7)
                        app.append(t+8)
                        app.append(t+9)
                        app.append(t+10)
                        app.append(t+11)
                    train_index = np.append(train_index,app)
                    test_index = np.multiply(test_index,12)
                    app= []
                    for t in test_index:
                        app.append(t+1)
                        app.append(t+2)
                        app.append(t+3)
                        app.append(t+4)
                        app.append(t+5)
                        app.append(t+6)
                        app.append(t+7)
                        app.append(t+8)
                        app.append(t+9)
                        app.append(t+10)
                        app.append(t+11)
                    test_index = np.append(test_index,app)
                        
                    X = X_raw
                    X0 = X[train_index]
                    y0 = y[train_index]
                    w0 = w[train_index]
                    X_val = X[test_index]
                    y_val = y[test_index]
                    w_val = w[test_index] 
                    ##########
                    # initializing Committee members
                    n_members = 2
                    learner_list = list()
                    for member_idx in range(n_members):
                        # initial training data
                        n_initial = 8
                        #train_idx = np.random.choice(range(X0.shape[0]), size=n_initial, replace=False)
                        X_train, X_test, train_idx, x_updated_index_array_test = train_test_split(X0, range(X0.shape[0]), train_size=n_initial,shuffle=True,random_state=epoch)
                               
                        X_train = X0[train_idx]
                        y_train = y0[train_idx]
                        w_train = w0[train_idx]
                        y_train = np.reshape(np.array(y_train),(-1,))                    
                        w_train = np.reshape(np.array(w_train),(-1,))   
                        # creating a reduced copy of the data with the known instances removed
                        X0 = np.delete(X0, train_idx, axis=0)
                        y0 = np.delete(y0, train_idx)
                        w0 = np.delete(w0, train_idx)               
                        # initializing learner
                        learner = ActiveLearner(
                            estimator=m,
                            X_training=X_train, y_training=w_train
                        )
                        learner_list.append(learner)                    
                    # assembling the committee
                    #print(eval(med))
                    committee = Committee(learner_list=learner_list,
                                          query_strategy=eval(med))
            
                    col_table = []
                    col_table2 = []
                    col_table3 = []
                    col_table4 = []
                    # query by committee
                    n_queries = 120
                    for idx in range(n_queries):
                        #print(X0.shape)
                        #try: 
                        query_idx, query_instance = committee.query(X0)
                     
                        committee.teach(
                            X=X0[query_idx].reshape(1, -1),
                            y=w0[query_idx].reshape(1, )#TEACH
                            #w=w0[query_idx].reshape(1, )
                        )
                        #add queried instance to labeled set
                        X_train =  np.concatenate((X_train,X0[query_idx].reshape(1, -1)), axis=0)   
                        y_train =  np.concatenate((y_train,y0[query_idx].reshape(1, )), axis=0)   
                        w_train =  np.concatenate((w_train,w0[query_idx].reshape(1, )), axis=0)  
                        #print(w0[query_idx].reshape( -1))
                        #remove queried instance from pool
                        X0 = np.delete(X0, query_idx, axis=0)
                        y0 = np.delete(y0, query_idx) 
                        w0 = np.delete(w0, query_idx)
                        #calculate performance metrics
                        
                        m1 = m
                        m1.fit(X_train, y_train)
                        y_pred_val = m1.predict(X_val)
                        m2 = m
                        m2.fit(X_train, w_train)
                        w_pred_val = m2.predict(X_val)  
                        acc_val = balanced_accuracy_score(y_val,y_pred_val)
                        f1 = f1_score(y_val,y_pred_val)
                        cluster_f1 = f1_score(w_val,w_pred_val, average ='weighted')
                        cluster_acc = balanced_accuracy_score(w_val,w_pred_val)
                        
                        
                        col_table = np.append(col_table,acc_val)
                        col_table = np.asarray(col_table)
                        col_table2 = np.append(col_table2,cluster_acc)
                        col_table2 = np.asarray(col_table2) 
                        col_table3 = np.append(col_table3,f1)
                        col_table3 = np.asarray(col_table3)
                        col_table4 = np.append(col_table4,cluster_f1)
                        col_table4 = np.asarray(col_table4)
    
                    try:
                        col_table0 = np.add(col_table0,col_table)
                    except Exception:
                        col_table0 = col_table
                    try:
                        col_table02 = np.add(col_table02,col_table2)
                    except Exception:
                        col_table02 = col_table2
                    try:
                        col_table03 = np.add(col_table03,col_table3)
                    except Exception:
                        col_table03 = col_table3
                    try:
                        col_table04 = np.add(col_table04,col_table4)
                    except Exception:
                        col_table04 = col_table4
            #save k-fold mean in excel-spreadsheet
            col_table0 = pd.DataFrame(data=col_table0)
            table = pd.concat([table,col_table0], axis=1)
            col_table02 = pd.DataFrame(data=col_table02)
            table2 = pd.concat([table2,col_table02], axis=1)
            col_table03 = pd.DataFrame(data=col_table03)
            table3 = pd.concat([table3,col_table03], axis=1)
            col_table04 = pd.DataFrame(data=col_table04)
            table4 = pd.concat([table4,col_table04], axis=1)      
            table_arr = table.to_numpy()
            table_arr2 = table2.to_numpy()
            table_arr3 = table3.to_numpy()
            table_arr4 = table4.to_numpy()
            table_array = np.divide(table_arr,(n_splits))
            table_array2 = np.divide(table_arr2,(n_splits))
            table_array3 = np.divide(table_arr3,(n_splits))
            table_array4 = np.divide(table_arr4,(n_splits))
            table = pd.DataFrame(data=table_array)
            table2 = pd.DataFrame(data=table_array2) 
            table3 = pd.DataFrame(data=table_array3)
            table4 = pd.DataFrame(data=table_array4)
            table.columns = col_names
            table2.columns = col_names2
            table3.columns = col_names3
            table4.columns = col_names4
            writer = pd.ExcelWriter("C:/Users/kbock/Desktop/Arbeitsordner/results/{}_/cluster/bacc/{}/{}.xlsx".format(z,med,epoch))
            table.to_excel(writer)
            writer.save()
            writer2 = pd.ExcelWriter("C:/Users/kbock/Desktop/Arbeitsordner/results/{}_/cluster/bacc_cluster/{}/{}.xlsx".format(z,med,epoch))
            table2.to_excel(writer2)
            writer2.save()
            writer3 = pd.ExcelWriter("C:/Users/kbock/Desktop/Arbeitsordner/results/{}_/cluster/f1/{}/{}.xlsx".format(z,med,epoch))
            table3.to_excel(writer3)
            writer3.save()
            writer4 = pd.ExcelWriter("C:/Users/kbock/Desktop/Arbeitsordner/results/{}_/cluster/f1_cluster/{}/{}.xlsx".format(z,med,epoch))
            table4.to_excel(writer4)
            writer4.save()
    
        
       
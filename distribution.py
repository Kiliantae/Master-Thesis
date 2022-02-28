# compute mean and std from all (5) AL runs (resulting from 6-fold cross-validation) and save it in an excel sheet
import pandas as pd
import os
per= ["bacc","bacc_cluster","f1","f1_cluster"]
for p in per:
    #results_path = 'C:/Users/kbock/Desktop/Arbeitsordner/results/random/100/MLP/{}/'.format(p)
    results_path = 'C:/Users/kbock/Desktop/Arbeitsordner/results/100_/frac/{}/'.format(p)
    for file in os.listdir(results_path):
        table1 = pd.DataFrame(data=[])
        file_path = os.path.join(results_path + file)
        for file_path2 in os.listdir(file_path):
            file_path3 = os.path.join(file_path + "/"+file_path2)
            df = pd.read_excel(file_path3)
            try:
                df = df.drop(["Unnamed: 0"], axis=1)
            except Exception:
                pass            
            table1 = pd.concat([table1,df], axis=1)
                  
        t = pd.concat([table1.std(axis=1),table1.mean(axis = 1)], axis=1,keys=['{}_std'.format(file), '{}_mean'.format(file)])
        print("################")  
        print(file)    
        writer = pd.ExcelWriter("C:/Users/kbock/Desktop/Arbeitsordner/results_distribution/100/frac/{}/{}.xlsx".format(p,file))
        t.to_excel(writer)
        writer.save()
    
        
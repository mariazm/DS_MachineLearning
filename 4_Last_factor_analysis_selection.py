import pandas as pd 
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import  precision_score, recall_score,f1_score, brier_score_loss,roc_curve, auc, accuracy_score
from sklearn.calibration import  calibration_curve


def fit_model(X_train, k ):
    '''
    This function calculates returns a factor object
    '''
    fa  = decomposition.FactorAnalysis(n_components= k, copy= True, random_state=1)
    fa.fit(X_train)
    return fa 

def create_factor_loads(X_train, k ):
    '''
    This function calculates the betas of the factor analysis, the weights 
    '''
    fa =  fit_model(X_train, k) 
    #Factor loads are the "betas" of the linear combination
    data_loads= pd.DataFrame(fa.components_)
    return data_loads

def create_factor_score(X_train, X_test, k ):
    '''
    This function returns the transformation of X_train and X_test datasets to k factors which will 
    be used later to modeling
    '''
    fa =  fit_model(X_train, k) 
    fa_tr = pd.DataFrame(fa.transform(X_train))
    fa_te = pd.DataFrame(fa.transform(X_test))
    return fa_tr, fa_te


def cross_validation(x_train,y_train, n_folds=5 ):
    # DB received is the training set

    models = {'logreg': LogisticRegression(), 'dectree': DecisionTreeClassifier(), 'rf':RandomForestClassifier()}

    params = {'logreg':{'C':[10**i for i in range(-2, 2)], 'penalty':['l1', 'l2']},      
              'dectree':{'min_samples_leaf':[50, 100, 500, 1000], 'criterion':['entropy']} ,
              'rf':{'n_estimators':[100, 200, 500, 1000], 'criterion':['entropy']}}
   
    #1. Split the and prepare data
    kfolds = KFold(x_train.shape[0], n_folds = n_folds)
    
    best_models = {}
    for classifier in models.keys():
        
        best_models[classifier] = GridSearchCV(models[classifier], params[classifier], cv = kfolds, scoring = 'roc_auc') 
        best_models[classifier].fit(x_train, y_train)   

    return best_models


def define_best_models_k(X_train, X_test,  Y_train, k, n_folds =5):
    fa_train, fa_test = create_factor_score(X_train, X_test, k)
    best_models =  cross_validation(fa_train, Y_train, n_folds)
    return best_models, fa_test , fa_train


def k_loop(X_train, X_test, Y_train, Y_test, list_k, n_folds=5):
    results = []
    models = []
    for k in list_k:
        best_models, fa_test, fa_train = define_best_models_k(X_train, X_test, Y_train, k, n_folds)
        for model in best_models.keys() :
            params  =  best_models[model].best_params_
            auc_cv =  best_models[model].best_score_
            #Test
            pred = best_models[model].predict_proba(fa_test)[:,1]
            fpr, tpr, thresholds = roc_curve(Y_test, pred)    
            roc_auc = auc(fpr, tpr)
            acc =  accuracy_score(Y_test, best_models[model].predict(fa_test))
            
            #Train
            pred1 = best_models[model].predict_proba(fa_train)[:,1]
            fpr1, tpr1, thresholds1 = roc_curve(Y_train, pred1)    
            roc_auc1 = auc(fpr1, tpr1)      
            acc1 =  accuracy_score(Y_train, best_models[model].predict(fa_train))
            
            results.append([k, model, roc_auc, roc_auc1, auc_cv, acc, acc1, params])
        models.append([k, best_models])
    return results, models

def get_list_per_factor(loads, X_columns):
    #Getting loads and updating col names
    var = loads
    var.columns =  X_columns
    
    #Get the absolute value of the load and its maximum per var
    var_abs =  pd.DataFrame(abs(var))
    idx_max =  var_abs.idxmax()
    
    var_load=[]
    zero_coef = []
    for i in range(var.shape[0]):
        set_idx =  pd.DataFrame(idx_max[idx_max ==i])
        set_idx.reset_index(inplace=True)
        var_load.append(list(set_idx['index']))

    return var_load, idx_max


if __name__ == '__main__':
	#Change file name and locations (Path)
	data_train = pd.read_pickle( "../Preprocessing/Factor/Datasets/Mod1/1_train_acp12_cat")
	data_test = pd.read_pickle( "../Preprocessing/Factor/Datasets/Mod1/1_test_acp12_cat")

	#In the algorithm only X must be inputted if there are more, select them
	X_train =  data_train[data_train.columns[2:]]
	Y_train = data_train["target"]

	X_test =  data_test[data_test.columns[2:]]
	Y_test = data_test["target"]
   
   	k_list1 =  [10,13, 16, 150, 200, 500]
   	results_1, models_1 =  k_loop(X_train, X_test, Y_train, Y_test,k_list1 , n_folds=5)
   	print pd.DataFrame(results_1, columns = ["K", "BestModel", "AUC_test", "AUC_train", "AUC_CV", "Acc_test", "Acc_train", "params"])

	##Generating input for Mary
	loads_1 = create_factor_loads(X_train, 16)
	var_load1, idx_max1  = get_list_per_factor(loads_1, X_train.columns)
	var_load1 = pd.DataFrame(var_load1).transpose()
	var_load1.to_csv("varlat16_max_mod1.csv")


    
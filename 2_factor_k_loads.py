import pandas as pd 
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt

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


def get_max2_factor(var_abs, idx_max):

	idx_max2_total = pd.Series()
	for i in range(var_abs.shape[1]):
		v1 =  var_abs[var_abs.columns[i]]
		v1[idx_max[i]] = -1
		idx_max2 =  v1.idxmax()
		idx_max2_total = idx_max2_total.set_value(var_abs.columns[i], idx_max2)

	print idx_max2_total.shape
	return idx_max2_total

def get_list_per_factor(loads, X_columns):
    #Getting loads and updating col names
    var = loads
    var.columns =  X_columns
    
    #Get the absolute value of the load and its maximum per var
    var_abs =  pd.DataFrame(abs(var))
    idx_max1 =  var_abs.idxmax()
    
    idx_max2 = get_max2_factor(var_abs, idx_max1)

    var_load1=[]
    var_load2 = []
    zero_coef = []
    for i in range(var.shape[0]):
        set_idx1 =  pd.DataFrame(idx_max1[idx_max1 ==i])
        set_idx1.reset_index(inplace=True)
        var_load1.append(list(set_idx1['index']))

        set_idx2 =  pd.DataFrame(idx_max2[idx_max2 ==i])
        set_idx2.reset_index(inplace=True)
        var_load2.append(list(set_idx2['index']))

    return var_load1, var_load2, idx_max1, idx_max2


if __name__ == '__main__':
    #Change file name and locations (Path)
    print "Task: Generating load for K pre selected"
    data_train = pd.read_pickle( "../2.Clean_TrainTest/1_train_rev_cat")
    data_test = pd.read_pickle( "../2.Clean_TrainTest/1_test_rev_cat")

    print "Datasets charged...Check the shapes"
    print "Train %s" %(str(data_train.shape))
    print "Test %s" %(str(data_test.shape))

	#In the algorithm only X must be inputted if there are more, select them
    X_train =  data_train[data_train.columns[2:]]
    Y_train = data_train["target"]

    X_test =  data_test[data_test.columns[2:]]
    Y_test = data_test["target"]
   
    print "Write K to generate loads, if many imputted separate by commas"
    k_sel = int(raw_input())

	##Generating input for Mary
    loads_1 = create_factor_loads(X_train, k_sel)
    var_load1, var_load2, idx_max1 , idx_max2 = get_list_per_factor(loads_1, X_train.columns)
    var_load1 = pd.DataFrame(var_load1).transpose()
    var_load2 = pd.DataFrame(var_load2).transpose()
    var_load1.to_csv("mod1_loads1_k%s_.csv" %(k_sel))
    var_load2.to_csv("mod1_loads2_k%s_.csv" %(k_sel))



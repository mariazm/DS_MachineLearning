import FeatureImportance as feature_importance  #Another of the scripts in the file, must be in the same path as this script
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd 
from sklearn.metrics import  precision_score, recall_score,f1_score, brier_score_loss,roc_curve, auc, accuracy_score


if __name__ == '__main__':
    data_train = pd.read_pickle( "./../2.Clean_TrainTest/2_train_rev_cat")
    data_test = pd.read_pickle( "./../2.Clean_TrainTest/2_test_rev_cat")

    #In the algorithm only X must be inputted if there are more, select them
    X_train =  data_train[data_train.columns[2:]]
    Y_train = data_train["target"]

    X_test =  data_test[data_test.columns[2:]]
    Y_test = data_test["target"]

    #First general
    rf_importance = feature_importance.FeatureImportance('random_forest')
    fi = rf_importance.calculate_importance(X_train, Y_train)
    #Keep on general
    print "Plot of importance Random Forest: remember that due to its randomenes results will not be exactly the same each time this module is run!"
    rf_importance.plot_importance()


    #Baseline model 
    las_m2 =  linear_model.LogisticRegression(penalty='l1')
    las_m2.fit(X_train, Y_train)
    pred = las_m2.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(Y_test, pred)    
    roc_auc = auc(fpr, tpr)
    acc =  accuracy_score(Y_test, las_m2.predict(X_test))
    print "Test indicators for a Lasso:  AUC %s, Accuracy: %s" %(roc_auc, acc)
    coef = pd.DataFrame(las_m2.coef_).transpose()
    coef["col"]  =  X_train.columns
    coef.columns = ["beta", "col"]
    selected_coef = list(coef[coef["beta"]!=0]["col"])
    print "Number of non-zero coefficients for the Lasso"
    print len(selected_coef)

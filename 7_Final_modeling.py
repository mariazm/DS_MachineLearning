import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from factor_analysis_kselection import define_best_models_k

def correlation(data, success, fi_pairs):
	corr_success = data.corr()[success].drop(success, axis=0)
	pos_corr = pd.DataFrame(corr_success > 0)
	neg_corr = pd.DataFrame(corr_success <= 0)
	importance_dict = dict([(f, i) for (i, f) in fi_pairs])
	res = pd.DataFrame.from_dict(importance_dict, orient='index')
	res.columns = ['feature_importance']
	res['pos_corr'] = (corr_success > 0)
	res['neg_corr'] = (corr_success <= 0)
	res = res.sort(columns=['feature_importance'], ascending=True)
	pos_features = res['feature_importance'] * res['pos_corr']
	neg_features = res['feature_importance'] * res['neg_corr']
	return pos_features, neg_features

def plot_rf_selected_importance(data, success, fi_pairs):
	pos_features, neg_features = correlation(data, success, fi_pairs)
	training_columns = [f for (i, f) in fi_pairs]
	y_pos = np.arange(len(training_columns))
	plt.figure(figsize=(6,7))
	ax = plt.subplot()
	ax.barh(y_pos, pos_features, height=0.4, color='r', label = 'Positive')
	ax.barh(y_pos, neg_features, height=0.4, color='b', label = 'Negative')
	ax.set_yticks(y_pos)
	ax.set_yticklabels(training_columns, size = 11)
	plt.grid(axis='both')
	plt.xlabel('Signed Feature Importance')
	plt.ylabel('Feature Name')
	plt.title('Is the correlation with the target variable positive or negative? -Model 2 RF')
	plt.legend(loc='lower right')
	plt.show()


if __name__ == '__main__':
    data_train = pd.read_pickle( "./../2.Clean_TrainTest/2_train_rev_cat")
    data_test = pd.read_pickle( "./../2.Clean_TrainTest/2_test_rev_cat")

    #In the algorithm only X must be inputted if there are more, select them
    X_train =  data_train[data_train.columns[2:]]
    Y_train = data_train["target"]

    X_test =  data_test[data_test.columns[2:]]
    Y_test = data_test["target"]

    #Applying final model 
    print "Input k selected"
    k_in = int(raw_input())
    models, fa_test, fa_train = define_best_models_k(X_train, X_test,  Y_train, k_in, n_folds =5)
    rf_fit = models['rf'].best_estimator_
    fa_train.columns = ["FA%i"%i for i in range(1,(k_in+1))]
    fi_pairs = [(i, j) for (i, j) in zip(rf_fit.feature_importances_, list(fa_train.columns))]
    fi_pairs.sort()

    data = fa_train.copy()
    data['success'] = Y_train
    plot_rf_selected_importance(data, 'success', fi_pairs)
    print "Plotted!"

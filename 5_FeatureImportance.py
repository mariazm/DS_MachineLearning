import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class FeatureImportance:
    def __init__(self, method):
        self.method = method
        self.importance = None

    def set_importance(self, importance):
        self.importance = importance

    def calculate_importance(self, X_train, Y_train):
        if self.method == 'random_forest':
            rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
        elif self.method == 'decision_tree':
            rf_model = DecisionTreeClassifier(min_samples_leaf=500, min_samples_split=2, criterion='entropy')
        else:
            raise NotImplementedError('method {} was not recognized'.format(self.method))
        rf_fit = rf_model.fit(X_train, Y_train)
        fi_pairs = [(i, j) for (i, j) in zip(rf_fit.feature_importances_, list(X_train.columns))]
        fi_pairs.sort()
        self.set_importance(fi_pairs)
        return fi_pairs

    def plot_importance(self):
        training_columns = [f for (i, f) in self.importance][-11:-1]
        y_pos = np.arange(len(training_columns))
        plt.figure(figsize=(6,7))
        plt.barh(y_pos, [i for (i, f) in self.importance][-11:-1])
        plt.yticks(y_pos, training_columns, size = 11)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title('Which are the most informative features?')
        plt.show()


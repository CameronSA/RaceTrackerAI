import Constants
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_graphviz
import graphviz

class DecisionTree:
    def __init__(self):
        print("Fetching dataset from path '" + str(Constants.RACE_SUMMARY_DATA_FILEPATH) + "'")
        self.data = pd.read_csv(Constants.RACE_SUMMARY_DATA_FILEPATH)
        self.data['Prize'] = self.data['Prize'].map(lambda x: x.lstrip('£').lstrip('€'))
        self.data['Going'] = self.data['Going'] == 'Good'
        columns = ['Going','Prize','Favourite Won','Favourite Odds','Second Favourite Odds','Average Odds Of Others','StdDev Odds Of Others','Number Of Horses','Class','NH','Handicap','Novice']
        features=['Going','Prize','Favourite Odds','Second Favourite Odds','Average Odds Of Others','StdDev Odds Of Others','Number Of Horses','Class','NH','Handicap','Novice']
        self.data = self.data[columns].dropna()
        print('Building decision tree model. . .')
        X = self.data[features].values
        Y = self.data['Favourite Won'].values
        kf = KFold(n_splits = 20, shuffle = True)
        accuracy = []
        precision = []
        recall = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, criterion='gini')
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))

        print("\naccuracy:", np.mean(accuracy), "+/-", np.std(accuracy))
        print("precision:", np.mean(precision), "+/-", np.std(precision))
        print("recall:", np.mean(recall), "+/-", np.std(recall),"\n")
        
        feature_names = features
        dot_file = export_graphviz(dt, feature_names=feature_names)
        graph = graphviz.Source(dot_file)
        graph.render(filename='tree', format='png', cleanup=True)
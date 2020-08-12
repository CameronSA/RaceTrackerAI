import Constants
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import statistics as st
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
        #print('Building decision tree model. . .')
        print('Building random forest model. . .')
        X = self.data[features].values
        Y = self.data['Favourite Won'].values
        kf = KFold(n_splits = 20, shuffle = True)
        accuracy = []
        precision = []
        recall = []
        positivePredictionRates = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            #dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, criterion='gini')
            dt = RandomForestClassifier(max_depth = 3)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            positivePredictionRates.append(sum(y_pred)/(len(y_pred)-sum(y_pred)))
            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))

        avPositivePredictionRate = round(100*st.mean(positivePredictionRates),2)
        avPositivePredictionRateError = round(100*st.stdev(positivePredictionRates),2)
        precisionMean = np.mean(precision)
        precisionMeanError = np.std(precision)
        percentageTotalPredictions = round(avPositivePredictionRate*precisionMean,2)
        percentageTotalPredictionsErrorSquared = (precisionMean*avPositivePredictionRateError)**2 + (avPositivePredictionRate*precisionMeanError)**2
        percentageTotalPredictionsError = round(percentageTotalPredictionsErrorSquared**0.5,2)

        print("\nAccuracy:", np.mean(accuracy), "+/-", np.std(accuracy))
        print("Precision:", precisionMean, "+/-", precisionMeanError)
        print("Recall:", np.mean(recall), "+/-", np.std(recall),"\n")
        print("\nAverage positive prediction rate:", str(avPositivePredictionRate) + " +/- " + str(avPositivePredictionRateError) + " %")
        print("\nPercentage of total predictions that are correct positive predictions (rate x precision): ", str(percentageTotalPredictions) + " +/- " + str(percentageTotalPredictionsError) + " %")
        


        #feature_names = features
        #dot_file = export_graphviz(dt, feature_names=feature_names)
        #graph = graphviz.Source(dot_file)
        #graph.render(filename='tree', format='png', cleanup=True)
import warnings
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
from datetime import datetime
import PreProcessData as ppd
from tqdm import tqdm

class RandomForestModel:
    def __init__(self):
        #self.data['Prize'] = self.data['Prize'].map(lambda x: x.lstrip('£').lstrip('€'))
        ppd.PreProcessData()        
        #self.columns = ['Going','Favourite Won','Favourite Odds','Second Favourite Odds','Average Odds Of Others','StdDev Odds Of Others','Number Of Horses','Class','NH','Handicap','Novice','Race Index']
        #self.features = ['Going','Favourite Odds','Second Favourite Odds','Average Odds Of Others','StdDev Odds Of Others','Number Of Horses','Class','NH','Handicap','Novice','Race Index']
        print("Fetching preprocessed dataset from path '" + str(Constants.PREPROCESSED_DATA_FILEPATH) + "'")
        self.data = pd.read_csv(Constants.PREPROCESSED_DATA_FILEPATH)
        columns = [x for x in self.data.columns.tolist() if x != 'Date']
        self.columns = columns
        self.features = [x for x in columns if x != 'Favourite Won']

        self.MaxDepths=[]

        self.Accuracies=[]
        self.Precisions=[]
        self.Recalls=[]
        self.AveragePositivePredictionRates=[]
        self.AveragePositivePredictionRatesTimesPrecision=[]

        self.AccuraciesErrors=[]
        self.PrecisionsErrors=[]
        self.RecallsErrors=[]
        self.AveragePositivePredictionRatesErrors=[]
        self.AveragePositivePredictionRatesTimesPrecisionErrors=[]

    def Run(self):     
        print('Starting decision tree model. . .')
        dataSet = self.data
        dataSet = dataSet[self.columns].dropna()
        self.X = dataSet[self.features].values
        self.Y = dataSet['Favourite Won'].values        
        self.__BuildModel(5)

    def __BuildModel(self, maxDepth):
        accuracy = []
        precision = []
        recall = []
        positivePredictionRates = []
        positivePredictionAverageProbabilities =[]
        positivePredictionStdDevProbabilities =[]
        negativePredictionAverageProbabilities =[]
        negativePredictionStdDevProbabilities =[]

        positivePredictionAverageProbabilitiesDifference =[]
        positivePredictionStdDevProbabilitiesDifference =[]
        negativePredictionAverageProbabilitiesDifference =[]
        negativePredictionStdDevProbabilitiesDifference =[]
        kf = KFold(n_splits = 20, shuffle = True)
        for train_index, test_index in tqdm(kf.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            #dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, criterion='gini')
            dt = RandomForestClassifier(max_depth = maxDepth)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            y_pred_proba = dt.predict_proba(X_test)
            positivePredictionRates.append(sum(y_pred)/(len(y_pred)-sum(y_pred)))
            y_pred_proba_positive = []
            y_pred_proba_negative = []
            y_pred_proba_positive_difference = []
            y_pred_proba_negative_difference = []
            for item in y_pred_proba:                
                if item[1] > item[0]:
                    y_pred_proba_positive.append(item[1])
                    y_pred_proba_positive_difference.append(item[1]-item[0])
                elif item[0] > item[1]:
                    y_pred_proba_negative.append(item[0])
                    y_pred_proba_negative_difference.append(item[0]-item[1])
            positivePredictionAverageProbabilities.append(np.mean(y_pred_proba_positive))
            positivePredictionStdDevProbabilities.append(np.std(y_pred_proba_positive))
            negativePredictionAverageProbabilities.append(np.mean(y_pred_proba_negative))
            negativePredictionStdDevProbabilities.append(np.std(y_pred_proba_negative))

            positivePredictionAverageProbabilitiesDifference.append(np.mean(y_pred_proba_positive_difference))
            positivePredictionStdDevProbabilitiesDifference.append(np.std(y_pred_proba_positive_difference))
            negativePredictionAverageProbabilitiesDifference.append(np.mean(y_pred_proba_negative_difference))
            negativePredictionStdDevProbabilitiesDifference.append(np.std(y_pred_proba_negative_difference))

            #print(y_pred[0],y_pred_proba[0])
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

        positivePredictionProbabilityMean = round(100*np.mean(positivePredictionAverageProbabilities),2)
        positivePredictionProbabilityError = round(100*AddInQuadrature(positivePredictionStdDevProbabilities)/float(len(positivePredictionAverageProbabilities))**0.5,2)
        negativePredictionProbabilityMean = round(100*np.mean(negativePredictionAverageProbabilities),2)
        negativePredictionProbabilityError = round(100*AddInQuadrature(negativePredictionStdDevProbabilities)/float(len(negativePredictionAverageProbabilities))**0.5,2)

        positivePredictionProbabilityMeanDifference = round(100*np.mean(positivePredictionAverageProbabilitiesDifference),2)
        positivePredictionProbabilityErrorDifference = round(100*AddInQuadrature(positivePredictionStdDevProbabilitiesDifference)/float(len(positivePredictionAverageProbabilitiesDifference))**0.5,2)
        negativePredictionProbabilityMeanDifference= round(100*np.mean(negativePredictionAverageProbabilitiesDifference),2)
        negativePredictionProbabilityErrorDifference = round(100*AddInQuadrature(negativePredictionStdDevProbabilitiesDifference)/float(len(negativePredictionAverageProbabilitiesDifference))**0.5,2)

        print("Max Depth: "+str(maxDepth))
        print("\n\tAccuracy:", np.mean(accuracy), "+/-", np.std(accuracy))
        print("\tPrecision:", precisionMean, "+/-", precisionMeanError)
        print("\tRecall:", np.mean(recall), "+/-", np.std(recall))

        print("\n\tAverage positive prediction rate:", str(avPositivePredictionRate) + " +/- " + str(avPositivePredictionRateError) + " %")
        print("\tPercentage of total predictions that are correct positive predictions (rate x precision): ", str(percentageTotalPredictions) + " +/- " + str(percentageTotalPredictionsError) + " %")

        print("\n\tAverage positive prediction probability:",str(positivePredictionProbabilityMean) + " +/- " +str(positivePredictionProbabilityError))
        print("\tAverage negative prediction probability:",str(negativePredictionProbabilityMean) + " +/- " +str(negativePredictionProbabilityError))

        print("\n\tAverage difference between positive and negative result certainty for a positive prediction:",str(positivePredictionProbabilityMeanDifference) + " +/- " +str(positivePredictionProbabilityErrorDifference))
        print("\tAverage difference between negative and positive result certainty for a negative prediction:",str(negativePredictionProbabilityMeanDifference) + " +/- " +str(negativePredictionProbabilityErrorDifference))
        
def AddInQuadrature(array):
    sumSquares = 0.0
    for number in array:
        sumSquares += number**2
    return sumSquares**0.5
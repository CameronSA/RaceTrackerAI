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
        ppd.PreProcessData()        
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
        self.betAmount = 10.0

    def Run(self):     
        print('Starting random forest model. . .')
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

        totalBets=[]
        totalWinningsPlusStakes=[]
        totalNumberWins = []
        totalNumberLosses = []

        kf = KFold(n_splits = 20, shuffle = True)
        for train_index, test_index in tqdm(kf.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            #dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, criterion='gini')
            dt = RandomForestClassifier(max_depth = maxDepth)
            dt.fit(X_train, y_train)

            odds = pd.DataFrame(X_test, columns=self.features)['Favourite Odds'].values
            y_pred = dt.predict(X_test)
            y_pred_proba = dt.predict_proba(X_test)
            totalBets.append(self.betAmount*sum(y_pred))
            totalWinningsPlusStake = 0.0
            numberWins = 0
            numberLosses = 0           
            for ind in range(len(y_pred)):
                if y_pred[ind]:
                    if y_test[ind]:
                        totalWinningsPlusStake += ((self.betAmount*odds[ind]) + self.betAmount)
                        numberWins+=1
                    else:
                        numberLosses+=1
                        totalWinningsPlusStake -= self.betAmount
            totalWinningsPlusStakes.append(totalWinningsPlusStake)
            totalNumberWins.append(numberWins)                        
            totalNumberLosses.append(numberLosses)    
            
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

        avTotalBet = round(np.mean(totalBets),2)
        avTotalWinnings = round(np.mean(totalWinningsPlusStakes),2)
        avTotalBetError = round(np.std(totalBets),2)
        avTotalWinningsError = round(np.std(totalWinningsPlusStakes),2)
        avNetProfit = round(avTotalWinnings-avTotalBet,2)
        avNetProfitError = round(((avTotalBetError**2)+(avTotalWinningsError**2))**0.5,2)

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

        print("\n\tAverage total bet:","£" + str(avTotalBet) + " +/- " + str(avTotalBetError))
        print("\tAverage total winnings including stake return:","£" + str(avTotalWinnings) + " +/- " + str(avTotalWinningsError))
        print("\tAverage net profit:","£" + str(avNetProfit) + " +/- " + str(avNetProfitError))
        print("\tAverage return:",str(round(100*avNetProfit/avTotalBet,2)) + "%")

        print("\n\tAverage number of wins:",str(np.mean(totalNumberWins)) + " +/- " + str(np.std(totalNumberWins)))
        print("\tAverage number of losses:",str(np.mean(totalNumberLosses)) + " +/- " + str(np.std(totalNumberLosses)))
        
def AddInQuadrature(array):
    sumSquares = 0.0
    for number in array:
        sumSquares += number**2
    return sumSquares**0.5
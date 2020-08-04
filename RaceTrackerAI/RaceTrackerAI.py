import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pandas as pd
from datetime import datetime
from decimal import Decimal
import csv
import sys
import statistics as stats
import Constants
import CommonFunctions as cf
import DecisionTreeModel as dtm
import os.path 

def CreateRaceSummaryDataset():
    if os.path.isfile(Constants.INPUT_DATA_FILEPATH) == 0:
        print("File: " + str(Constants.INPUT_DATA_FILEPATH) + " not found. Cannot create dataset")
        exit()

    df = pd.read_csv(Constants.INPUT_DATA_FILEPATH)
    df['Position'] = df['Position'].str.strip() == '1'

    raceDict={}
    print("Getting dictionary of races. . .")
    for index, row in df.iterrows(): 
        dateStr = str(row['Date']) + ' ' + str(row['Time'])
        raceDate = datetime.strptime(dateStr,'%Y-%m-%d %H%M')
        rowList = [row['Going'],row['Surface'],row['Race Type'],row['Age'],row['Prize'],row['Distance'],row['Racetrack'],row['Position'],row['Horse Name'],row['Horse Age'],row['Horse Weight'],row['ISP'],row['Class'],row['NH'],row['Handicap'],row['Novice']]

        cf.AddToDict(raceDict,raceDate,rowList)

    raceDataDict={}
    for race in raceDict:

        try:
            going = str(raceDict[race][0][0])
            surface = str(raceDict[race][0][1])
            raceType = str(raceDict[race][0][2])
            age = str(raceDict[race][0][3])
            prize = str(raceDict[race][0][4])
            distance = str(raceDict[race][0][5])
            raceTrack = str(raceDict[race][0][6])
      
            oddsList = []
            numberHorses=0
            for i in range(len(raceDict[race])):            
                oddsList.append(raceDict[race][i][11])
                numberHorses+=1
            oddsList = list(map(float, oddsList))
            favOdds = min(oddsList)
            favIndex = oddsList.index(favOdds)
            favWon = raceDict[race][favIndex][7] == 1
            favName = raceDict[race][favIndex][8]
            favAge = raceDict[race][favIndex][9]
            favWeight = raceDict[race][favIndex][10]
    
            oddsList.remove(favOdds)
            secondFavOdds = min(oddsList)
            oddsList.remove(secondFavOdds)
            try:
                avOddsOfOthers = stats.mean(oddsList)
                oddsSpreadOfOthers = stats.pstdev(oddsList)
            except Exception as e:
                print('Stats error in average odds of others for race: '+str(race) + '. ', e)
                avOddsOfOthers=-1.0
                oddsSpreadOfOthers=0.0

            raceClass = raceDict[race][0][12]
            nationalHunt = raceDict[race][0][13] == 1
            handicap = raceDict[race][0][14] == 1
            novice = raceDict[race][0][15] == 1

            cf.AddToDict(raceDataDict,'Date',race)
            cf.AddToDict(raceDataDict,'Going',going)
            cf.AddToDict(raceDataDict,'Surface',surface)
            cf.AddToDict(raceDataDict,'Race Type',raceType)
            cf.AddToDict(raceDataDict,'Age',age)
            cf.AddToDict(raceDataDict,'Prize',prize)
            cf.AddToDict(raceDataDict,'Distance',distance)
            cf.AddToDict(raceDataDict,'Race Track',raceTrack)
            cf.AddToDict(raceDataDict,'Favourite Won',favWon)
            cf.AddToDict(raceDataDict,'Favourite Age',favAge)
            cf.AddToDict(raceDataDict,'Favourite Name',favName)
            cf.AddToDict(raceDataDict,'Favourite Odds',favOdds)
            cf.AddToDict(raceDataDict,'Second Favourite Odds',secondFavOdds)
            cf.AddToDict(raceDataDict,'Average Odds Of Others',avOddsOfOthers)
            cf.AddToDict(raceDataDict,'StdDev Odds Of Others',oddsSpreadOfOthers)
            cf.AddToDict(raceDataDict,'Number Of Horses',numberHorses)
            cf.AddToDict(raceDataDict,'Class',raceClass)
            cf.AddToDict(raceDataDict,'NH',nationalHunt)
            cf.AddToDict(raceDataDict,'Handicap',handicap)
            cf.AddToDict(raceDataDict,'Novice',novice)
        except Exception as err:
            print('Parsing error. Race: '+str(race)+' skipped', err);

    raceSummaryData = pd.DataFrame.from_dict(raceDataDict)
    print(raceSummaryData)
    raceSummaryData.to_csv(Constants.RACE_SUMMARY_DATA_FILEPATH,index=False)

def CheckRaceSummaryDatasetExists():
    if os.path.isfile(Constants.RACE_SUMMARY_DATA_FILEPATH): return True
    else: return False

def StartDecisionTreeModel():
    descisionTree = dtm.DecisionTree()

if len(sys.argv) > 1:
    if sys.argv[1].lower() == '-decisiontree': 
        if CheckRaceSummaryDatasetExists(): 
            print('\nDataset found at path: ' + str(Constants.RACE_SUMMARY_DATA_FILEPATH) + '. Starting decision tree model. . .')
            StartDecisionTreeModel()
        else: 
            print('\nDataset not found at path: ' + str(Constants.RACE_SUMMARY_DATA_FILEPATH) + '. Creating. . .') 
            CreateRaceSummaryDataset()
            StartDecisionTreeModel()
    elif sys.argv[1].lower() == "-getdataset": CreateRaceSummaryDataset()
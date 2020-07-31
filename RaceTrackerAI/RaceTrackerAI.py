import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
import csv
import sys
import statistics as stats

def AddToDict(dict,key,value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]

df = pd.read_csv("C:\\Users\\camme\\Desktop\\RaceData\\OutputProcessedRaceDataExtraCols.csv")
df['Position'] = df['Position'].str.strip() == '1'

raceDict={}
print("Getting dictionary of races. . .")
for index, row in df.iterrows(): 
    dateStr = str(row['Date']) + ' ' + str(row['Time'])
    raceDate = datetime.strptime(dateStr,'%Y-%m-%d %H%M')
    rowList = [row['Going'],row['Surface'],row['Race Type'],row['Age'],row['Prize'],row['Distance'],row['Racetrack'],row['Position'],row['Horse Name'],row['Horse Age'],row['Horse Weight'],row['ISP'],row['Class'],row['NH'],row['Handicap'],row['Novice']]

    AddToDict(raceDict,raceDate,rowList)

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

        AddToDict(raceDataDict,'Going',going)
        AddToDict(raceDataDict,'Surface',surface)
        AddToDict(raceDataDict,'Race Type',raceType)
        AddToDict(raceDataDict,'Age',age)
        AddToDict(raceDataDict,'Prize',prize)
        AddToDict(raceDataDict,'Distance',distance)
        AddToDict(raceDataDict,'Race Track',raceTrack)
        AddToDict(raceDataDict,'Favourite Won',favWon)
        AddToDict(raceDataDict,'Favourite Name',favName)
        AddToDict(raceDataDict,'Favourite Odds',favOdds)
        AddToDict(raceDataDict,'Second Favourite Odds',secondFavOdds)
        AddToDict(raceDataDict,'Average Odds Of Others',avOddsOfOthers)
        AddToDict(raceDataDict,'StdDev Odds Of Others',oddsSpreadOfOthers)
        AddToDict(raceDataDict,'Number Of Horses',numberHorses)
        AddToDict(raceDataDict,'Class',raceClass)
        AddToDict(raceDataDict,'NH',nationalHunt)
        AddToDict(raceDataDict,'Handicap',handicap)
        AddToDict(raceDataDict,'Novice',novice)
    except Exception as err:
        print('Parsing error. Race: '+str(race)+' skipped', err);

raceSummaryData = pd.DataFrame.from_dict(raceDataDict)
print(raceSummaryData)
raceSummaryData.to_csv("C:\\Users\\camme\\Desktop\\RaceData\\RaceSummaryData.csv");

    

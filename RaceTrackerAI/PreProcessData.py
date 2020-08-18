import Constants
import os
import os.path
import pandas as pd
from datetime import datetime
import re
from tqdm import tqdm

class PreProcessData:
    def __init__(self):   
        print('Started data preprocessing. . .')
        if(self.__CheckDatasetExists()):
            print("Preprocessed dataset found at '"+Constants.PREPROCESSED_DATA_FILEPATH+"'. Overwrite? (y/n): ",end='')
            if input().lower() == 'y':
                self.__PreProcessData()
        else :
            self.__PreProcessData()
        
        print('Data preprocessing complete.')
    
    def __PreProcessData(self):        
        print("Fetching dataset from path '" + str(Constants.RACE_SUMMARY_DATA_FILEPATH) + "'")
        self.data = pd.read_csv(Constants.RACE_SUMMARY_DATA_FILEPATH)    
        self.__IndexRaces()
        self.__FormatDistance()
        self.__SplitOnGoing()       
        self.__SplitOnRaceType()
        self.__SplitOnRaceTrack()
        self.__DropUnwantedColumns()
        print("\nWriting preprocessed data to '"+str(Constants.PREPROCESSED_DATA_FILEPATH)+"'")
        self.data.to_csv(Constants.PREPROCESSED_DATA_FILEPATH, index=None, sep=',', mode='w')

    def __IndexRaces(self):
        print('\nIndexing races. . .')
        self.data['Race Index'] = -1
        self.data=self.data.sort_values(['Race Track','Date'])
        index=0
        prevDate = datetime.min
        prevTrack = "123"
        for ind, row in tqdm(self.data.iterrows()):
            currentDateWithTime = datetime.strptime(row['Date'],'%Y-%m-%d %H:%M:%S')
            currentDate = datetime.date(currentDateWithTime)
            currentTrack = row['Race Track']
            if currentDate != prevDate or currentTrack != prevTrack:
                index=0
                prevDate = currentDate
                prevTrack = currentTrack
            else:
                index+=1
            self.data.loc[ind,'Race Index'] = index

    def __SplitOnGoing(self):       
        print('\nCreating going columns. . .')
        uniqueGoings = sorted(set(self.data['Going']))      
        print('\tInitialising cells. . .')
        for going in tqdm(uniqueGoings):
            self.data[going] = False
        print('\tPopulating cells. . .')
        for going in tqdm(uniqueGoings):
            for ind, row in self.data.iterrows():
                if going == row['Going']:
                    self.data.loc[ind,going] = True

    def __SplitOnRaceType(self):
        print('\nCreating race type columns. . .')
        uniqueRaceTypes = sorted(set(self.data['Race Type']))   
        print('\tInitialising cells. . .')
        for raceType in tqdm(uniqueRaceTypes):
            self.data[raceType] = False
        print('\tPopulating cells. . .')
        for raceType in tqdm(uniqueRaceTypes):
            for ind, row in self.data.iterrows():
                if raceType == row['Race Type']:
                    self.data.loc[ind,raceType] = True

    def __FormatDistance(self):
        print('\nFormatting distance column. . .')
        for index, row in tqdm(self.data.iterrows()):
            splitString = row['Distance'].strip().split()
            miles = 0.0
            furlongs = 0.0
            yards = 0.0
            for item in splitString:
                if 'm' in item:
                    miles = float(re.sub("[^0-9]", "", item))
                if 'f' in item:                    
                    furlongs = float(re.sub("[^0-9]", "", item))
                if 'y' in item:                    
                    yards = float(re.sub("[^0-9]", "", item))
            distance = self.__ConvertDistanceToKm(miles,furlongs,yards)
            self.data.loc[index,'Distance'] = distance

    def __SplitOnRaceTrack(self):
        print('\nSplitting on race track. . .')
        uniqueRaceTracks = sorted(set(self.data['Race Track']))
        print('\tInitialising cells. . .')
        for raceTrack in tqdm(uniqueRaceTracks):
            self.data[raceTrack] = False
        print('\tPopulating cells. . .')
        for raceTrack in tqdm(uniqueRaceTracks):
            for ind, row in self.data.iterrows():
                if raceTrack == row['Race Track']:
                    self.data.loc[ind,raceTrack] = True

    def __DropUnwantedColumns(self):
        print('\nDropping unwanted columns. . .')
        self.data = self.data.drop(['Date','Surface','Prize','Age','Favourite Name','Going','Race Type','Race Track'],axis=1)

    def __CheckDatasetExists(self):
        if os.path.isfile(Constants.PREPROCESSED_DATA_FILEPATH): return True
        else: return False

    def __ConvertDistanceToKm(self,miles,furlongs,yards):
        return (miles*1.609)+(furlongs*0.201)+(yards*0.000914)
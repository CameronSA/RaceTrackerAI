import Constants
import os
import os.path
import pandas as pd
from datetime import datetime

class PreProcessData:
    def __init__(self):   
        print('Started data preprocessing. . .')
        if(self.__CheckDatasetExists()):
            print("Dataset found at '"+Constants.PREPROCESSED_DATA_FILEPATH+"'. Overwrite? (y/n)")
            if input().lower() == 'y':
                self.__PreProcessData()
        else :
            self.__PreProcessData()
        
        print('Data preprocessing complete.')
    
    def __PreProcessData(self):        
        print("Fetching dataset from path '" + str(Constants.RACE_SUMMARY_DATA_FILEPATH) + "'")
        self.data = pd.read_csv(Constants.RACE_SUMMARY_DATA_FILEPATH)        
        self.__DropUnwantedColumns()
        self.__IndexRaces()
        self.__SplitOnGoing()    
        print("Writing preprocessed data to '"+str(Constants.PREPROCESSED_DATA_FILEPATH)+"'")
        self.data.to_csv(Constants.PREPROCESSED_DATA_FILEPATH, index=None, sep=',', mode='w')

    def __IndexRaces(self):
        print('Indexing races. . .')
        self.data['Race Index'] = -1
        self.data=self.data.sort_values(['Race Track','Date'])
        index=0
        prevDate = datetime.min
        prevTrack = "123"
        for ind, row in self.data.iterrows():
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
        print('Creating going columns. . .')
        uniqueGoings = sorted(set(self.data['Going']))
        for going in uniqueGoings:
            self.data[going] = False
        for going in uniqueGoings:
            for ind, row in self.data.iterrows():
                if going == row['Going']:
                    self.data.loc[ind,going] = True
        self.data = self.data.drop(['Going'], axis=1)

    def __SplitOnRaceType(self):
        print('Splitting on race type (incomplete). . .')

    def __FormatDistance(self):
        print('Formatting distance column (incomplete). . .')

    def __DropUnwantedColumns():
        print('Dropping unwanted columns. . .')
        self.data = self.data.drop(['Date','Surface','Prize','Age'],axis=1)

    def __CheckDatasetExists(self):
        if os.path.isfile(Constants.PREPROCESSED_DATA_FILEPATH): return True
        else: return False

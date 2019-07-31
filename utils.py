
import math
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import csv


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, folder="data/", max_height=200, max_width=200, tag="classify",key=0):
        self.localFolder = ""
        self.dataFolder = folder
        self.max_height = max_height
        self.max_width = max_width
        self.tag = tag  # split audio file
        self.key = key  # split audio index key
        self.classify=[]
        self.raw_data = []  # audio raw data
        self.raw_label = []  # audio raw data
        self.ans_label = []  # audio raw data
        self.x_train = []  # cut raw data
        self.y_train = []  # label data
        self.z_train = []  # audio classify
        self.x_test = []  # cut raw data
        self.y_test = []  # label data
        self.z_test = []  # audio classify
        self.y_index = 0
        self.loadData()
        self.findCenterSize()
    def dataSize(self):
        return len(self.raw_data)

    def loadNumpy(self, name):
        npdata = np.load(name)
        return npdata
    def loadData(self):
        dataframe = pd.read_csv(self.localFolder + "list.csv")
        # print(dataframe.shape)
        x_train = []
        y_train = []
        for index, row in dataframe.iterrows():
            if(not os.path.isfile(self.localFolder +
                                  self.dataFolder + row['file_name'] + '.npy')):
                continue
            # print(npdata.shape)
            x_train.append(self.localFolder +
                           self.dataFolder + row['file_name'] + '.npy')
            y_train.append(row[self.tag])
        unique, counts = np.unique(y_train, return_counts=True)
        self.classify = dict(zip(unique, counts))
        self.raw_data = x_train
        self.raw_label = y_train
        print("total data count:", self.dataSize())
    # split data for nomalize size
    def findCenterSize(self):
        y_index = 0
        start_data = len(self.raw_data) / 5
        for data in self.raw_data:
            emdata = self.loadNumpy(data)
            if(emdata.shape[0]<self.max_height or emdata.shape[1]<self.max_width):
                continue
            used="train"
            thisLabel=self.raw_label[y_index]
            startH = int((emdata.shape[0] - self.max_height) / 2)
            startW = int((emdata.shape[1] - self.max_width) / 2)
            #emdata=emdata.reshape((self.max_height,self.max_width))#change size變形

            endata=emdata[startH:startH + self.max_height,startW:startW + self.max_width]#切割
            #print(endata.shape)
            if(y_index > start_data * self.key and y_index < start_data * (self.key + 1)):

                self.x_test.append(endata)
                self.y_test.append(thisLabel)
                self.z_test.append(y_index)
            else:
                self.x_train.append(endata)
                self.y_train.append(thisLabel)
                self.z_train.append(y_index)
                #print([data,self.raw_label[y_index],y_index])
            y_index = y_index + 1
        # print(self.x_train.shape)
        self.x_train = (np.array(self.x_train))
        self.y_train = np.array(self.y_train)
        self.z_train = np.array(self.z_train)
        self.x_test = (np.array(self.x_test))
        self.y_test = np.array(self.y_test)
        self.z_test = np.array(self.z_test)
if __name__ == '__main__':
    getDL = DataLoader(max_height=100, max_width=30)
    print(f'x_train.shape = {getDL.x_train.shape}')
    print(f'y_train.shape = {getDL.y_train.shape}')
    print(f'x_test.shape = {getDL.x_test.shape}')
    print(f'y_test.shape = {getDL.y_test.shape}')
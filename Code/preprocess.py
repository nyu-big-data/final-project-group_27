import numpy as np
import pandas as pd
import os
import dask
import contstants as const

# And pyspark.sql to get the spark session
# from pyspark.sql import SparkSession


#A class used to preprocess data
#And save out data to Data_Partitions Folder
class DataPreprocessor():
    def __init__(self, file_path) -> None:
        self.file_path = file_path                      #File Path to Read in Data
        self.working_dir = os.getcwd()                  #Get Current Working Directory
        self.csv_names = const.CSV_NAME_LIST            #Grab csv names List like: ['rationgs.csv','tags.csv'...]

    def clean_data(self):
        #Format Date Time

        #Fix Duplicates

        #
        
        pass

#Some change
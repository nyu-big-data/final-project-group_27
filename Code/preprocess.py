import numpy as np
import pandas as pd
import os
import dask
# And pyspark.sql to get the spark session
# from pyspark.sql import SparkSession


#A class used to preprocess data
#And save out data to Data_Partitions Folder
class DataPreprocessor():
    def __init__(self, file_path) -> None:
        self.file_path = file_path                      #File Path to Read in Data
        self.working_dir = os.getcwd()                  #Get Current Working Directory
        self.tags = ['movies.csv','ratings.csv',        #4 DIfferent files you need
                    'tags.csv','links.csv']

    def clean_data(self):
        #Format Date Time

        #Fix Duplicates

        #
        
        pass

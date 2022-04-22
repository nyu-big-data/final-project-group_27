"""
A python file to import constants that point to data sources
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = "hdfs:/user/" + netID
CSV_NAME_LIST = ['movies.csv','ratings.csv','tags.csv','links.csv']
SMALL = 'ml-latest-small/'
LARGE = 'ml-latest/'
DATASET_DICT = {'small': HPC_DATA_FILEPATH+SMALL,
                 'large': HPC_DATA_FILEPATH+LARGE}
SAVE_FILE_PATH = netID + "/scratch"
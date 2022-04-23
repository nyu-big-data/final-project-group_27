"""
A python file to import constants that point to data sources
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = "hdfs:/user/" + netID
MODEL_SAVE_FILE_PATH = netID + "scratch/rec_models/"
RESULTS_SAVE_FILE_PATH = netID + "scratch/big_data_final_results/results.txt"   
"""
A python file to import constants that point to file read/write paths
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = f"hdfs:/user/{netID}"
MODEL_SAVE_FILE_PATH = f"scratch/gjd9961/rec_models/"
RESULTS_SAVE_FILE_PATH = f"/scratch/gjd9961/big_data_final_results/results.txt"   
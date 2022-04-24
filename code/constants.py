"""
A python file to import constants that point to data sources
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = f"hdfs:/user/{netID}"
MODEL_SAVE_FILE_PATH = f"scratch/{netID}rec_models/"
RESULTS_SAVE_FILE_PATH = f"/scratch/{netID}big_data_final_results/results.txt"   
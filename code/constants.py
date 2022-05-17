"""
A python file to import constants that point to file read/write paths
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = f"hdfs:/user/{netID}"
RESULTS_SAVE_FILE_PATH = f"/scratch/gjd9961/big_data_final_results/results.txt"   
VAL_TEST_SCHEMA = "rating FLOAT, userId INT, movieId INT, date STRING, title STRING"
ALS_TRAIN_SCHEMA = "rating DOUBLE, userId INT, movieId INT, date STRING, title STRING, movie_mean DOUBLE, user_mean DOUBLE"
CHECKPOINT_DIR = HPC_DATA_FILEPATH + "checkpoint_dir"
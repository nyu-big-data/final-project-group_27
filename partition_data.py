#Import a bunch of things
from code.preprocess import DataPreprocessor
import code.constants as const
import sys
from pyspark.sql import SparkSession

def main(spark, dataset_size):
    """
    This program should be ran from the command line with an argument of what dataset to run on.
    The program will complete the code inside the main block below, and output train,val, and test sets into the Data_Partitions Folder.
    The code Will run off of the code in data_preprocessor
    """
    #Grab releveant file path
    filepath = const.HPC_DATA_FILEPATH
    print(f"file_path: {filepath}")
    
    #Initialize DataPreprocessor Object
    data = DataPreprocessor(spark,filepath)
               
    #Ideally Preprocess Data Something Like this:
    print(f"file_path: {filepath}")

    #THIS IS A TEST - REMOVE LATER - Call clean_data 
    clean_data = data.clean_data()

    print(f"Splitting Train/Val/Test for {filepath}")
    #THIS IS A TEST - REMOVE LATER - Call train/test/val splits
    train, val, test = data.create_train_val_test_splits(clean_data=clean_data)
    #Output Train/Test/Val Splits into Data_partitions
    print(f"Saving train/val/test splits to {const.HPC_DATA_FILEPATH}")
    train.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_train")
    val.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_small_val")
    test.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_test")
    print("Finished Saving")

    print(f"Running data.preprocess() for {filepath}")
    #THIS IS A TEST - REMOVE LATER
    data.preprocess()        

    #Let us know when its done
    print(f"Done for dataset {filepath}")

# Only enter this block if we're in main
if __name__ == "__main__":
    #Initalize spark session
    spark = SparkSession.builder.appName('Spark_Session_Name').getOrCreate()
    dataset_size = sys.argv[1]
    main(spark, dataset_size) #Either 'small' or 'large' should be passed through
   
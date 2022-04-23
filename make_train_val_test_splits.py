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
    
    #Initialize DataPreprocessor Object
    data = DataPreprocessor(spark,filepath)
               
    print(f"Splitting Train/Val/Test for {filepath}")
    #Call train/test/val splits - Returns train, val, test data splits
    train, val, test = data.preprocess()

    #Output Train/Test/Val Splits into Data_partitions
    print(f"Saving train/val/test splits to {const.HPC_DATA_FILEPATH}")
    train.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_train")
    val.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_small_val")
    test.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}_test")

    #Let us know when done
    print("Finished Saving")

# Only enter this block if we're in main
if __name__ == "__main__":
    #Initalize spark session
    spark = SparkSession.builder.appName('Make_Train_Val_Test_Splits_Session').getOrCreate()
    #Either 'small' or 'large' should be passed through -> sys.argv[1]
    dataset_size = sys.argv[1]
    main(spark, dataset_size) 
   
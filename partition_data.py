#Import a bunch of things
from code.preprocess import DataPreprocessor
import code.constants as const
import sys
from pyspark.sql import SparkSession


def main(spark, filepath_arg):
    """
    This program should be ran from the command line with an argument of what dataset to run on.
    The program will complete the code inside the main block below, and output train,val, and test sets into the Data_Partitions Folder.
    The code Will run off of the code in data_preprocessor
    """
    #Grab releveant file path
    filepath = const.DATASET_DICT[filepath_arg]
    print(f"filepath_arg: {filepath_arg} file_path: {filepath}")
    
    data = DataPreprocessor(spark,filepath)
    data.delete_dupe_ids()
    
    #Ideally Preprocess Data Something Like this:
    print(f"Cleaning the {filepath_arg} dataset")
    #data.clean_data()
    #Output Train/Test/Val Splits into Data_partitions
    print(f"Splitting Train/Val/Test for {filepath_arg}")

    #Let us know when its done
    print(f"Done for dataset {filepath}")

# Only enter this block if we're in main
if __name__ == "__main__":

    #Initalize spark session
    spark = SparkSession.builder.appName('Spark_Session_Name').getOrCreate()
    main(spark, sys.argv[1]) #Either 'small' or 'large' should be passed through
   
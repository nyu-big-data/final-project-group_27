#Import a bunch of things
from code.preprocess import DataPreprocessor
import code.constants as const
import sys
from pyspark.sql import SparkSession

def main(spark, dataset_size):
    """
    This program should be ran from the command line with an argument of what dataset to run on.
    dataset_size arg is either 'small' or 'large. The files you have stored in hdfs must be
    named like "small-ratings.csv" or "large-movies.csv"
    The program will complete the code inside the main block below, and output train,val, and test sets into the Data_Partitions Folder.
    The code Will run off of the code in preprocess
    """
    #Grab releveant file path files need to be saved like small-movies.csv
    filepath = const.HPC_DATA_FILEPATH + dataset_size + "-"
    
    #Initialize DataPreprocessor Object
    data = DataPreprocessor(spark,filepath)
               
    print(f"Splitting Train/Val/Test for {filepath}")
    #Call train/test/val splits - Returns train, val, test data splits
    train, val, test = data.preprocess(sanity_checker=True)

    #Output Train/Test/Val Splits into Data_partitions
    print(f"Saving train/val/test splits to {const.HPC_DATA_FILEPATH}")
    train.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}-train.csv")
    val.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}-val.csv")
    test.write.csv(f"{const.HPC_DATA_FILEPATH}{dataset_size}-test.csv")

    #test.write.format("csv").mode("overwrite").options(header="true",sep="'").save(path=self.output_file_path)

    #Let us know when done
    print("Finished Saving")

# Only enter this block if we're in main
if __name__ == "__main__":
    #Initalize spark session
    spark = SparkSession.builder.appName('Proj').getOrCreate()
    #Either 'small' or 'large' should be passed through -> sys.argv[1]
    dataset_size = sys.argv[1]
    if dataset_size not in ['small','large']:
        raise Exception("Terminating, you must enter 'small' or 'large'")
    else:
        main(spark, dataset_size) 
   
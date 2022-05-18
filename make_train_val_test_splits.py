# Import a bunch of things
from code.preprocess import DataPreprocessor
import code.constants as const
import sys
from pyspark.sql import SparkSession


def main(spark, dataset_size, model_type):
    """
    This program should be ran from the command line with an argument of what dataset to run on.
    dataset_size arg is either 'small' or 'large. The files you have stored in hdfs must be
    named like "small-ratings.csv" or "large-movies.csv"
    The program will complete the code inside the main block below, and output train,val, and test sets into the Data_Partitions Folder.
    The code Will run off of the code in preprocess
    """
    # Grab releveant file path files need to be saved like small-movies.csv
    filepath = const.HPC_DATA_FILEPATH + dataset_size + "-"

    # Initialize DataPreprocessor Object
    data = DataPreprocessor(
        spark=spark, file_path=filepath, model_type=model_type)

    print(f"Splitting Train/Val/Test for {filepath}")
    # Call train/test/val splits - Returns train, val, test data splits
    train, val, test = data.preprocess(sanity_checker=True)

    # Add ALS Rating Normalization Preprocess if necessary
    if model_type == 'als':
        train = data.als_normalize_ratings(train)

    # Output Train/Test/Val Splits into Data_partitions
    print(f"Saving train/val/test splits to {const.HPC_DATA_FILEPATH}")
    train.write.parquet(
        f"{const.HPC_DATA_FILEPATH}{model_type}-{dataset_size}-train.parquet", mode="overwrite")
    val.write.parquet(
        f"{const.HPC_DATA_FILEPATH}{dataset_size}-val.parquet", mode="overwrite")
    test.write.parquet(
        f"{const.HPC_DATA_FILEPATH}{dataset_size}-test.parquet", mode="overwrite")

    # Let us know when done
    print("Finished Saving")


# Only enter this block if we're in main
if __name__ == "__main__":
    # Initalize spark session
    spark = SparkSession.builder.appName('Proj').getOrCreate()
    # Either 'small' or 'large' should be passed through -> sys.argv[1]
    dataset_size = sys.argv[1]
    model_type = sys.argv[2]
    if dataset_size not in ['small', 'large']:
        raise Exception(
            f"Terminating, you must enter 'small' or 'large', you entered {dataset_size}")
    if model_type not in ['als', 'baseline']:
        raise Exception(
            f"Terminating, you must enter 'als' or 'baseline' you entered {model_type}")

    main(spark, dataset_size, model_type)

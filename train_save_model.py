from pyspark.sql import SparkSession
from code.model import Model
import code.constants as const
import sys
import json
from pyspark.ml.recommendation import ALS

# Main Function that will define model behavior


def main(spark, model_size, model_type, rank,maxIter,regParam):
    
    print(f"Filepath: {const.HPC_DATA_FILEPATH}{model_size}")
    # Grab the filepaths for model_size
    train_file_path = f"{const.HPC_DATA_FILEPATH}{model_type}-{model_size}-train.csv" #Depends on what model we want - ALS includes more preprocessing
    test_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-test.csv"
    val_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-val.csv"
    

    train = spark.read.csv(train_file_path,
                            schema=const.ALS_TRAIN_SCHEMA)
    test = spark.read.csv(test_file_path,
                        schema=const.VAL_TEST_SCHEMA)
    val = spark.read.csv(val_file_path,
                        schema=const.VAL_TEST_SCHEMA)


    als = ALS(maxIter=maxIter, rank=rank, regParam=regParam,
                  nonnegative=False, seed=10, userCol="userId",
                  itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

    als = als.fit(train)
    als.save(const.MODEL_SAVE_FILE_PATH+f"{model_size}-{rank}-{maxIter}-{regParam}")

# Enter this block if we're in __main__
if __name__ == '__main__':
    """
    Model argumensts hsould be entered in the following order:
    1) model size -> either "small" or "large"
    2) model type -> either "baseline" or "als" (More to Do)
    3) Dictionary of {parameter:argument} pairs that will be parsed to the model
    i.e. '{"rank":10, "maxIter":10,"regParam":0.05}'
    """
    # Initialize spark context
    spark = SparkSession.builder.appName('Proj').getOrCreate()
    # Model size is either "small" or "large"
    model_size = sys.argv[1]
    # Define the model type in second argument:
    model_type = sys.argv[2]
    # Model Args:
    rank = sys.argv[3]
    maxIter = sys.argv[3]
    regParam = sys.argv[3]

    # Make sure input is valid
    if model_size not in ['small', 'large']:
        raise Exception(
            f"Model Size must either be 'small' or 'large', you entered {model_size}")
    if model_type not in ['als', 'baseline']:
        raise Exception(
            f"Model Type must be either 'als' or 'baseline', you entered {model_type}")
    #Call Main
    main(spark, model_size, model_type, rank,maxIter,regParam)

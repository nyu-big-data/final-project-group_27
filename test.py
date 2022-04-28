from pyspark.sql import SparkSession
from code.model import Model
import code.constants as const
import sys
import json
from code.preprocess import DataPreprocessor
# Main Function that will define model behavior


def main(spark, model_size):

    # # Grab the filepaths for model_size
    # train_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-train.csv"
    # test_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-test.csv"
    # val_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-val.csv"

    train, val, test = DataPreprocessor(spark,const.HPC_DATA_FILEPATH+"small-").preprocess(sanity_checker=True)
    # Read data for file paths
    # train = spark.read.csv(train_file_path, header=True,
    #                        schema=const.TRAIN_VAL_TEST_SCHEMA)
    # test = spark.read.csv(test_file_path, header=True,
    #                       schema=const.TRAIN_VAL_TEST_SCHEMA)
    # val = spark.read.csv(val_file_path, header=True,
    #                      schema=const.TRAIN_VAL_TEST_SCHEMA)

    m = Model(model_type='baseline',min_ratings=0)
    df = m.run_model(train,val)
    df.write.csv("Debugging_test.csv")
    df = df.toPandas()
    df.to_csv(f"{const.MODEL_SAVE_FILE_PATH}debugging_df.csv")
    print(vars(m))
    

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

    main(spark, model_size)

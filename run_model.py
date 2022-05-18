from pyspark.sql import SparkSession
from pyspark import SparkContext
from code.model import Model
import code.constants as const
import sys
import json

# Main Function that will define model behavior


def main(spark, model_size, model_type, model_args):
    
    print(f"Filepath: {const.HPC_DATA_FILEPATH}{model_size}")
    # Grab the filepaths for model_size
    train_file_path = f"{const.HPC_DATA_FILEPATH}{model_type}-{model_size}-train.csv" #Depends on what model we want - ALS includes more preprocessing
    test_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-test.csv"
    val_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-val.csv"
    
    #Try setting checkpoint dir
    SparkContext.setCheckpointDir(dirName=const.CHECKPOINT_DIR)

    if model_type == 'als':
        train = spark.read.csv(train_file_path,
                            schema=const.ALS_TRAIN_SCHEMA)
    else:
        train = spark.read.csv(train_file_path,
                          schema=const.VAL_TEST_SCHEMA )
    test = spark.read.csv(test_file_path,
                        schema=const.VAL_TEST_SCHEMA)
    val = spark.read.csv(val_file_path,
                        schema=const.VAL_TEST_SCHEMA)


    # Pass through dictionary of keyword arguments to Model()
    reccomender_system = Model(model_size=model_size, model_type=model_type, **model_args)
    # Run the model
    reccomender_system.run_model(train=train, val=None, test=test)

    #Grab the key:value pairs of instance variables
    instance_vars = vars(reccomender_system)
    #Get rid of "methods" nested dict - it has the function calls which can't be written to output file
    del instance_vars["methods"]

    # Write our results and model parameters
    print(f"Recording the model_params to: {const.RESULTS_SAVE_FILE_PATH}")
    with open(const.RESULTS_SAVE_FILE_PATH, 'a') as output_file:
        output_file.write(json.dumps(instance_vars))
        output_file.write("\n")

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
    model_args = json.loads(sys.argv[3])

    # Make sure input is valid
    if model_size not in ['small', 'large']:
        raise Exception(
            f"Model Size must either be 'small' or 'large', you entered {model_size}")
    if model_type not in ['als', 'baseline']:
        raise Exception(
            f"Model Type must be either 'als' or 'baseline', you entered {model_type}")
    #Call Main
    main(spark, model_size, model_type, model_args)

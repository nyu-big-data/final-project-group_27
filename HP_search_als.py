from pyspark.sql import SparkSession
from code.model import Model
import code.constants as const
import sys
import json
import numpy as np


# Main Function that will define model behavior


def main(spark, model_size, ranks, regParam, maxIter):

    # Grab the filepaths for model_size
    train_file_path = f"{const.HPC_DATA_FILEPATH}als-{model_size}-train.csv"
    val_file_path = f"{const.HPC_DATA_FILEPATH}{model_size}-val.csv"

    # Read data for file paths
    train = spark.read.csv(train_file_path, header=True,
                           schema=const.ALS_TRAIN_SCHEMA)
    val = spark.read.csv(val_file_path, header=True,
                         schema=const.VAL_TEST_SCHEMA)

    
    #Iterate over K values - Calculate Baseline Model and Search for best K on val performance
    # regParams = np.logspace(-3,-1,5)
    
    for rank in ranks:
    
        print(f"Running Model: Max Iter:{maxIter}, Rank: {rank}, Reg Param: {regParam}")
        # Pass through dictionary of keyword arguments to Model()
        reccomender_system = Model(model_size=model_size, model_type='als', rank=int(rank), maxIter = maxIter,regParam = regParam)
        # Run the model
        reccomender_system.run_model(train, val)

        #Grab the key:value pairs of instance variables
        instance_vars = vars(reccomender_system)
        print(f'instance vars: {instance_vars}')
        #Get rid of "methods" nested dict - it has the function calls which can't be written to output file
        del instance_vars["methods"]
        del reccomender_system
        
        print("Writing results")
        # Write our results and model parameters
        print("Recording the model_params")
        with open(const.RESULTS_SAVE_FILE_PATH, 'a') as output_file:
            output_file.write(json.dumps(instance_vars))
            output_file.write("\n")

        #Clear anything thats cached
        spark.catalog.clearCache()

# Enter this block if we're in __main__
if __name__ == '__main__':
    """
    Script used to tune hyper parameters for some model
    """
    # Initialize spark context
    spark = SparkSession.builder.appName('proj').getOrCreate()
    # Model size is either "small" or "large"
    model_size = sys.argv[1]
    ranks_start = int(sys.argv[2])
    ranks_stop = int(sys.argv[3])
    step = int(sys.argv[4])
    exp = int(sys.argv[5])
    maxIter = int(sys.argv[6])
    regParam = 10**(-exp)
    ranks = range(ranks_start, ranks_stop+step, step)

    # Make sure input is valid
    if model_size not in ['small', 'large']:
        raise Exception(f"Model Size must either be 'small' or 'large', you entered {model_size}")
    #Call Main
    main(spark, model_size, ranks, regParam, maxIter)

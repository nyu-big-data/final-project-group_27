from pyspark.sql import SparkSession
from code.model import Model
import code.constants as const
import sys
import json

#Main Function that will define model behavior
def main(spark, model_size,model_type,model_args):

    #Grab the filepaths for model_size
    print(f"Getting train / test / val splits for {model_size}")
    train_file_path = const.HPC_DATA_FILEPATH + model_size + "train.csv"
    test_file_path = const.HPC_DATA_FILEPATH + model_size + "test.csv"
    val_file_path = const.HPC_DATA_FILEPATH + model_size + "val.csv"
    
    #Read data for file paths
    train = spark.read.csv(train_file_path)
    test = spark.read.csv(test_file_path) #Ignore this during validation tuning
    val = spark.read.csv(val_file_path)
    print("Data Read Successfully")

    print(f"Args being passed to model: {model_args.items()}")
    #Pass through dictionary of keyword arguments to Model()
    reccomender_system = Model(**model_args)
    
    #Run model according to what model_type parameter was passed
    #If model is Alternating Least Squares use ALS_fit_and_run
    if model_type == 'als':
        #IMPORTANT: I HAVE THIS CURRENTLY ONLY PREICTING AND MEASURING ERROR ON VAL - WHEN WE'RE DONE TUNING AND EVALUATING
        #RESULTS WE'LL HAVE TO GO INTO Als_fit_and_run AND CHANGE THE CODE!
        userRecs, movieRecs = reccomender_system.ALS_fit_and_run(training=train,val=val)

    #If baseline... you guessed it, use baseline
    elif model_type == 'baseline':
        userRecs, movieRecs = reccomender_system.baseline(train,val,test)


if __name__ == '__main__':
    """
    Model argumensts hsould be entered in the following order:
    1) model size -> either "small" or "large"
    2) model type -> either "baseline" or "als" (More to Do)
    3) Dictionary of {parameter:argument} pairs that will be parsed to the model
    i.e. '{"rank":10, "maxIter"=10,"regParam":0.05}'
    """
    #Initialize spark context
    spark = SparkSession.builder.appName('Run_Model').getOrCreate()
    #Model size is either "small" or "large"
    model_size = sys.argv[1]+"_"
    #Define the model type in second argument:
    model_type = sys.argv[2]
    #Model Args:
    model_args = json.loads(sys.argv[3])
    main(spark, model_size,model_type,model_args)
#Import a bunch of things
from code.preprocess import DataPreprocessor
import code.constants as const
import sys



def main(filepath_arg):
    """
    This program should be ran from the command line with an argument of what dataset to run on.
    The program will complete the code inside the main block below, and output train,val, and test sets into the Data_Partitions Folder.
    The code Will run off of the code in data_preprocessor
    """
    #Grab releveant file path
    filepath = const.DATASET_DICT[filepath_arg]
    print(f"filepath_arg: {filepath_arg} file_path: {filepath}")
    #data = DataProcessor(filepath_arg)

    #Ideally Preprocess Data Something Like this:
    print(f"Cleaning the {filepath_arg} dataset")
    #data.clean_data()
    #Output Train/Test/Val Splits into Data_partitions
    print(f"Splitting Train/Val/Test for {filepath_arg}")

    #Let us know when its done
    print(f"Done for dataset {filepath}")

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    #spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    #netID = getpass.getuser()

    # Call our main routine
    main(sys.argv[1]) #Either 'small' or 'large' should be passed through
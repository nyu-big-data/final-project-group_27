# CodeCoding Folder

This folder should be used to store various libraries of code we develop for this project. We should use an object oriented approach to keep our code tidy, efficient, and debbugable.

## Contstants and how to Find Files / Run Scripts Properly

The constants.py python file should be imported into the other scripts we run and holds file paths for datasets and other constant variables.
> For Example: `RESULTS_SAVE_FILE_PATH="/gjd9961/scratch/big_data_final_results/results.txt"`

For the script to be able to find the correct files in hdfs, you must save your files in hdfs with the naming convention: `"small-movies.csv"` or `"large-ratings.csv"`

When we run preprocess, train test and val splits will be saved to your hdfs with the following naming convention: `"small_val"` or `"large_train"`

## Preprocess

This python script will hold a class, `DataPreprocessor` that will have the responsabilities to preprocess our data as necessary.

## HFS

- Add files to hadoop cluster

> hfs -put

- List files on Hadoop Cluster

> hfs -ls

- Remove Files on Hadoop Cluster

> hfs -rm -r `file_name`

## Yarn Logs

- Access Yarn Logs

> yarn logs -applicationId application_1648648882306_28175 -log_files stdout

## Spark

- Submit Job

> spark-submit some_script.py `args`

- Long way to Submit Job (Recommended)

> spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true 

To execute run_model.py with ALS model: enter the following:

> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true run_model.py small als '{"rank":5, "maxIter":5,"regParam":0.05}'
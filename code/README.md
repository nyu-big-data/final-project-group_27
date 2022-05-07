# Coding README

## Getting Started

Firstly, make sure to run `source setup.sh` in the HPC terminal. Secondly, make sure you have the datasets saved in your `hdfs` filesystem. They must follow this naming convention: `<dataset_size>-<file_name>.csv`. For example: `large-movies.csv`. If they're not named this way, the scripts won't be able to find them. Similarly, if you need to access the Validation/Test splits in your `hdfs` filesystem, they are named with the following convention: `<dataset_size>-<name_of_set>` For example, `small-train.csv`. Training is named differently depending on model type, for example `"als-small-train.csv"`.

## How to execute make_train_val_test_splits.py

> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true make_train_val_test_splits.py `dataset_size` `model_type`

For example:

> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true make_train_val_test_splits.py small

## How to execute run_model.py in the cluster

> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true run_model.py `dataset_size` `model_type` `param_dict`

For Example:

> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true run_model.py small als '{"rank":5, "maxIter":5,"regParam":0.05}'
> spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true run_model.py large baseline '{"bias":100}'

Make sure the param dict is in single quotes.

## Contstants and how to Find Files / Run Scripts Properly

The constants.py python file should be imported into the other scripts we run and holds file paths for datasets and other constant variables.

> For Example: `RESULTS_SAVE_FILE_PATH="/gjd9961/scratch/big_data_final_results/results.txt"`

For the script to be able to find the correct files in hdfs, you must save your files in hdfs with the naming convention: `"small-movies.csv"` or `"large-ratings.csv"`

When we run preprocess, train test and val splits will be saved to your hdfs with the following naming convention: `"small_val"` or `"large_train"`

For the training splits, the model type is added to the naming convention: `"als-small-train.csv"`

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

> `spark-submit --conf spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true run_model.py small als '{"rank":5, "maxIter":5,"regParam":0.05}'`

`run_model.py small als '{"rank":5,"maxIter":10,"regParam":0.1}'`

`run_model.py small baseline '{"min_ratings":10}'` or `run_model.py small baseline '{"bias":10}'`

Optional parameter -> `positive_rating_threshold: 'int'` used in custom ranking metrics

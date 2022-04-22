# CodeCoding Folder

This folder should be used to store various libraries of code we develop for this project. We should use an object oriented approach to keep our code tidy, efficient, and debbugable.

## Contstants

This python folder should be imported into the other scripts we run and holds file paths for datasets and other constant variables.
> For Example: `HPC_SMALL_FILEPATH = "/scratch/work/courses/DSGA1004-2021/movielens"`

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

- Long way to Submit Job

> spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true `python_file` `args`
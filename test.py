import pyspark

from pyspark.sql import SparkSession

spark = spark = SparkSession.builder.appName('Spark_Session_Name').getOrCreate()

file_path = "/scratch/work/courses/DSGA1004-2021/movielens/ml-latest-small/"
spark.read.csv(file_path + 'movies.csv', schema='movieId INT, title STRING, genres STRING')
print("Sucess")
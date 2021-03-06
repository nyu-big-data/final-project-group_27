import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import IntegerType

#A class used to preprocess data and to return train/val/test splits
class DataPreprocessor():
    def __init__(self, spark, file_path) -> None:
        self.spark = spark                              #Spark Driver
        self.file_path = file_path                      #File Path to Read in Data


    #Main Method - Call this in partition_data.py to get train/val/test splits returned
    def preprocess(self):
        """
        Goal: Save train/val/test splits to netID/scratch - all using self methods
        Step 1: self.clean_data: clean the data, format timestamp to date, and remove duplicate movie titles
        Step 2: self.create_train_val_test_splits: reformats data, drops nans, and returns train,val and test splits
        """
        #Format Date Time and Deduplicate Data
        clean_data = self.clean_data()                                                  #No args need to be passed, returns RDD of joined data (movies,ratings), without duplicates
        #Get Utility Matrix
        train, val, test = self.create_train_val_test_splits(clean_data)                #Needs clean_data to run, returns train/val/test splits
        #Return top 100 recs for movies / users
        return train, val, test
    
    #preprocess calls this function
    def clean_data(self):
        """
        goal: for movie titles with multiple movieIDs, in the movies dataset,
        remove the duplicate IDs with the least ratings for each movie. 
        Additionally, remove those IDs from the ratings dataset, so we get a 1:1 mapping
        between movie title and movie ID
        inputs: None, however - self.file_path -> this should link to your hfs/netid/
        outputs: all_data - a RDD of joined data (movies,reviews) - deduplicated of titles that appear more than once
                this loses only 6 records (reviews from users) for small
        """

        #Import the movies data + add to schema so it can be used by SQL + header=True because there's a header
        movies = self.spark.read.csv(self.file_path + 'movies.csv', header=True, \
                                    schema='movieId INT, title STRING, genres STRING')
    
        #Same for ratings - TIMESTAMP MUST BE STRING
        ratings = self.spark.read.csv(self.file_path + 'ratings.csv', header=True, \
                    schema='userId INT, movieId INT, rating FLOAT, timestamp STRING') 
        
        #Get the MM-dd-yyyy format for timestamp values producing new column, Date
        ratings = ratings.withColumn("date",from_unixtime(col("timestamp"),"MM-dd-yyyy"))
        ratings = ratings.drop("timestamp") #Drop timestamp, we now have date

        #Join Dfs - Join Movies with Ratings on movieId, LEFT JOIN used, select only rating, userId, movieId, title and date
        joined = ratings.join(movies, ratings.movieId==movies.movieId, how='left').select(\
                            ratings.rating,ratings.userId,\
                            ratings.movieId,ratings.date,movies.title)

        #Find Movie Titles that map to multiple IDs
        dupes = joined.groupby("title").agg(countDistinct("movieId").alias("countD")).filter(col("countD")>1)

        #Isolate non-dupes into a df
        non_dupes = joined.join(dupes, joined.title==dupes.title, how='leftanti')
    
        #Get all of the dupes data - ratings, userId, ect - again from Joined
        dupes = dupes.join(joined, joined.title==dupes.title, how='inner').select(\
                                        joined.movieId,joined.rating,\
                                        joined.date,dupes.title,joined.userId)
    
        #Clean the dupes accordingly
        #Step 1: Aggregate by title/movie Id, then count userId - give alias
        #Step 2: Create a window to partition by - we iterate over titles ranking by 
        #countD (count distinct of userId) - movieId forces a deterministic ranking based off movieId
        #Step 3: Filter max_dupes so we only grab top ranking movieIds
        windowSpec = Window.partitionBy("title").orderBy("countD","movieId")
        max_dupes = dupes.groupBy(["title","movieId"]).agg(countDistinct("userId").alias("countD"))
        max_dupes = max_dupes.withColumn("dense_rank",dense_rank().over(windowSpec))
        max_dupes = max_dupes.filter(max_dupes.dense_rank=="2")
        max_dupes = max_dupes.drop("countD","dense_rank")
        
        #Get a list of movie ids ~len(5) for small - which are the ones we want to keep
        ids = list(max_dupes.toPandas()['movieId'])
        cleaned_dupes = dupes.where(dupes.movieId.isin(ids))
        
        #Get the union of the non_dupes and cleaned_dupes
        clean_data = non_dupes.union(cleaned_dupes)

        #Subtract 2.5 from each review to create negative reviews
        clean_data = clean_data.withColumn("rating",col("rating")-2.5)
        
        #For testing purposes should be 100,830 for small dataset
        #print(f"The length of the combined and de-deduped joined data-set is: {len(clean_data.collect())}")

        #Return clean_data -> Type: Spark RDD Ready for more computation
        return clean_data

    #TO DO?? Should we enforce min_review cutoff to make sure no cold-start for any prediction?
    def enforce_min_review(self):
        pass
    #Create Train Test Val Splits - .preprocess() calls this function
    def create_train_val_test_splits(self, clean_data):
        """
        input: RDD created by joining ratings.csv and movies.csv - cleaned of duplicates and formatted accordingly
        output: training 60%, val 20%, test 20% splits with colums cast to integer type and na's dropped
        """
        #Type Cast the cols to numeric
        ratings = clean_data.withColumn('movieId',col('movieId').cast(IntegerType())).withColumn("userId",col("userId").cast(IntegerType()))
        #Drop nulls
        ratings = ratings.na.drop("any")
        #Create training, val, test splits
        (training, val, test) = ratings.randomSplit([0.6, 0.2, 0.2])
        # u = u.pivot(index='userId', columns = 'title', values ='rating')
        return training, val, test

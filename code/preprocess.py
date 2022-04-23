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
    def preprocess(self, sanity_checker=False):
        """
        Goal: Save train/val/test splits to netID/scratch - all using self methods
        Step 1: self.clean_data: clean the data, format timestamp to date, and remove duplicate movie titles
        Step 2: self.create_train_val_test_splits: reformats data, drops nans, and returns train,val and test splits
        input:
        -----
        sanity_checker: boolean - Flag that decides if we call self.sanity_check()
        -----
        output: 
        train: RDD of Training Set Data
        val: RDD of Validation Set Data
        test: RDD of Validation Set Data
        """
        #Format Date Time and Deduplicate Data
        clean_data = self.clean_data()                                                  #No args need to be passed, returns RDD of joined data (movies,ratings), without duplicates
        #Get Utility Matrix
        train, val, test = self.create_train_val_test_splits(clean_data)                #Needs clean_data to run, returns train/val/test splits
        
        #Check if we should perform sanity check
        if sanity_checker:
            flag = self.sanity_check(train,val,test)
            #If flag == True we're good
            if flag:
                print("The val and test splits are disjoint!")
            #Otherwise raise exception
            else:
                raise Exception("The Validation and Test sets are not disjoint!")

        #Return train val test sets
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
        
        #Reorder Columns so union works
        cleaned_dupes = cleaned_dupes.select('rating', 'userId', 'movieId', 'date', 'title')

        
        #Get the union of the non_dupes and cleaned_dupes
        clean_data = non_dupes.union(cleaned_dupes)

        #Subtract 2.5 from each review to create negative reviews
        clean_data = clean_data.withColumn("rating",col("rating")-2.5)
        
        #For testing purposes should be 100,830 for small dataset
        #print(f"The length of the combined and de-deduped joined data-set is: {len(clean_data.collect())}")

        #Return clean_data -> Type: Spark RDD Ready for more computation
        return clean_data

    #Create Train Test Val Splits - .preprocess() calls this function
    def create_train_val_test_splits(self, clean_data):
        """
        Procedure: 
        Create two columns - the first will measure the specific row count for a specific user
        the other will be static fixed at the total number of reviews for that user. The row count
        is sorted by date ascending, so the first row is the oldest review.
        
        Then, subset training to be where row_count <= .6 *length, grabbing the oldest 60% of reviews, for
        all users.
        
        We then subset the remaining data into a hold out, with the goal of creating two disjoint validation
        and test data sets when looking at userId (meaning they should not have any shared userId values), 
        but still have roughly the same amount of data, or whatever percentage we want to achieve
        
        To obtain approximate equality and disjoint userId membership, for the remiaining data
        sort userId by user_review_count descending, then alternate values in that list, assigning
        half to test and half to validation.
        -----
        input: RDD created by joining ratings.csv and movies.csv - cleaned of duplicates and formatted accordingly
        -----
        -----
        output: training 60%, val 20%, test 20% splits with colums cast to integer type and na's dropped
        -----
        """
        #Type Cast the cols to numeric
        ratings = clean_data.withColumn('movieId',col('movieId').cast(IntegerType())).withColumn("userId",col("userId").cast(IntegerType()))
        #Drop nulls
        ratings = ratings.na.drop("any")
    
        #strategy, partition by userId, and userId order by date, 
        #take the first 60% of reviews for all users
        w1 = Window.partitionBy("userId")
        w2 = Window.partitionBy("userId").orderBy("date")
        ratings = (ratings.withColumn("row_num", row_number().over(w2))
                       .withColumn('length', count('userId').over(w1))
                  )

        #store in training RDD by 
        #selecting all rows where the row_count for that user <= 60% total reviews for that user
        
        training = ratings.filter("row_num <=.6*length")
        #now for validation and test set, we want those to have no users in common, but for them to
        #be approximately equal size. 
        holdout_df = ratings.filter("row_num >.6*length")
        
        #strategy, of the data not in my train set, group users by number of movies they have seen
        #sort descending
        holdout_split = holdout_df.groupBy("userId").count().orderBy("count", ascending=False).toPandas()
        
        #store the list of userIds sorted by descending total movie count
        holdout_split = list(holdout_split.userId)
        
        #partition list of userIds by taking every other index and putting it in the validation set
        val_users = holdout_split[::2]
        
        #create a validation and test set by filtering holdout data based on whether movieId isin val_users
        val = holdout_df.filter(holdout_df.userId.isin(val_users))
        test = holdout_df.filter(~holdout_df.userId.isin(val_users))

        #Return train/val/test splits
        return training, val, test

    #TO DO?? Should we enforce min_review cutoff to make sure no cold-start for any prediction?
    def enforce_min_review(self):
        pass

    #Check to train/val/test splits to make sure approx 60/20/20 split is achieved
    def sanity_check(self,train,val,test):
        """
        Method to print out the shape of train/val/test splits, and a check to make sure that
        val and test splits are disjoint (no distinct userId appears in both)
        input:
        -----
        train: RDD - Training data split created from .create_train_val_test_splits
        val: RDD - Validation data split created from .create_train_val_test_splits
        test: RDD - Testing data split created from .create_train_val_test_splits
        -----
        output:
        -----
        returnFlag: boolean - True means test and val splits are disjoint on userId
        """

        #Get observatio counts for training, val, and test sets
        training_obs = train.count()
        val_obs = val.count()
        test_obs = test.count()

        #Print them out
        print(f"Training Data Len: {training_obs} Val Len: {val_obs}, Test Len: {test_obs}")

        #Check if there are any overlapping_ids in the sets
        overllaping_ids = val.join(test, test.userId==val.userId,how='inner').count()
        
        #Return True if they're disjoint, False if there's overlap
        return overllaping_ids == 0
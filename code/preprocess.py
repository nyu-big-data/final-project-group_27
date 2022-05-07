import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from code.unit_tests import UnitTest

# A class used to preprocess data and to return train/val/test splits


class DataPreprocessor():
    def __init__(self, spark, file_path, model_type) -> None:
        self.spark = spark  # Spark Driver
        self.file_path = file_path  # File Path to Read in Data
        self.model_type = model_type

    # Main Method - Call this in partition_data.py to get train/val/test splits returned

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
        # Format Date Time and Deduplicate Data
        # No args need to be passed, returns RDD of joined data (movies,ratings), without duplicates
        clean_data = self.clean_data()
        # Get Utility Matrix
        # Needs clean_data to run, returns train/val/test splits
        train, val, test = self.create_splits(clean_data)

        # Check if we should perform sanity check
        if sanity_checker:
            tester = UnitTest()
            flag = tester.sanity_check(train, val, test)
            # If flag == True we're good
            if flag:
                print("All Three Data Sets are Disjoint!")
            # Otherwise raise exception
            else:
                raise Exception(
                    "One Pair of Train/Val/Test Sets are not disjoint, check stdout!")

        # Return train val test sets
        return train, val, test

    # preprocess calls this function
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

        # Import the movies data + add to schema so it can be used by SQL + header=True because there's a header
        movies = self.spark.read.csv(self.file_path + 'movies.csv', header=True,
                                     schema='movieId INT, title STRING, genres STRING')

        # Same for ratings - TIMESTAMP MUST BE STRING
        ratings = self.spark.read.csv(self.file_path + 'ratings.csv', header=True,
                                      schema='userId INT, movieId INT, rating FLOAT, timestamp STRING')

        # Get the MM-dd-yyyy format for timestamp values producing new column, Date
        ratings = ratings.withColumn(
            "date", from_unixtime(col("timestamp"), "MM-dd-yyyy"))
        ratings = ratings.drop("timestamp")  # Drop timestamp, we now have date

        # Join Dfs - Join Movies with Ratings on movieId, LEFT JOIN used, select only rating, userId, movieId, title and date
        joined = ratings.join(movies, ratings.movieId == movies.movieId, how='left').select(
            ratings.rating, ratings.userId,
            ratings.movieId, ratings.date, movies.title)

        # Find Movie Titles that map to multiple IDs
        dupes = joined.groupby("title").agg(countDistinct(
            "movieId").alias("countD")).filter(col("countD") > 1)

        # Isolate non-dupes into a df
        non_dupes = joined.join(dupes, joined.title ==
                                dupes.title, how='leftanti')

        # Get all of the dupes data - ratings, userId, ect - again from Joined
        dupes = dupes.join(joined, joined.title == dupes.title, how='inner').select(
            joined.movieId, joined.rating,
            joined.date, dupes.title, joined.userId)

        # Clean the dupes accordingly
        # Step 1: Aggregate by title/movie Id, then count userId - give alias
        # Step 2: Create a window to partition by - we iterate over titles ranking by
        # countD (count distinct of userId) - movieId forces a deterministic ranking based off movieId
        # Step 3: Filter max_dupes so we only grab top ranking movieIds
        windowSpec = Window.partitionBy("title").orderBy("countD", "movieId")
        max_dupes = dupes.groupBy(["title", "movieId"]).agg(countDistinct(
            "userId").alias("countD")).orderBy("countD", ascending=False)
        max_dupes = max_dupes.withColumn(
            "dense_rank", dense_rank().over(windowSpec))
        max_dupes = max_dupes.filter(max_dupes.dense_rank == "1")
        max_dupes = max_dupes.drop("countD", "dense_rank")

        # Get a list of movie ids ~len(5) for small - which are the ones we want to keep
        ids = list(max_dupes.toPandas()['movieId'])
        cleaned_dupes = dupes.where(dupes.movieId.isin(ids))

        # Reorder Columns so union works
        cleaned_dupes = cleaned_dupes.select(
            'rating', 'userId', 'movieId', 'date', 'title')

        # Get the union of the non_dupes and cleaned_dupes
        clean_data = non_dupes.union(cleaned_dupes)

        # Repartition for efficiency:
        clean_data = clean_data.repartition(30)

        # Return clean_data -> Type: Spark RDD Ready for more computation
        return clean_data

    # Create Train Test Val Splits - .preprocess() calls this function
    def create_splits(self, clean_data):
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
        # Drop Any Nan rows
        ratings = clean_data
        ratings = ratings.na.drop("any")

        # Coalesce data frame into one partition so window functions are accurate
        ratings = ratings.coalesce(1)
        row_number_window = Window.partitionBy(
            "userId").orderBy("date", "movieId")
        # Get Row Numer for each user - ordered by date/movieId to ensure determinist ranking
        ratings = ratings.withColumn(
            "row_number", row_number().over(row_number_window))
        # Create percentiles for the data
        ratings = ratings.withColumn("ntile", ntile(
            10).over(row_number_window).cast("integer"))
        # Take First 6 Percentiles, label them Training, last 4 Are Evaluation
        ratings = ratings.withColumn("trainEval", when(
            ratings.ntile > 6, "Eval").otherwise("Train"))
        # Persist to ensure no data leakage
        ratings = ratings.persist()

        # Split Data into Training / Evaluation Sets
        training = ratings.filter(col("trainEval") == "Train")
        eval_data = ratings.filter(col("trainEval") == "Eval")

        # Bifurcate the Evaluation Data based on userId - This gives us our approximate splits 20% Val 20% Test Splits
        val = eval_data.filter(col("userId") % 2 == 0)
        test = eval_data.filter(col("userId") % 2 == 1)

        # Repartition for computational efficiency
        training = training.repartition(30)
        val = val.repartition(30)
        test = test.repartition(30)

        # Subtract 2.5 from each review to create negative reviews if doing baseline (Screws up ALS normalization if we do it regardless of model type)
        if self.model_type == 'baseline':
            training = training.withColumn("rating", col("rating")-2.5)

        #Val and Test will always be have rating adjusted -2.5, rating is adjusted for train in model code for ALS later
        val = val.withColumn("rating", col("rating")-2.5)
        test = test.withColumn("rating", col("rating")-2.5)
        
        #Drop uneeded columns
        training = training.drop("row_number","ntile","trainEval")
        val = val.drop("row_number","ntile","trainEval")
        test = test.drop("row_number","ntile","trainEval")
        
        # Return train/val/test splits
        return training, val, test

    def als_normalize_ratings(self, train):
        """
        Preprocess normalization method for ALS model implementation. 
        Calculates the mean ratings for each movie and for each user, then subtracts from a rating m(i,j)
        one half of the sum of the average rating for user i and the average rating of movie j
        Ratings after normalization for movie m(i,j) i-> user, j-> movie:
        rating = (rating) - .5 * ( avg(rating for user i) + avg(rating for movie j) )
        Input:
        train: PySpark DF of training data
        Output:
        train: PySpark DF with preprocessed rating
        movieAndUserMeans: PySpark DF with movieId,userId,movie_mean,user_mean to be added back to predictions
        """
        # Create window specs - average over userId and average over movieId
        windowSpecUserAgg = Window.partitionBy("userId")
        windowSpecMovieAgg = Window.partitionBy("movieId")

        # Add agg columns
        train = train.withColumn("movie_mean", avg(col("rating")).over(windowSpecMovieAgg)).\
            withColumn("user_mean", avg(col("rating")).over(windowSpecUserAgg))

        # Adjust each rating accordingly by subtracting one half of the sum of user_mean and movie_mean
        train = train.withColumn("rating", col(
            "rating")-(.5 * (col("user_mean")+col("movie_mean"))))

        return train

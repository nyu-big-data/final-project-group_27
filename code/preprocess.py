import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# And pyspark.sql to get the spark session
# from pyspark.sql import SparkSession


#A class used to preprocess data
#And save out data to Data_Partitions Folder
class DataPreprocessor():
    def __init__(self, spark, file_path) -> None:
        self.spark = spark
        self.file_path = file_path                      #File Path to Read in Data

    def clean_data(self):
        #Format Date Time

        #Fix Duplicates

        #
        
        pass
    
    def delete_dupe_ids(self):
        """
        goal: for movie titles with multiple movieIDs, in the movies dataset,
        remove the duplicate IDs with the least ratings for each movie. 
        Additionally, remove those IDs from the ratings dataset, so we get a 1:1 mapping
        between movie title and movie ID

        inputs: ratings_df, movies_df
        outputs: deduped_ratings_df, deduped_movies_df
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
        all_data = non_dupes.union(cleaned_dupes)

        #For testing purposes should be 100,830
        print(f"The length of the combined and de-deduped joined data-set is: {len(all_data.collect())}")
        #Return all_data
        return all_data

    def create_utility_matrix(self,deduped_ratings_df,deduped_movies_df):
        """
        input: deduped movies and ratings df, meaning title and movieId map 1:1
        and all duplicative movieIds have been removed from ratings_df

        returns: a sparse utility matrix, where rows are user ids and columns 
        are movie titles
        """
        # utility_matrix = deduped_ratings_df.merge(deduped_movies_df, how = 'left', on = 'movieId')
        # u = utility_matrix[['userId','title','rating']]
        # u = u.pivot(index='userId', columns = 'title', values ='rating')
        # return(u)

    def create_train_val_test_splits(self, deduped_ratings_df, deduped_movie_df, u):
        """
        inputs, deduped_ratings_df and utility matrix

        returns: 3 sparse dataframes that are corresponding 
        """

        #strategy, initialize 3 empty dataframes in the same dimmensions as 
        #our full utility matrix

        # fill = u.shape
        # train= pd.DataFrame(np.zeros(fill), columns = u.columns)
        # val = pd.DataFrame(np.zeros(fill), columns = u.columns)
        # test = pd.DataFrame(np.zeros(fill), columns = u.columns)


        # #create a ratings dataframe that has movie title as a column
        # rating_mt = deduped_ratings_df.merge(deduped_movies_df, how = 'left', on ='movieId')

        # for idx in range(len(u.index)):
        #     #for each user id, we find all the movies they watched
        #     u_id = idx+1
        #     user_ratings = rating_mt.loc[rating_mt['userId'] == u_id].sort_values(by='timestamp')

        #     #we take the first sixty percent of movie names and associated ratings through indexing
        #     #the user ratings dataframe, and taking the values of the title and ratings columns
        #     msk1 = int(len(user_ratings)*.6)
        #     train_cols, big_test_cols = list(user_ratings[:msk1].title.values), list(user_ratings[msk1:].title.values)
        #     train_vals, big_test_vals = list(user_ratings[:msk1].rating.values),list(user_ratings[msk1:].rating.values)

        #     #create a dictionary that will be passed into the .replace() method to assign the true ratings for this 
        #     #user for the given set of movies in train and big test 
        #     d_train = {train_cols[i]:{0:train_vals[i]} for i in range(len(train_cols))}
        #     #modify the row to replace 0s with the appropriate value, using replace
        #     train.iloc[idx] = train.replace(d_train).iloc[idx]

        #     #subset big_test into true validation/test sets, taking 50% of 40% (i.e. 20%)
        #     msk2 = int(len(big_test_cols)*.5)
        #     val_cols, test_cols = list(user_ratings[msk1:][:msk2].title.values), list(user_ratings[msk1:][msk2:].title.values)
        #     val_vals, test_vals = list(user_ratings[msk1:][:msk2].rating.values),list(user_ratings[msk1:][msk2:].rating.values)

        #     d_val = {val_cols[i]:{0:val_vals[i]} for i in range(len(val_cols))}
        #     d_test = {test_cols[i]:{0:test_vals[i]} for i in range(len(test_cols))}

        #     #update the validation and test datasets to be a matrix in the same dimmension as u, but with
        #     #that given user's rating for each movie 
        #     val.iloc[idx] = val.replace(d_val).iloc[idx]
        #     test.iloc[idx] = test.replace(d_test).iloc[idx]

        # #replace 0's with NaN's
        # train = train.replace(0, np.nan)
        # val = val.replace(0, np.nan)
        # test = test.replace(0, np.nan)

        # #set indicidees to match the original, my indexing was off by 1 (no userId = 0)
        # train = train.set_index(u.index)
        # val = val.set_index(u.index)
        # test = test.set_index(u.index)

        # return(train,val,test)
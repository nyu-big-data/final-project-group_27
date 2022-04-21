
#most popular recommendation as a baseline
#given a df indexed by user (should be Joby's format), aggregate all movie ratings
#then report the movie with highest average rating
#optional cutoff for minimum number of reviews to be incorporated into main

import sys
import pandas as pd

class ReccomendationSystem:
    def __init__(self, data=None, min_reviews=None) -> None:
        self.min_reviews = min_reviews
        self.data = data
        #Some other args...

    #Maybe have a recommend method that does some behavior based on args passed through
    def reccomend(self):
        # self.condense_popularity()
        # self.min_review_cutoff()
        # self. some behavior...
        pass

    #Design choice, make these in place vs return new df
    def condense_popularity(self):
        """
        input: dataframe of ratings indexed by user

        output: df condensed to have average rating and number of reviews by MOVIE

        qs: should we do by movieID and should we pass the orig df indexed by movieID
        """

        #first get the number of non-zero values in each COLUMN
        reviewsByMovie = self.data.astype(bool).sum(axis=0)

        #get average rating for each movie
        averageRating = self.data.mean()

        outDF = pd.DataFrame({"movie":self.data.columns,"num_reviews":reviewsByMovie,"average_rating":averageRating})

        return outDF

    def min_review_cutoff(self):
        """
        input_dataFrame: output of condense popularity
        min_reviews: int, minimum number of reviews to be considered for recommendation

        """

        #drop movies with too few reviews
        arg_max = self.data.where(self.data["num_reviews"]>=self.min_reviews).idxmax(axis = 0)["average_rating"]

        return self.data.loc[arg_max]["movie"]

    def fastmax(self):
        """
        use this function if there is no minimum number of reviews cutoff

        outputs: movie with the highest ave rating
        """

        return self.data.mean().argmax()



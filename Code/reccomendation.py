
#Imports
import sys
import pandas as pd
import pyspark

#Class to be imported
class ReccomendationSystem:
    def __init__(self, data=None, min_reviews=None, keepNum=None) -> None:
        self.min_reviews = min_reviews
        self.data = data
        self.keepNum = keepNum

    #Maybe have a recommend method that does some behavior based on args passed through
    def reccomend(self):
        """
        Method that should include most behavior pased on parameters passed through to the ReccomendationSystem object
        """
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
        Only recommend most popular movies with a certain number of reviews.
        """

        #drop movies with too few reviews
        filtered = self.data.where(self.data["num_reviews"]>=self.min_reviews).dropna()[["movie"],["average_rating"]]

        #return the first self.keepNum titles
        return filtered.sort_values(by = "average_rating", ascending = False)["movie"][:self.keepNum].tolist()

    def fastmax(self):
        """
        use this function if there is no minimum number of reviews cutoff

        outputs: movie with the highest ave rating
        """

        #return first n most popular movies
        return self.data.mean().sort(ascending = False).index()[:self.keepNum]


#Some change
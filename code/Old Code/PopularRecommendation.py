
#most popular recommendation as a baseline
#given a df indexed by user (should be Joby's format), aggregate all movie ratings
#then report the movie with highest average rating
#optional cutoff for minimum number of reviews to be incorporated into main

import sys
import pandas as pd

def condense_popularity (input_dataFrame):
    """
    input: dataframe of ratings indexed by user

    output: df condensed to have average rating and number of reviews by MOVIE

    qs: should we do by movieID and should we pass the orig df indexed by movieID
    """

    #first get the number of non-zero values in each COLUMN
    reviewsByMovie = input_dataFrame.astype(bool).sum(axis=0)

    #get average rating for each movie
    averageRating = input_dataFrame.mean()

    outDF = pd.DataFrame({"movie":input_dataFrame.columns,"num_reviews":reviewsByMovie,"average_rating":averageRating})

    return outDF

def min_review_cutoff (input_dataFrame, min_reviews):
    """
    input_dataFrame: output of condense popularity
    min_reviews: int, minimum number of reviews to be considered for recommendation

    """

    #drop movies with too few reviews
    arg_max = input_dataFrame.where(input_dataFrame["num_reviews"]>=min_reviews).idxmax(axis = 0)["average_rating"]

    return input_dataFrame.loc[arg_max]["movie"]

def fastmax(input_dataFrame):
    """
    use this function if there is no minimum number of reviews cutoff

    outputs: movie with the highest ave rating
    """

    return input_dataFrame.mean().argmax()

def main(input_dataFrame, min_reviews=1):

    if min_reviews > 1:

        condensed = condense_popularity(input_dataFrame)
        min_review_cutoff(condensed, min_reviews)

    else:
        fastmax(input_dataFrame)

if __name__ == "__main__":

    input_dataFrame = sys.argv[0]

    if (sys.argv[1]):

        main(input_dataFrame, min_reviews = sys.argv[1])

    else:

        main(input_dataFrame)


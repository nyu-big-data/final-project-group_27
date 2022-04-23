import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import inspect

class Model():
    """
    Abstract Model class that will contain various methods to deploy collaborative filtering.
    Model Parameters that need to be passed thorugh:
    -----
    rank: int - Rank of latent factors used in decomposition
    maxIter: int - represents number of iterations to run algorithm
    regParam: float - Regularization Parameter
    seed: int - Random seed, default set to 10 for reproducibility
    nonnegative: boolean - Flag as to whether or not outputs can be negative or positive
    alpha: float - Alpha parameter in case we do any implicitFeedback methods
    model_save: boolean - Flag to determine if we should save the model progress or not
    num_recs: int - Top X number of reccomendations to return - default set to 100
    -----
    Model methods:
    -----
    ALS_fit_and_run: Latent Factor model which uses the Alternating Least Squares Pyspark Class to fit and predict.
    Baseline: uses a baseline popularity model that returns the top X most popular movies (avg of movie rating)

    -----
    """

    #Constructor for Model
    def __init__(self, rank=10, maxIter=5,regParam=0.01, \
                    seed=10,nonnegative=True,alpha=1,model_save=False, \
                    num_recs=100):
        #Model Attributes                    
        self.rank = rank                                                                    #Rank of latent factors used in decomposition
        self.maxIter = maxIter                                                              #Number of iterations to run algorithm, recommended 5-20
        self.regParam = regParam                                                            #Regularization Parameter
        self.seed = seed                                                                    #Random seed - default set to 10 for reproducibility
        self.nonnegative = nonnegative                                                      #Flag as to whether or not outputs can be negative or positive
        self.alpha = alpha                                                                  #Alpha parameter in case we do any implicitFeedback methods
        self.num_recs = num_recs                                                            #Top X number of reccomendations to return
        self.model_save = model_save                                                        #Flag used to determine whether or not we should save our model somewhere
        self.model_save_path = const.MODEL_SAVE_FILE_PATH                                   #NO Arg needed to be passed thorugh
        self.results_file_path = const.RESULTS_SAVE_FILE_PATH                               #Filepath to write model results like rmse and model params


    #This method uses the Alternating Least Squares Pyspark Class to fit and run a model
    def ALS_fit_and_run(self, training, val, test):
        """
        -----
        Input: train,val,test sets
        -----
        Output: userRecs, movieRecs - list of Top self.num_recs long for reccommendations
        -----
        """

        #Create the model with certain params - coldStartStrategy="drop" means that we'll have no nulls in val / test set
        als = ALS(maxIter=5, rank=5, regParam=0.01, nonnegative = True, seed=10, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        #Fit the model
        model = als.fit(training)

        #Create predictions
        predictions = model.transform(val)
        #Evalaute Predictions
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        #Calculate RMSE
        rmse = evaluator.evaluate(predictions)

        #Print out predictions
        print(f"Root-mean-square error for Val = {rmse}")

        #Write our results and model parameters
        with open(self.results_file_path, 'a') as output_file:
            print("Recording the following: Model, rmse, rank, maxIter, regParam, nonnegative")
            output_file.write(f"ALS Model, {rmse},{self.rank},{self.maxIter},{self.regParam},{self.nonnegative}")

        # Generate top 10 movie recommendations for each user
        userRecs = model.recommendForAllUsers(self.num_recs)
        # Generate top 10 user recommendations for each movie
        movieRecs = model.recommendForAllItems(self.num_recs)

        #Save model if we need to
        if self.model_save:
            self.save_model(als)

        #Return top self.num_recs movie recs for each user, top self.num_recs user recs for each movie
        return userRecs, movieRecs

    #Baseline model that returns top X most popular items (highest avg rating)
    def baseline(self, training, val, test):
        pass

    #Method to save model to const.MODEL_SAVE_FILE_PATH
    def save_model(self, model):
        model.save(const.MODEL_SAVE_FILE_PATH)

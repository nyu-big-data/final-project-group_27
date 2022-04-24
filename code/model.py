import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics 
import time
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
    def __init__(self, rank=10, maxIter=5, regParam=0.01, seed=10, nonnegative=True, \
                                            alpha=1, model_save=False, num_recs=100):
        #Model Attributes                    
        self.rank = rank                                                                    #Rank of latent factors used in decomposition
        self.maxIter = maxIter                                                              #Number of iterations to run algorithm, recommended 5-20
        self.regParam = regParam                                                            #Regularization Parameter
        self.seed = seed                                                                    #Random seed - default set to 10 for reproducibility
        self.nonnegative = nonnegative                                                      #Flag as to whether or not outputs can be negative or positive - default False
        self.alpha = alpha                                                                  #Alpha parameter in case we do any implicitFeedback methods
        self.num_recs = num_recs                                                            #Top X number of reccomendations to return
        self.model_save = model_save                                                        #Flag used to determine whether or not we should save our model somewhere
        self.model_save_path = const.MODEL_SAVE_FILE_PATH                                   #NO Arg needed to be passed thorugh
        self.results_file_path = const.RESULTS_SAVE_FILE_PATH                               #Filepath to write model results like rmse and model params


    #This method uses the Alternating Least Squares Pyspark Class to fit and run a model
    def ALS_fit_and_run(self, training=None, val=None, test=None):
        """
        -----
        Input: train,val,test sets
        -----
        Output: userRecs, movieRecs - list of Top self.num_recs long for reccommendations
        -----
        """
        
        #Record dummy variable (used later in writing results) if we're evaluating Val or Test predictions
        if val:
            predicted_set = "Val"
            input = val
        else:
            predicted_set = "Test"
            input = test

        #Time the function start to finish
        start = time.time()
        #Create the model with certain params - coldStartStrategy="drop" means that we'll have no nulls in val / test set
        als = ALS(maxIter=self.maxIter, rank=self.rank, regParam=self.regParam, nonnegative = self.nonnegative, seed=self.seed, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        #Fit the model
        model = als.fit(training)
        #End time and calculate delta
        end = time.time()
        time_elapsed_train = end - start

        #Time predictions as well
        start = time.time()
        #Create predictions
        predictions = model.transform(input)
        ebd = time.time()
        time_elapsed_predict = end - start
        
        #Use self.record_metrics to evaluate model on RMSE, R^2, Precision at K, Mean Precision, and NDGC
        self.record_metrics(predictions, input,model="ALS",\
                            predicted_set=predicted_set,time_elapsed_train=time_elapsed_train,\
                            time_elapsed_predict=time_elapsed_predict)
        #Ranking metrics test
        metrics = RankingMetrics(model) #expects: predictionAndLabels : :py:class:`pyspark.RDD`an RDD of (predicted ranking, ground truth set) pairs.
        #average_precision = metrics.meanAveragePrecision(10)

        # Generate top 10 movie recommendations for each user
        userRecs = model.recommendForAllUsers(self.num_recs)
        # Generate top 10 user recommendations for each movie
        movieRecs = model.recommendForAllItems(self.num_recs)

        #Save model if we need to
        if self.model_save:
            self.save_model(model_type="ALS", model=als)

        #Return top self.num_recs movie recs for each user, top self.num_recs user recs for each movie
        return userRecs, movieRecs

    #Baseline model that returns top X most popular items (highest avg rating)
    def baseline(self, training, val, test):
        pass
    
    def record_metrics(self, predictions,labels, model,predicted_set,time_elapsed_train, time_elapsed_predict):
        """
        Method that will contain all the code to evaluate model on metrics: RMSE, R^2, Precistion At K, Mean Precision, and NDGC
        input:
        -----
        predictions:
        labels:
        mode: String - Type of model being evaluated
        predicted_set: String - Which set are we making predictions on? Either Val or Test
        time_elapsed_train: time.time() - How long it took the model to train
        time_elapsed_predict: time.time() - How long it took the model to predict
        -----
        returns: None - but writes the results to results.txt in /gjd9961/scratch/big_data_final_results/results.txt
        """

        #Evalaute Predictions

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        #Calculate RMSE
        rmse = evaluator.evaluate(predictions)

        #Print out predictions
        print(f"Root-mean-square error for Val = {rmse}")


        ## TO DO: record results for Precistion at K, Mean Precision, and NDGC
        #Write our results and model parameters
        with open(self.results_file_path, 'a') as output_file:
            print("Recording the following: Net ID, Time When it Ran, Model Type, Set Predicted, Train Time, Predict Time, RMSE, Rank, MaxIter, RegParam, NonNegative")
            output_file.write(f"{const.netID},{time.time()},ALS Model, \
                {predicted_set},{time_elapsed_train},{time_elapsed_predict},\
                    {rmse},{self.rank},{self.maxIter},{self.regParam},{self.nonnegative}")

    #Method to save model to const.MODEL_SAVE_FILE_PATH
    def save_model(self, model_type=None, model=None):
        """
        Inputs:
        -----
        model_type: str - string designating what type of model is being saved
        model: obj - model object that has .save method
        -----
        """
        #Make sure a non-null object was passed
        if model:
            model.save(const.MODEL_SAVE_FILE_PATH + model_type)
        #Otherwise throw error
        else:
            raise Exception("Model not passed through")

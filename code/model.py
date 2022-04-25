import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics 
import time
from datetime import datetime
from pyspark.mllib.evaluation import RankingMetrics 
from pyspark.sql import Row
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.sql.window import Window


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
                                            alpha=1, model_save=False, num_recs=100, min_reviews =10 ):
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
        self.min_reviews = min_reviews                                                      #minimum number of reviews to qualify for baseline
        
        #This method uses the Alternating Least Squares Pyspark Class to fit and run a model
    def ALS_fit_and_predict(self, training=None, val=None, test=None):
        """
        Builds and fits an ALS latent factor model. Calls self.record_metrics(precitions,labels,model_params)
        to record the results. Some dummy variables are made to record whether or not we are using the validation set
        or the testing set. This will help us record our results accurately. Training and predicting are also timed. 
        -----
        Input: train,val,test sets
        -----
        Output: userRecs, movieRecs - list of Top self.num_recs long for reccommendations
        -----
        """
        
        #Record dummy variable (used later in writing and evaluating results) if we're evaluating Val or Test predictions
        if val:
            predicted_set = "Val" #Gets written out by record_metrics
            predicted_data = val #Input gets passed to evaluator as labels
        else:
            predicted_set = "Test" #Gets written out by record_metrics
            predicted_data = test #Input gets passed to evaluator as labels

        #Time the function start to finish
        start = time.time()
        #Create the model with certain params - coldStartStrategy="drop" means that we'll have no nulls in val / test set
        als = ALS(maxIter=self.maxIter, rank=self.rank, regParam=self.regParam,\
                    nonnegative = self.nonnegative, seed=self.seed, userCol="userId", \
                    itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        #Fit the model
        model = als.fit(training)
        #End time and calculate delta
        end = time.time()
        time_elapsed_train = end - start

        #Time predictions as well
        start = time.time()
        #Create predictions
        predictions = model.transform(predicted_data)
        end = time.time()
        time_elapsed_predict = end - start

        #Get when model was ran
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        #Package Parameters into a dictionary to ease recording
        model_params = {"Net ID": const.netID,
                        "Time when it ran": now,
                        "model_type":"ALS",
                        "predicted_set":predicted_set,
                        "time_elapsed_train":time_elapsed_train,
                        "time_elapsed_predict": time_elapsed_predict,
                        "Rank":self.rank,
                        "maxIter":self.maxIter,
                        "regParam":self.regParam,
                        "NonNegative":self.nonnegative,
                        "Random Seed":self.seed}

        #Use self.record_metrics to evaluate model on RMSE, R^2, Precision at K, Mean Precision, and NDGC
        self.record_metrics(predictions, labels=predicted_data,model_params=model_params)
        
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
        """
        spark df: joined df with all user reviews
        keepNum: how many of the most popular movies we want to keep (should be 100 it seems)
        min_count: how many reviews a movie must have in order to be considered
    
        output: list of n movie titles where n=num_recs... future format tbd
        """
    
        spark_df.createOrReplaceTempView("joined")
        result = spark.sql(f"SELECT title, AVG(rating) as A, COUNT(userID) FROM joined GROUP BY movieID HAVING COUNT(userID)>={self.min_count} ORDER BY A DESC LIMIT {self.num_recs}")
        out=result.select("movieID").collect()
    
        return(out)
    
    def record_metrics(self, predictions,labels, model_params):
        """
        Method that will contain all the code to evaluate model on metrics: RMSE, R^2, Precistion At K, Mean Precision, and NDGC
        input:
        -----
        predictions:
        labels:
        param_dict: dict - stores all the params passed to model that we will record
        -----
        returns: None - but writes the results to results.txt in /gjd9961/scratch/big_data_final_results/results.txt
        """
        
        ##Evaluate Predictions for Regression Task##
        evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction")
        #Calculate RMSE and r_2 metrics
        rmse = evaluator.evaluate(predictions,{evaluator.metricName: "rmse"})
        r_2 = evaluator.evaluate(predictions,{evaluator.metricName: "r2"})
        #Package our model parameters and metrics neatly so its easy to write
        metrics = [rmse, r_2]
        
        ##Evalaute Predictions for Ranking Tests##
        
        #Make predictions Binary
        predictions = predictions.withColumn("prediction",when(predictions.rating >0,1).otherwise(0).cast("double"))
        #Window function to partition by userId predictions in descending order
        windowSpec_pred = Window.partitionBy('userId').orderBy(col('prediction').desc())
        #Window function to partition reviews in the validation set by user id sort by date
        windowSpec_label = Window.partitionBy('userId').orderBy(col('date').desc())

        #Grab oldest watched movies for each user - Ouput RDD with cols userId, movieId where movieId is a list of watched movie ids -> [movieId,...]
        labels = labels.select('userId', 'movieId', 'date', rank().over(windowSpec_label).alias('rank')) \
                                    .groupBy('userId').agg(expr('collect_list(movieId) as items'))
        #Order predictions by high confidence to low - Ouput RDD with cols userId, movieId where movieId is a list of reccomended movie ids -> [movieId,...]
        prediction_subset = predictions.select('userId', 'movieId', 'prediction', rank().over(windowSpec_pred).alias('rank')) \
                                        .groupBy('userId').agg(expr('collect_list(movieId) as movies'))
        
        #Calculate MAP over 100 recs at the end (The value is independent of K so we just do it once)
        labelsAndPredictions = prediction_subset.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(labelsAndPredictions)
        MAP = rankingMetrics.meanAveragePrecision #No function call necessary
        metrics.append(MAP)

        #Get the per user predicted Items, Iterate for different values of K
        for k in [10,20,50,100]:
            #Grab the K most confident predictions
            prediction_subset = predictions.select('userId', 'movieId', 'prediction', rank().over(windowSpec_pred).alias('rank')) \
                                        .where(f'rank <= {k}').groupBy('userId').agg(expr('collect_list(movieId) as movies'))

            #Join together with the labels, the lambda makes a RDD of structure userId, movieId where movie id is a tuple in the form -> ([JavaList],[Java.List]])
            labelsAndPredictions = prediction_subset.join(labels, 'userId').rdd.map(lambda row: (row[1], row[2]))
            #Initialize a ranking metrics object with the joined data
            rankingMetrics = RankingMetrics(labelsAndPredictions)
            #Calculate and Append MAP at K, NDCG at K
            metrics.append(rankingMetrics.meanAveragePrecisionAt(k))
            metrics.append(rankingMetrics.ndcgAt(k))

        ##ROC Metric Evaluation##
        #For ROC Binary Classification
        evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='rating', metricName='areaUnderROC')
        #Append ROC to our Metrics list
        metrics.append(evaluator.evaluate(predictions))

        #Convert dictionary values to list
        model_args = list(model_params.values())
        #Add the metrics to our list - in place
        model_args.extend(metrics)

        #Write our results and model parameters
        print("Recording the following: model_params")
        with open(self.results_file_path, 'a') as output_file:
            #Write each element of metrics seperated by a comma, for the last one use a new line character
            for i in range(len(model_args)-1):
                output_file.write(f"{model_args[i]},") 
            output_file.write(f"{model_args[-1]}\n")           

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
        if model and model_type:
            model.save(const.MODEL_SAVE_FILE_PATH + model_type)
        #Otherwise throw error
        else:
            raise Exception("Model and or Model_type not passed through")

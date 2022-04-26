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
    ### For ALS Model ###
    -----
    rank: int - Rank of latent factors used in decomposition
    maxIter: int - represents number of iterations to run algorithm
    regParam: float - Regularization Parameter
    model_save: boolean - Flag to determine if we should save the model progress or not
    -----
    ### For baseline Model ###
    -----
    min_ratings: int - Minimum number of reviews to qualify for baseline (Greater Than or Equal to be included)
    -----
    ### No Input Necessary ###
    -----
    model_size: str - Either "large" or "small" used to demarcate which dataset we are running on
    model_type: str - Which model type we intent to run, i.e. ALS or baseline
    evaluation_data_name: str - Dummy variable used to keep track of which dataset we are making predictions on, either "Val" or "Test"
    time_when_ran: datetime - Time when model was run
    time_to_fit: datetime - Time it took to fit the model
    time_to_predict: datetime - Time it took to make predictions
    metrics: dict - Dictionary used to store the various metrics calculated in self.record_metrics()
    -----
    ### Misc ###
    -----
    num_recs: int - Top X number of reccomendations to return - default set to 100
    -----
    ### Model Methods ###
    -----
    run_model: Runs the corresponding method that was passed to self.model_type
    alternatingLeastSquares: Latent Factor model which uses the Alternating Least Squares Pyspark Class to fit and predict.
    baseline: uses a baseline popularity model that returns the top X most popular movies (decided by avg rating per movie)
    record_metrics: Calculates metrics for prediction,label pairs
    save_model: Used for advanced models like ALS or extensions where we may want to save the model itself
    -----
    """

    # Constructor for Model
    def __init__(self, model_size=None, model_type=None, rank=None, maxIter=None, regParam=None, seed=10, nonnegative=True,
                 model_save=False, num_recs=100, min_ratings=None):
        # Model Attributes
        # NO Arg needed to be passed thorugh
        self.netID = const.netID
        self.model_save_path = const.MODEL_SAVE_FILE_PATH
        # Filepath to write model results like rmse and model params
        self.results_file_path = const.RESULTS_SAVE_FILE_PATH
        # Dictionary to access variable methods
        self.methods = {"als": self.alternatingLeastSquares,
                        "baseline": self.baseline}
        self.num_recs = num_recs  # Top X number of reccomendations to return - set to 100, probably won't change

        # Passed through by user
        self.model_size = model_size
        self.model_type = model_type

        # For ALS
        self.rank = rank  # Rank of latent factors used in decomposition
        self.maxIter = maxIter  # Number of iterations to run algorithm, recommended 5-20
        self.regParam = regParam  # Regularization Parameter
        self.model_save = model_save # Flag used to determine whether or not we should save our model somewhere
        
        # For baseline
        self.min_ratings = min_ratings # Minimum number of reviews to qualify for baseline (Greater Than or Equal to be included)

        # Add the attributes we're gonna compute when we fit and predict
        self.evaluation_data_name = None
        self.time_when_ran = None
        self.time_to_fit = None
        self.time_to_predict = None
        self.metrics = {}

    def run_model(self, train, val=None, test=None):
        """
        Run_model is what is called to fit, run, and record the metrics for respective model types.
        Function behavior is dependent on the argument passed to self.model_type.
        -----
        inputs:
        -----
        train: RDD - Training data set
        val: RDD - Validation data set
        test: RDD - Test set
        -----
        outputs:
        -----
        model_output: Variable Type - Output of whichever model ran -> check self.model_type
        -----
        """
        # Get when model was ran
        self.time_when_ran = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        # Identify if we're predicting on the Validation Set or the Test Set
        if val:
            self.evaluation_data_name = "Val"
            evaluation_data = val
        elif test:
            self.evaluation_data_name = "Test"
            evaluation_data = test

        # Grab method for whichever model corresponds to self.model_type
        model = self.methods[self.model_type]
        # Run model on training / evaluation data
        model_output = model(train, evaluation_data)
        # Return model output
        return model_output

    # This method uses the Alternating Least Squares Pyspark Class to fit and run a model
    def alternatingLeastSquares(self, training, evaluation_data):
        """
        Builds and fits a PySpark alternatingLeastSquares latent factor model. Calls self.record_metrics(precitions,labels)
        to record the results. Some dummy variables are made to record whether or not we are using the validation set
        or the testing set. This will help us record our results accurately. Training and predicting are also timed. 
        -----
        Input: 
        training: RDD - Training data set
        evaluation_data: RDD - Either Validation data set, or Training data set
        -----
        Output: [userRecs, movieRecs] - list containing two lists, each of length == self.numrecs 
        -----
        """

        # Time the function start to finish
        start = time.time()
        # Create the model with certain params - coldStartStrategy="drop" means that we'll have no nulls in val / test set
        als = ALS(maxIter=self.maxIter, rank=self.rank, regParam=self.regParam,
                  nonnegative=False, seed=10, userCol="userId",
                  itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

        # Fit the model
        model = als.fit(training)
        # End time and calculate delta
        end = time.time()
        self.time_to_fit = end - start

        # Time predictions as well
        start = time.time()
        # Create predictions, matrix with additional column of prediction
        predictions = model.transform(evaluation_data)
        end = time.time()
        self.time_to_predict = end - start

        # Use self.record_metrics to evaluate model on RMSE, R^2, Precision at K, Mean Precision, and NDGC
        self.record_metrics(predictions=predictions, labels=evaluation_data)

        # Generate top 10 movie recommendations for each user
        userRecs = model.recommendForAllUsers(self.num_recs)
        # Generate top 10 user recommendations for each movie
        movieRecs = model.recommendForAllItems(self.num_recs)

        # Save model if we need to
        if self.model_save:
            self.save_model(model_type=self.model_type, model=als)

        # Return top self.num_recs movie recs for each user, top self.num_recs user recs for each movie
        return [userRecs, movieRecs]

    # Baseline model that returns top X most popular items (highest avg rating)
    def baseline(self, training, evaluation_data):
        """
        Baseline model for recommendation system. No personalization, just recommend the Top 100 movies by avg(rating)
        A movie must have at least self.min_ratings to be considered
        input:
        -----
        training: RDD - training set data
        evaluation_data: RDD - Validation set or Test set data
        self.min_ratings: int - how many ratings a movie must have in order to be considered in top 100
        -----
        output: RDD of Top 100 movieIds by avg(rating)
        """
        # Time model Fit
        start = time.time()
        # Get Top 100 Most Popular Movies - Avg(rating) becomes prediction
        top_100_movies = training.groupBy("movieId").agg(avg("rating").alias("prediction"),
                                                         count("movieId").alias("movie_count")).where(f"movie_count>={self.min_ratings}").\
            orderBy("prediction", ascending=False).limit(100)
        # Grab Distinct User Ids
        ids = evaluation_data.select("userId").distinct()
        # Cross Join Distinct userIds with Top 100 Most Popular Movies
        predictions = ids.crossJoin(top_100_movies)
        # Record end time after RDD operations
        end = time.time()
        self.time_to_fit = end - start

        # Time predictions as well
        self.time_to_predict = 0  # Recommends in constant time

        # Use self.record_metrics to evaluate model on RMSE, R^2, Precision at K, Mean Precision, and NDGC
        self.record_metrics(predictions=predictions, labels=evaluation_data)

        # Return The top 100 most popular movies above self.min_ratings threshold
        return top_100_movies

    def record_metrics(self, predictions, labels):
        """
        Method that will contain all the code to evaluate model on metrics: RMSE, R^2, ROC, Precistion At K, Mean Precision, and NDGC
        input:
        -----
        predictions: RDD - PySpark Dataframe containing the following columns at the minimum: [userId,movieId,prediction] - if not baseline model must include rating column
        labels: RDD - PySpark Dataframe containing the following columns at the minimum: [userId,movieId,rating, date]
        -----
        returns: 
        None - Writes the results to self.metrics dictionary
        """
        if self.model_type != 'baseline':
            ##Evaluate Predictions for Regression Task##
            evaluator = RegressionEvaluator(
                labelCol="rating", predictionCol="prediction")
            # Calculate RMSE and r_2 metrics and append to metrics
            self.metrics["rmse"] = evaluator.evaluate(
                predictions, {evaluator.metricName: "rmse"})
            self.metrics["r2"] = evaluator.evaluate(
                predictions, {evaluator.metricName: "r2"})

            ##ROC Metric Evaluation##
            # For ROC Binary Classification
            # Make predictions Binary
            binary_predicts = predictions.withColumn("prediction", when(
                predictions.rating > 0, 1).otherwise(0).cast("double"))
            evaluator = BinaryClassificationEvaluator(
                rawPredictionCol='prediction', labelCol='rating', metricName='areaUnderROC')
            # Append ROC to our Metrics list
            self.metrics["ROC"] = evaluator.evaluate(binary_predicts)
        else:
             self.metrics["rmse"] = np.nan
             self.metrics["r2"] = np.nan
             self.metrics["ROC"] = np.nan

        ##Evalaute Predictions for Ranking Tests##

        # Window function to partition by userId predictions in descending order
        windowSpec_pred = Window.partitionBy(
            'userId').orderBy(col('prediction').desc())
        # Window function to partition reviews in the validation set by user id sort by date
        windowSpec_label = Window.partitionBy(
            'userId').orderBy(col('date').desc())

        # Grab oldest watched movies for each user - Ouput RDD with cols userId, movieId where movieId is a list of watched movie ids -> [movieId,...]
        labels = labels.select('userId', 'movieId', 'date', rank().over(windowSpec_label).alias('rank')) \
            .groupBy('userId').agg(expr('collect_list(movieId) as items'))
        # Order predictions by high confidence to low - Ouput RDD with cols userId, movieId where movieId is a list of reccomended movie ids -> [movieId,...]
        prediction_subset = predictions.select('userId', 'movieId', 'prediction', rank().over(windowSpec_pred).alias('rank')) \
            .groupBy('userId').agg(expr('collect_list(movieId) as movies'))

        # Calculate MAP over 100 recs at the end (The value is independent of K so we just do it once)
        labelsAndPredictions = prediction_subset.join(
            labels, 'userId').rdd.map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(labelsAndPredictions)
        # No function call necessary
        self.metrics["MAP"] = rankingMetrics.meanAveragePrecision

        # Get the per user predicted Items, Iterate for different values of K
        for k in [10, 20, 50, 100]:
            # Grab the K most confident predictions
            prediction_subset = predictions.select('userId', 'movieId', 'prediction', rank().over(windowSpec_pred).alias('rank')) \
                .where(f'rank <= {k}').groupBy('userId').agg(expr('collect_list(movieId) as movies'))

            # Join together with the labels, the lambda makes a RDD of structure userId, movieId where movie id is a tuple in the form -> ([JavaList],[Java.List]])
            labelsAndPredictions = prediction_subset.join(
                labels, 'userId').rdd.map(lambda row: (row[1], row[2]))
            # Initialize a ranking metrics object with the joined data
            rankingMetrics = RankingMetrics(labelsAndPredictions)
            # Calculate and Append MAP at K, NDCG at K
            self.metrics[f"meanAveragePrecisionAt{k}"] = rankingMetrics.meanAveragePrecisionAt(
                k)
            self.metrics[f"ndcgAt{k}"] = rankingMetrics.ndcgAt(k)

    # Method to save model to const.MODEL_SAVE_FILE_PATH
    def save_model(self, model_type=None, model=None):
        """
        Inputs:
        -----
        model_type: str - string designating what type of model is being saved
        model: obj - model object that has .save method
        -----
        """
        # Make sure a non-null object was passed
        if model and model_type:
            model.save(const.MODEL_SAVE_FILE_PATH + model_type)
        # Otherwise throw error
        else:
            raise Exception("Model and or Model_type not passed through")

import numpy as np
import pandas as pd
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
import code.constants as const
from code.unit_tests import UnitTest

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
    def __init__(self, model_size=None, model_type=None, rank=None, maxIter=None, regParam=None,
                 model_save=False, num_recs=100, min_ratings=0, positive_rating_threshold=0, k=100, sanity_check = None):
        # Model Attributes
        # NO Arg needed to be passed thorugh
        # Dictionary to access variable methods
        self.netId = const.netID
        self.methods = {"als": self.alternatingLeastSquares,
                        "baseline": self.baseline}
        # Top X number of reccomendations to return - set to 100, probably won't change
        self.num_recs = num_recs
        self.k = k
        # Passed through by user
        self.model_size = model_size
        self.model_type = model_type
        self.positive_rating_threshold = positive_rating_threshold
        self.sanity_check = sanity_check
        # For ALS
        self.rank = rank  # Rank of latent factors used in decomposition
        self.maxIter = maxIter  # Number of iterations to run algorithm, recommended 5-20
        self.regParam = regParam  # Regularization Parameter
        # (Optional) Flag used to determine whether or not we should save our model somewhere
        self.model_save = model_save

        # For baseline
        # Minimum number of reviews to qualify for baseline (Greater Than or Equal to be included)
        self.min_ratings = min_ratings

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

        #Check for leakage between the sets
        if self.sanity_check:
            tester = UnitTest()
            if tester.data_leakage_check(train=train,val=evaluation_data) == False:
                raise Exception("Data Leakage Occured - Check stdout")
            else:
                print(f"Passed Data Leakage Check Between Train and {self.evaluation_data_name}")
                
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
        regression_predictions = model.transform(evaluation_data)
        # Generate 100 Top Movies for All Users
        userRecs = model.recommendForAllUsers(self.num_recs)

        # Unpack userRecs, go from userId, list({movieId:predicted_rating}) -> userId, movieId
        ranking_predictions = userRecs.select(
            "userId", explode("recommendations.movieId").alias("movieId"))
        end = time.time()
        self.time_to_predict = end - start

        # Use self.record_metrics to evaluate model on Precision at K, Mean Precision, and NDGC
        self.metrics['MAP'], self.metrics[f'precisionAt{self.k}'], self.metrics[f'recallAt{self.k}'], self.metrics[f'ndgcAt{self.k}'] = self.OTB_ranking_metrics(
            preds=ranking_predictions, labels=evaluation_data, k=self.k)
        # Use self.non_ranking_metrics to compute RMSE, R^2, and ROC of Top 100 Predictions - No special Filtering ATM
        self.metrics['RMSE'], self.metrics['R2'], self.metrics['ROC'], = self.non_ranking_metrics(
            regression_predictions)

        # Save model if we need to
        if self.model_save:
            self.save_model(model_type=self.model_type, model=als)

        # Return top self.num_recs movie recs for each user
        return userRecs

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
        # Make sure the right params have been passed to Model()
        if self.min_ratings is None:
            raise Exception(
                "Must pass through a value for self.min_ratings for baseline to compute")

        # Time model Fit
        start = time.time()
        # Get Top 100 Most Popular Movies - Avg(rating) becomes prediction
        temp = training
        top_100_movies = temp.groupBy("movieId").agg(avg("rating").alias(
            "prediction"), count("movieId").alias("movie_count"))
        top_100_movies = top_100_movies.where(
            col("movie_count") >= self.min_ratings)
        top_100_movies = top_100_movies.select("movieId").orderBy(
            "prediction", ascending=False).limit(100)

        # Grab Distinct User Ids
        temp2 = evaluation_data
        ids = temp2.select("userId").distinct()
        # Cross Join Distinct userIds with Top 100 Most Popular Movies
        predictions = ids.crossJoin(top_100_movies)

        # Record end time after RDD operations
        end = time.time()
        self.time_to_fit = end - start

        #If sanity_checker = True then 
        if self.sanity_check == True:
            tester = UnitTest()
            tester.baseline_prediction_check(preds=predictions)

        # Time predictions as well
        self.time_to_predict = 0  # Recommends in constant time
        self.metrics['MAP'], self.metrics[f'precisionAt{self.k}'], self.metrics[f'recallAt{self.k}'], self.metrics[f'ndgcAt{self.k}'] = self.OTB_ranking_metrics(
            preds=predictions, labels=evaluation_data, k=self.k)

        self.metrics["Custom Precision"], self.metrics["Custom Recall"] = self.baseline_CUSTOM_ranking_metrics(preds=predictions, labels=evaluation_data)
        # Return The top 100 most popular movies above self.min_ratings threshold
        return predictions

    # Non-Ranking Metrics Calculated Here
    def non_ranking_metrics(self, predictions):
        """
        Input: 
        predictions
        Output:
        rmse: evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        r2: evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        roc: evaluator.evaluate(binary_predicts)
        """
        ##Evaluate Predictions for Regression Task##
        evaluator = RegressionEvaluator(
            labelCol="rating", predictionCol="prediction")
        # Calculate RMSE and r_2 metrics and append to metrics
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        ##ROC Metric Evaluation##
        # Make predictions Binary
        binary_predicts = predictions.withColumn("prediction", when(
            predictions.rating > 0, 1).otherwise(0).cast("double"))
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol='prediction', labelCol='rating', metricName='areaUnderROC')
        # Append ROC to our Metrics list
        roc = evaluator.evaluate(binary_predicts)
        return rmse, r2, roc

    def OTB_ranking_metrics(self, preds, labels, k):
        """
        Input: 
        preds: DF - Tall DF of userId, movieId predictions
        labels: DF - 
        k: int - used in precisionAt(k), ndgcAt(k)
        Output:
        rankingMetrics.meanAveragePrecision
        rankingMetrics.recallAt(k)
        rankingMetrics.ndcgAt(k)
        """

        predictions = preds \
            .select('userId', 'movieId')\
            .groupBy('userId') \
            .agg(expr('collect_list(movieId) as movieId'))

        labels = labels \
            .select('userId', 'movieId')\
            .groupBy('userId') \
            .agg(expr('collect_list(movieId) as movieId'))

        predictionsAndLabels = predictions.join(broadcast(labels), 'userId', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        rankingMetrics = RankingMetrics(predictionsAndLabels)
        return rankingMetrics.meanAveragePrecision, rankingMetrics.precisionAt(k), rankingMetrics.recallAt(k), rankingMetrics.ndcgAt(k)

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

    def baseline_CUSTOM_ranking_metrics(self, preds, labels):
        """
        Input: preds, labels - PySpark DFs
        output: precision, recall - Float
        """
        # Dummy var
        precision_arr = []
        # Collect movie_recs into set
        movie_recs = preds.select("userId", "movieId")\
            .groupBy(col("userId"))\
            .agg(collect_list(col("movieId")).alias("movieId")).collect()

        labels = labels.withColumn("tuple", concat_ws(
            ",", col("userId"), col("movieId")))

        # Collect val labels into list
        label_set = labels \
            .select('userId', 'tuple', 'rating')\
            .groupBy('userId') \
            .agg(collect_list(col("tuple")).alias("tuple"), collect_list(col("rating")).alias("rating")).collect()

        # grab arrays for calculating metrics
        precision, recall = self.precision_and_recall(movie_recs, label_set)

        return precision.sum()/len(precision), recall.sum()/len(recall)

    def precision_and_recall(self, preds, eval_data):
        """
        preds: dataFrame of predictions, do not collect list, each row has one userid and one movieid
        eval_data: validation or test set, same format as preds
        returns: new rdd with intersection of both 

        """
        # generate set to check userid, movieid tuple
        seen = set()
        out_prec = list()
        out_rec = list()

        for row in preds:
            for movieId in row[1]:
                # add tuple of userId, movieId to set
                seen.add(str(row[0])+","+str(movieId))

        for row in eval_data:
            # initialize intersection count variables
            numerator, prec_denom, rec_denom = 0, 0, 0

            # create temp list
            temp = list()

            # iterate over collected lists
            keys = row[1]
            ratings = row[2]

            for i in range(len(keys)):
                # is an element of v+
                if ratings[i] > 0:
                    rec_denom += 1
                # is an element of p intersect v
                if keys[i] in seen:
                    prec_denom += 1

                    # is and element of p intersect v+
                    if ratings[i] > 0:
                        numerator += 1

            # calculate metrics
            if prec_denom != 0:
                out_prec.append(float(numerator/prec_denom))
            if rec_denom != 0:
                out_rec.append(float(numerator/rec_denom))
        # Return np.arrays of precision and recall
        return np.array(out_prec), np.array(out_rec)

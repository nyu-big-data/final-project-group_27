import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
import time
from datetime import datetime
from pyspark.mllib.evaluation import RankingMetrics, MulticlassMetrics
from pyspark.sql import Row
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
    -----
    """

    # Constructor for Model
    def __init__(self, model_size=None, model_type=None, rank=None, maxIter=None, regParam=None,
                 num_recs=100, min_ratings=None, bias=None, positive_rating_threshold=0, k=100, sanity_check=None, save_model=0, checkpointInterval=0):
        # Model Attributes
        # NO Arg needed to be passed thorugh
        # Dictionary to access variable methods
        self.checkpointInterval = checkpointInterval
        self.netId = const.netID
        self.methods = {"als": self.alternatingLeastSquares,
                        "baseline": self.baseline}
        # Top X number of reccomendations to return - set to 100, probably won't change
        self.num_recs = num_recs
        self.k = k
        self.bias = bias
        # Passed through by user
        self.model_size = model_size
        self.model_type = model_type
        self.positive_rating_threshold = positive_rating_threshold
        self.sanity_check = sanity_check
        # For ALS
        self.rank = rank  # Rank of latent factors used in decomposition
        self.maxIter = maxIter  # Number of iterations to run algorithm, recommended 5-20
        self.regParam = regParam  # Regularization Parameter
        self.save_model = int(save_model)
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

        # Check for leakage between the sets
        if self.sanity_check:
            tester = UnitTest()
            if tester.data_leakage_check(train=train, val=evaluation_data) == False:
                raise Exception("Data Leakage Occured - Check stdout")
            else:
                print(
                    f"Passed Data Leakage Check Between Train and {self.evaluation_data_name}")

        # Grab method for whichever model corresponds to self.model_type
        model = self.methods[self.model_type]
        # Run model on training / evaluation data
        model(train, evaluation_data)
     

    # This method uses the Alternating Least Squares Pyspark Class to fit and run a model
    def alternatingLeastSquares(self, training, evaluation_data):
        """
        Builds and fits a PySpark alternatingLeastSquares latent factor model. 
        Calls self.ALS_metrics(means, userRecs, regression_precitions,evaluation_data)
        to record the results. Some dummy variables are made to record whether or not we are using the validation set
        or the testing set. This will help us record our results accurately. Training and predicting are also timed. 
        -----
        Input: 
        training: RDD - Training data set
        evaluation_data: RDD - Either Validation data set, or Training data set
        -----
        Output: 
        userRecs: PySpark DF -  containing (userId, [movie_rec_1,...,movie_rec_n]) where n = self.num_recs
        -----
        """
        # Time the function start to finish
        start = time.time()
        # Create the model with certain params - coldStartStrategy="drop" means that we'll have no nulls in val / test set
        als = ALS(maxIter=self.maxIter, rank=self.rank, regParam=self.regParam,
                  nonnegative=False, seed=10, userCol="userId",
                  itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", checkpointInterval=self.checkpointInterval,
                  numUserBlocks=50,numItemBlocks=50)

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
        end = time.time()
        self.time_to_predict = end - start

        # Get Movie / User Means from training DF
        means = training.select("movieId", "userId",
                                "movie_mean", "user_mean", "rating").alias("means")
        # Calculate Metrics
        self.ALS_metrics(means=means, userRecs=userRecs,
                         regression_predictions=regression_predictions, evaluation_data=evaluation_data)

        if self.save_model == 1:
            model.save(const.HPC_DATA_FILEPATH+f"{self.model_size}-{self.rank}-{self.maxIter}")

        # Return top self.num_recs movie recs for each user
        # return userRecs

    # Baseline model that returns top X most popular items (highest avg rating)
    def baseline(self, training, evaluation_data):
        """
        Baseline model for recommendation system. No personalization, just recommend the Top 100 movies by popularity.
        Popularity is calculated by taking avg(rating) for each movieId. Popularity has one of the two parameters, depending
        on which you pass to the model: self.min_ratings or self.bias. See self.baseline_min_ratings and self.baseline_bias
        for more details.
        input:
        -----
        training: RDD - training set data
        evaluation_data: RDD - Validation set or Test set data
        -----
        output: RDD of Top 100 movieIds by avg(rating)
        """
        # Recommends in constant time
        self.time_to_predict = 0

        # Make sure the right params have been passed to Model()
        if self.min_ratings is None and self.bias is None:
            raise Exception(
                f"Must pass through a value for self.min_ratings or self.bias for baseline to compute, you passed through {self.min_ratings}")

        # Time model Fit
        start = time.time()

        if self.min_ratings:
            top_100_movies = self.baseline_min_ratings(training)
        else:
            top_100_movies = self.baseline_bias(training)

        # Grab Distinct User Ids
        temp2 = evaluation_data
        ids = temp2.select("userId").distinct()

        # Cross Join Distinct userIds with Top 100 Most Popular Movies
        predictions = ids.crossJoin(top_100_movies)

        # Record end time after RDD operations
        end = time.time()
        self.time_to_fit = end - start

        # If sanity_checker = True then
        if self.sanity_check == True:
            tester = UnitTest()
            tester.baseline_prediction_check(preds=predictions)

        # Calculate Metrics in place
        self.Baseline_metrics(predictions=predictions,
                              evaluation_data=evaluation_data)

        # Return The top 100 most popular movies above self.min_ratings threshold
        return predictions

    def baseline_bias(self, training):
        """
        Defines top 100 most popular movies by adding in a bias term to the denominator.
        Popularity = sum(ratings) / (count(ratings) + self.bias)

        input: 
        -----
        training: PySpark DF - training data set
        -----
        output:
        top_100_movies: PySpark DF - top 100 movies by popularity
        """
        temp = training.alias("temp")
        temp = temp.groupBy("movieId").agg(sum("rating").alias(
            "sum"), (count("movieId")+int(self.bias)).alias("count"))
        top_100_movies = temp.withColumn("prediction", col("sum")/col("count"))
        top_100_movies = top_100_movies.select("movieId", "prediction").orderBy(
            "prediction", ascending=False).limit(100)
        return top_100_movies

    def baseline_min_ratings(self, training):
        """
        Defines top 100 most popular movies by enforcing a minimum amount of ratings for each movie.
        Popularity = avg(rating) where count(movie) >= self.min_reviews

        input: 
        -----
        training: PySpark DF - training data set
        -----
        output:
        top_100_movies: PySpark DF - top 100 movies by popularity
        """
        # Get Top 100 Most Popular Movies - Avg(rating) becomes prediction
        temp = training.alias("temp")
        top_100_movies = temp.groupBy("movieId").agg(avg("rating").alias(
            "prediction"), count("movieId").alias("movie_count"))
        top_100_movies = top_100_movies.where(
            col("movie_count") >= self.min_ratings)
        top_100_movies = top_100_movies.select("movieId").orderBy(
            "prediction", ascending=False).limit(100)
        return top_100_movies

    def Baseline_metrics(self, predictions, evaluation_data):
        """
        Calculates OTB Ranking Metrics, Custom Precision and Recall in place
        """
        self.OTB_ranking_metrics(
            preds=predictions, labels=evaluation_data)
        self.custom_precision(predictions=predictions,
                              eval_data=evaluation_data)
        self.custom_recall(predictions=predictions, eval_data=evaluation_data)

    def unpack_means(self, means):
        # Unpack userRecs, go from userId, list({movieId:predicted_rating}) -> userId, movieId
        # Select User and Movie Means into Seperate DFs to join - drop duplicates is necessary
        user_means = means.select("userId", "user_mean").alias("user_means")
        user_means = user_means.dropDuplicates()
        movie_means = means.select(
            "movieId", "movie_mean").alias("movie_means")
        movie_means = movie_means.dropDuplicates()
        return user_means, movie_means

    def format_user_recs(self, userRecs):
        # Get format for custom/precision / recall
        recs = userRecs.select("userId", explode(
            "recommendations").alias("tuple")).select("userId", "tuple.*")
        # Rename column for clarity
        recs = recs.withColumnRenamed("rating", "prediction")
        return recs

    def ALS_metrics(self, means, userRecs, regression_predictions, evaluation_data):
        """
        Calculates OTB Ranking Metrics, Custom Precision and Recall, and Non-Ranking Metrics in place
        """
        # Unpack Means
        user_means, movie_means = self.unpack_means(means)

        # Testing to see if aliasing helps the error
        user_means_2 = user_means.alias("user_means_2")
        movie_means_2 = movie_means.alias("movie_means_2")

        # Format recs
        preds = self.format_user_recs(userRecs)

        # Undo the normalization for both ranking predictions and regression predictions
        ranking_predictions = self.ALS_undo_normalization(
            movie_means=movie_means, user_means=user_means, preds=preds).select("movieId", "userId", "prediction")

        regression_predictions = self.ALS_undo_normalization(
            movie_means=movie_means_2, user_means=user_means_2, preds=regression_predictions).select("rating", "movieId", "userId", "prediction")

        # Use self.record_metrics to evaluate model on Precision at K, Mean Precision, and NDGC
        self.OTB_ranking_metrics(
            preds=ranking_predictions, labels=evaluation_data)

        # Calculate custom precision / recall
        self.custom_precision(
            predictions=ranking_predictions, eval_data=evaluation_data)
        self.custom_recall(predictions=ranking_predictions,
                           eval_data=evaluation_data)

        # Use self.non_ranking_metrics to compute RMSE, R^2, and ROC of Top 100 Predictions - No special Filtering ATM
        self.non_ranking_metrics(regression_predictions)

    def ALS_undo_normalization(self, user_means, movie_means, preds):
        # Join user_means first, then movie_means
        preds = preds.join(user_means, "userId", how='left').select(
            preds['*'], user_means.user_mean)
        preds = preds.join(movie_means, "movieId", how='left').select(
            preds['*'], movie_means.movie_mean)

        # For those that don't appear in the predicted movies just have 0
        preds = preds.withColumn("user_mean", when(
            col("user_mean").isNull(), 0).otherwise(col("user_mean")))
        preds = preds.withColumn("movie_mean", when(
            col("movie_mean").isNull(), 0).otherwise(col("movie_mean")))

        # Fix the predictions, un-normalize the predicted ratings score
        # Then subtract 2.5 to stay consistent with boolean logic in metric functions
        fixed_predictions = preds.withColumn("prediction", col(
            "prediction") - 2.5 + (.5 * (col("movie_mean")+col("user_mean"))))

        return fixed_predictions

    # Non-Ranking Metrics Calculated Here
    def non_ranking_metrics(self, predictions):
        """
        Takes in a predictions rdd as input, saves rmse, r2, weighted recall, precision and F1 to self.metrics
        """
        ##Evaluate Predictions for Regression Task##
        evaluator = RegressionEvaluator(
            labelCol="rating", predictionCol="prediction")
        # Calculate RMSE and r_2 metrics and append to metrics
        self.metrics['RMSE'] = evaluator.evaluate(
            predictions, {evaluator.metricName: "rmse"})
        self.metrics['R2'] = evaluator.evaluate(
            predictions, {evaluator.metricName: "r2"})

    def OTB_ranking_metrics(self, preds, labels):
        """
        Calculates MAP, Precistion at K, Recall at K, NDGC at K in place and saves outputs
        to self.metrics["metric_name"]
        Input: 
        preds: DF - Tall DF of userId, movieId predictions
        labels: DF - Validation or Testing Data Split
        """
        print("Running OTB Ranking Metrics")

        # Define window spec to grab top 100 reviews - use movieId as tie breaker (especially for baseline)
        windowSpec = Window.partitionBy('userId').orderBy(
            col('prediction').desc(), col("movieId"))
        predictions = preds \
            .select('userId', 'movieId', 'prediction', rank().over(windowSpec).alias('rank')) \
            .where(f'rank <= {self.num_recs}') \
            .groupBy('userId') \
            .agg(expr('collect_list(movieId) as movieId'))

        labels = labels \
            .select('userId', 'movieId')\
            .groupBy('userId') \
            .agg(expr('collect_list(movieId) as movieId'))

        print("Creating Predictions and Labels")
        predictionsAndLabels = predictions.join(broadcast(labels), 'userId', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        rankingMetrics = RankingMetrics(predictionsAndLabels)

        print("Calculating MAP")
        # Update Metrics
        self.metrics['MAP'] = rankingMetrics.meanAveragePrecision

        print("Calculating Precision")
        self.metrics[f'precisionAt{self.num_recs}'] = rankingMetrics.precisionAt(
            self.num_recs)
        print("Calculating Recall")
        self.metrics[f'recallAt{self.num_recs}'] = rankingMetrics.recallAt(
            self.num_recs)
        print("Calculating NDGC")
        self.metrics[f'ndgcAt{self.num_recs}'] = rankingMetrics.ndcgAt(
            self.num_recs)

        # For Test Set
        # self.metrics[f'precisionAt10'] = rankingMetrics.precisionAt(10)
        # self.metrics[f'recallAt10'] = rankingMetrics.recallAt(10)
        # self.metrics[f'ndgcAt10'] = rankingMetrics.ndcgAt(10)

        # self.metrics[f'precisionAt25'] = rankingMetrics.precisionAt(25)
        # self.metrics[f'recallAt25'] = rankingMetrics.recallAt(25)
        # self.metrics[f'ndgcAt25'] = rankingMetrics.ndcgAt(25)

    def custom_precision(self, predictions, eval_data) -> None:
        """
        Function to calculate accuracy -> TP / (TP + FP), saved in place
        True positives are movies we predicted they would like and they appeared in their evaluation set
        and the user rated them positive. False Positives are movies that we predicted they liked, and they
        watched them in their evaluation set, but did not rate them positively.
        -----
        input:
        -----
        predictions: PySpark DF of movie predictions for baseline model
        eval_data: PySpark DF of evaluation data
        -----
        output:
        -----
        float: accuracy calculated as TP / (TP + FP)
        """
        # Make our predictions equal to 1, as all movies are guessed as positive with baseline
        if self.model_type == 'baseline':
            predictions = predictions.withColumn(
                "prediction", lit(1).cast("double"))
        else:
            # Otherwise if we're doing something like an ALS model then we just binarize
            predictions = predictions.withColumn("prediction", when(
                predictions.prediction > 0, 1).otherwise(0).cast("double"))

        # Set up join by aliasing and setting a join conidtion
        preds = predictions.alias("preds")
        labels = eval_data.alias("labels")
        cond = [(labels.movieId == preds.movieId)
                & (labels.userId == preds.userId)]

        # Join
        intersection = labels.join(preds, cond, how='inner').select(
            labels.userId, labels.movieId, preds.prediction, labels.rating)

        # Binarize rating to positive / negative reviews -- make double
        intersection = intersection.withColumn(
            "rating", when(col("rating") > 0, 1).otherwise(0).cast("double"))

        # Sum rating column (this gives us TP), sum predicted column (gives us TP + FP)
        intersection = intersection.groupBy("userId").agg(
            sum(col("rating")).alias("TP"), sum(col("prediction")).alias("TPandFP"))

        # Calculate precision
        intersection = intersection.withColumn(
            "precision", col("TP")/col("TPandFP"))
        # Return mean accuracy across all userIds
        self.metrics["Custom Precision"] = intersection.select(
            "precision").agg({"precision": "avg"}).collect()[0][0]

    def custom_recall(self, predictions, eval_data) -> None:
        """
        Custom function to calculate model recall  (TP / TP + FN) in place. 
        True Positives (TP) are movies that we predicted the user would like
        and they actually watched and liked it. False Negatives (FN) are movies we did not recommend
        in the case of the baseline, or movies we predicted they would not like (for ALS) that the user ended up
        watching in their evaluation set and rated positively.
        -----
        input:
        -----
        predictions: PySpark DF of movie predictions for baseline model
        eval_data: PySpark DF of evaluation data
        """
        # Make our predictions equal to 1, as all movies are guessed as positive with baseline
        if self.model_type == 'baseline':
            predictions = predictions.withColumn(
                "prediction", lit(1).cast("double"))
        else:
            # Otherwise if we're doing something like an ALS model then we just binarize
            predictions = predictions.withColumn("prediction", when(
                predictions.prediction > 0, 1).otherwise(0).cast("double"))

        # Set up Join
        preds = predictions.alias("preds")
        labels = eval_data.alias("labels")
        labels = labels.filter(col("rating") > 0)
        cond = [(labels.movieId == preds.movieId)
                & (labels.userId == preds.userId)]

        # Join - use left here so rows from labels are included
        intersection = labels.join(preds, cond, how='left').select(
            labels.userId, labels.movieId, preds.prediction, labels.rating)

        # Make movies we didn't have a prediction for be 0 instead of null (these are False Negatives)
        intersection = intersection.withColumn("prediction", when(
            col("prediction").isNull(), 0).otherwise(col("prediction")))

        # Binarize rating to positive / negative reviews -- make double
        intersection = intersection.withColumn(
            "rating", when(col("rating") > 0, 1).otherwise(0).cast("double"))

        # Sum rating column (this gives us TP), sum predicted column (gives us TP + FP)
        intersection = intersection.groupBy("userId").agg(
            sum(col("rating")).alias("TPandFN"), sum(col("prediction")).alias("TP"))

        # Calculate recall
        intersection = intersection.withColumn(
            "recall", col("TP")/col("TPandFN"))

        # Return mean recall across all userIds
        self.metrics["Custom Recall"] = intersection.select(
            "recall").agg({"recall": "avg"}).collect()[0][0]
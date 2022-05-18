import pyspark
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import *

class Util():
    def __init__(self, spark, preds, val):
        self.preds = preds
        self.val = val
        self.metrics = {}
        self.num_recs = 100
        self.spark = spark
    def run_metrics(self):
        """
        Method to run all metrics - OTB PySpark Ranking metrics and Custom Precision / Recall
        """
        #self.format_preds()
        self.OTB_ranking_metrics()
        self.custom_precision()
        self.custom_recall()
        
    
    def format_preds(self):
        """
        Take self.preds, and make them into an RDD then DF with correct column names
        """
        columns = ["id","recs"]
        self.preds = self.spark.sparkContext.parallelize(self.preds).toDF(columns)
        
    def OTB_ranking_metrics(self):
        """
        Calculates MAP, Precistion at K, Recall at K, NDGC at K in place and saves outputs
        to self.metrics["metric_name"]
        Input: 
        preds: DF - Tall DF of userId, movieId predictions
        labels: DF - Validation or Testing Data Split
        """
        print("Running OTB Ranking Metrics")

        labels = self.val \
            .select('id', 'movieId')\
            .groupBy('id') \
            .agg(expr('collect_list(movieId) as movieId'))

        predictionsAndLabels = self.preds.join(labels, 'id', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

        rankingMetrics = RankingMetrics(predictionsAndLabels)

        # Update Metrics
        self.metrics['MAP'] = rankingMetrics.meanAveragePrecision
        self.metrics[f'precisionAt{self.num_recs}'] = rankingMetrics.precisionAt(self.num_recs)
        self.metrics[f'recallAt{self.num_recs}'] = rankingMetrics.recallAt(self.num_recs)
        self.metrics[f'ndgcAt{self.num_recs}'] = rankingMetrics.ndcgAt(self.num_recs)

    def custom_precision(self) -> None:
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
        predictions = self.preds.select("id",explode("recs").alias("movieId"))
        predictions = predictions.withColumn("prediction", lit(1).cast("double"))

    
        # Set up join by aliasing and setting a join conidtion
        preds = predictions.alias("preds")
        labels = self.val.alias("labels")
        cond = [(labels.movieId == preds.movieId)
                & (labels.id == preds.id)]

        # Join
        intersection = labels.join(preds, cond, how='inner').select(
            labels.id, labels.movieId, preds.prediction, labels.rating)

        # Binarize rating to positive / negative reviews -- make double
        intersection = intersection.withColumn(
            "rating", when(col("rating") > 0, 1).otherwise(0).cast("double"))

        # Sum rating column (this gives us TP), sum predicted column (gives us TP + FP)
        intersection = intersection.groupBy("id").agg(
            sum(col("rating")).alias("TP"), sum(col("prediction")).alias("TPandFP"))

        # Calculate precision
        intersection = intersection.withColumn(
            "precision", col("TP")/col("TPandFP"))
        # Return mean accuracy across all userIds
        self.metrics["Custom Precision"] = intersection.select(
            "precision").agg({"precision": "avg"}).collect()[0][0]

    def custom_recall(self) -> None:
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
        predictions = self.preds.select("id",explode("recs").alias("movieId"))
        predictions = predictions.withColumn("prediction", lit(1).cast("double"))
        
        # Set up Join
        preds = predictions.alias("preds")
        labels = self.val.alias("labels")
        labels = labels.filter(col("rating") > 0)
        cond = [(labels.movieId == preds.movieId)
                & (labels.id == preds.id)]

        # Join - use left here so rows from labels are included
        intersection = labels.join(preds, cond, how='left').select(
            labels.id, labels.movieId, preds.prediction, labels.rating)

        # Make movies we didn't have a prediction for be 0 instead of null (these are False Negatives)
        intersection = intersection.withColumn("prediction", when(
            col("prediction").isNull(), 0).otherwise(col("prediction")))

        # Binarize rating to positive / negative reviews -- make double
        intersection = intersection.withColumn(
            "rating", when(col("rating") > 0, 1).otherwise(0).cast("double"))
     
        # Sum rating column (this gives us TP), sum predicted column (gives us TP + FP)
        intersection = intersection.groupBy("id").agg(
            sum(col("rating")).alias("TPandFN"), sum(col("prediction")).alias("TP"))

        # Calculate recall
        intersection = intersection.withColumn(
            "recall", col("TP")/col("TPandFN"))

        # Return mean recall across all userIds
        self.metrics["Custom Recall"] = intersection.select(
            "recall").agg({"recall": "avg"}).collect()[0][0]
import pandas as pd
import numpy as np
import pyspark
import pyspark
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RankingMetrics

def evaluate_metrics(predictsAndLabels):

    #Make RDD from Prediction and Labels
    rdd = pyspark.SparkContext.parallelize(predictsAndLabels)
    metrics = RankingMetrics(rdd)

    #Need to understand this better
    return metrics.meanAveragePrecision, metrics.ndcgAt(100)